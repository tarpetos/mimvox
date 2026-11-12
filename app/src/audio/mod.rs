//! Microphone capture and playback (cpal + hound). Event-driven, never blocks the UI thread: the capture callback
//! reports the input level at a capped rate ([`METER_HZ`]) to a [`MeterSink`], the playback callback reports when the
//! queue runs dry to a [`DrainSink`]. Nothing has to be polled.
//!
//! cpal streams are not `Send`, so each `Recorder` / `Player` owns a small worker thread that creates and holds the
//! stream; the public handles only share atomics and short-lived mutexes with it (so they are `Send + Sync`).
//! Without audio devices (e.g. WSL) everything returns `Err` with a readable message instead of panicking.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use anyhow::{Context, anyhow};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

pub struct RecInfo {
    pub path: PathBuf,
    pub duration_s: f32,
    pub peak: f32,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn load_f32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

/// Stores `v` if it is larger than the current value (peak hold). Single writer per atomic is not guaranteed,
/// so this uses a CAS loop.
fn fetch_max_f32(a: &AtomicU32, v: f32) {
    let mut cur = a.load(Ordering::Relaxed);
    while v > f32::from_bits(cur) {
        match a.compare_exchange_weak(cur, v.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------------------------------------------

/// Names of the available input devices; the first is the system default.
pub fn input_devices() -> Vec<String> {
    let host = cpal::default_host();
    let default = host.default_input_device().and_then(|d| d.name().ok());
    let mut names: Vec<String> = host
        .input_devices()
        .map(|it| it.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default();
    if let Some(def) = default {
        names.retain(|n| *n != def);
        names.insert(0, def);
    }
    names.dedup();
    names
}

fn pick_input(device: Option<&str>) -> anyhow::Result<cpal::Device> {
    let host = cpal::default_host();
    if let Some(wanted) = device {
        let found = host
            .input_devices()
            .map_err(|e| anyhow!("cannot list input devices: {e}"))?
            .find(|d| d.name().map(|n| n == wanted).unwrap_or(false));
        if let Some(d) = found {
            return Ok(d);
        }
        // Fall through to the default if the saved device is gone.
    }
    host.default_input_device()
        .ok_or_else(|| anyhow!("no microphone found: connect an input device and check the system sound settings"))
}

// ---------------------------------------------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------------------------------------------

/// Level reports per second from a running capture (the meter's redraw rate).
pub const METER_HZ: u32 = 25;

/// What a running capture reports, from the audio thread.
#[derive(Clone, Debug)]
pub enum CaptureEvent {
    /// About [`METER_HZ`] times a second: the meter level 0..1 (peak with a 0.25 s decay), the highest sample so far,
    /// and the seconds captured.
    Level { level: f32, peak: f32, elapsed_s: f32 },
    /// The backend reported a stream error (e.g. the microphone was unplugged).
    Error(String),
}

/// Receives [`CaptureEvent`]s on the audio thread: keep it short (hand the value on, see `ui::events::Latest`).
pub type MeterSink = Arc<dyn Fn(CaptureEvent) + Send + Sync>;

/// Called on the audio thread when the playback queue has played out (it held audio and is now empty).
pub type DrainSink = Arc<dyn Fn() + Send + Sync>;

struct RecShared {
    /// Mono samples at the device rate.
    buffer: Mutex<Vec<f32>>,
    /// Max |sample| since the start (f32 bits).
    peak: AtomicU32,
    samples: AtomicU64,
    error: Mutex<Option<String>>,
}

pub struct Recorder {
    shared: Arc<RecShared>,
    sample_rate: u32,
    device_name: String,
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

fn build_input<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: Arc<RecShared>,
    meter: Option<MeterSink>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels.max(1) as usize;
    let rate = config.sample_rate.0.max(1);
    let err_shared = shared.clone();
    let err_meter = meter.clone();
    let mut scratch: Vec<f32> = Vec::new();
    // Meter state, counted in samples (the audio clock): the peak since the last report and the shown level.
    let every = (rate / METER_HZ).max(1) as u64;
    let (mut since, mut block_max, mut shown) = (0u64, 0.0f32, 0.0f32);
    device.build_input_stream(
        config,
        move |data: &[T], _| {
            scratch.clear();
            let mut block_peak = 0.0f32;
            for frame in data.chunks(channels) {
                let mut sum = 0.0f32;
                for s in frame {
                    sum += <f32 as FromSample<T>>::from_sample_(*s);
                }
                let mono = sum / frame.len() as f32;
                block_peak = block_peak.max(mono.abs());
                scratch.push(mono);
            }
            lock(&shared.buffer).extend_from_slice(&scratch);
            let total = shared.samples.fetch_add(scratch.len() as u64, Ordering::Relaxed) + scratch.len() as u64;
            fetch_max_f32(&shared.peak, block_peak);
            if let Some(sink) = &meter {
                since += scratch.len() as u64;
                block_max = block_max.max(block_peak);
                if since >= every {
                    let dt = since as f32 / rate as f32;
                    shown = block_max.max(shown * (-dt / 0.25).exp()).min(1.0);
                    sink(CaptureEvent::Level { level: shown, peak: load_f32(&shared.peak), elapsed_s: total as f32 / rate as f32 });
                    since = 0;
                    block_max = 0.0;
                }
            }
        },
        move |err| {
            *lock(&err_shared.error) = Some(err.to_string());
            if let Some(sink) = &err_meter {
                sink(CaptureEvent::Error(err.to_string()));
            }
        },
        None,
    )
}

impl Recorder {
    /// Starts capturing from `device` (None = system default), mono. `meter` receives the live level (see
    /// [`CaptureEvent`]).
    pub fn start(device: Option<&str>, meter: Option<MeterSink>) -> anyhow::Result<Self> {
        let shared = Arc::new(RecShared {
            buffer: Mutex::new(Vec::with_capacity(48_000 * 120)),
            peak: AtomicU32::new(0),
            samples: AtomicU64::new(0),
            error: Mutex::new(None),
        });
        let (init_tx, init_rx) = mpsc::channel::<Result<(u32, String), String>>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let device_wanted = device.map(str::to_string);
        let thread_shared = shared.clone();

        let thread = thread::Builder::new()
            .name("audio-capture".into())
            .spawn(move || {
                let setup = (|| -> anyhow::Result<(cpal::Stream, u32, String)> {
                    let dev = pick_input(device_wanted.as_deref())?;
                    let name = dev.name().unwrap_or_else(|_| "unknown device".into());
                    let supported = dev
                        .default_input_config()
                        .map_err(|e| anyhow!("cannot open the microphone \"{name}\": {e}"))?;
                    let rate = supported.sample_rate().0;
                    let format = supported.sample_format();
                    let config: cpal::StreamConfig = supported.into();
                    let stream = match format {
                        SampleFormat::F32 => build_input::<f32>(&dev, &config, thread_shared, meter),
                        SampleFormat::I16 => build_input::<i16>(&dev, &config, thread_shared, meter),
                        SampleFormat::U16 => build_input::<u16>(&dev, &config, thread_shared, meter),
                        SampleFormat::I32 => build_input::<i32>(&dev, &config, thread_shared, meter),
                        SampleFormat::I8 => build_input::<i8>(&dev, &config, thread_shared, meter),
                        SampleFormat::U8 => build_input::<u8>(&dev, &config, thread_shared, meter),
                        SampleFormat::F64 => build_input::<f64>(&dev, &config, thread_shared, meter),
                        other => return Err(anyhow!("unsupported microphone sample format {other:?}")),
                    }
                    .map_err(|e| anyhow!("cannot start capture on \"{name}\": {e}"))?;
                    stream
                        .play()
                        .map_err(|e| anyhow!("cannot start capture on \"{name}\": {e}"))?;
                    Ok((stream, rate, name))
                })();
                match setup {
                    Ok((stream, rate, name)) => {
                        let _ = init_tx.send(Ok((rate, name)));
                        let _ = stop_rx.recv(); // returns on stop() or when the Recorder is dropped
                        drop(stream);
                    }
                    Err(e) => {
                        let _ = init_tx.send(Err(format!("{e:#}")));
                    }
                }
            })
            .context("cannot start the audio capture thread")?;

        match init_rx.recv() {
            Ok(Ok((sample_rate, device_name))) => Ok(Self {
                shared,
                sample_rate,
                device_name,
                stop: Some(stop_tx),
                thread: Some(thread),
            }),
            Ok(Err(msg)) => {
                let _ = thread.join();
                Err(anyhow!(msg))
            }
            Err(_) => Err(anyhow!("audio capture thread ended unexpectedly")),
        }
    }

    /// Highest sample so far 0..1 (clipping warning when >= 0.99).
    pub fn peak(&self) -> f32 {
        load_f32(&self.shared.peak)
    }

    pub fn elapsed_s(&self) -> f32 {
        self.shared.samples.load(Ordering::Relaxed) as f32 / self.sample_rate as f32
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// A stream error reported by the audio backend (e.g. the microphone was unplugged), if any.
    pub fn error(&self) -> Option<String> {
        lock(&self.shared.error).clone()
    }

    fn halt(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }

    /// Stops and writes a mono 16-bit WAV at the device rate.
    pub fn stop(mut self, path: &Path) -> anyhow::Result<RecInfo> {
        self.halt();
        let samples = std::mem::take(&mut *lock(&self.shared.buffer));
        if samples.is_empty() {
            return Err(match self.error() {
                Some(e) => anyhow!("nothing was recorded: {e}"),
                None => anyhow!("nothing was recorded: the microphone delivered no audio"),
            });
        }
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
        }
        write_wav(path, &samples, self.sample_rate)?;
        let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs())).min(1.0);
        Ok(RecInfo { path: path.to_path_buf(), duration_s: samples.len() as f32 / self.sample_rate as f32, peak })
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.halt();
    }
}

/// Writes mono 16-bit PCM.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).with_context(|| format!("cannot write {}", path.display()))?;
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    w.finalize()?;
    Ok(())
}

/// Reads any PCM/float WAV as mono f32 (channels averaged). Returns the samples and the file's sample rate.
pub fn read_wav_mono(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample.clamp(1, 32) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    let mono = if channels == 1 {
        interleaved
    } else {
        interleaved.chunks(channels).map(|f| f.iter().sum::<f32>() / f.len() as f32).collect()
    };
    Ok((mono, spec.sample_rate))
}

/// Linear-interpolation resampler (adequate for speech playback).
pub fn resample_linear(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let out_len = ((input.len() as f64) / ratio).round().max(1.0) as usize;
    let last = input.len() - 1;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let idx = (pos.floor() as usize).min(last);
            let frac = (pos - idx as f64) as f32;
            let a = input[idx];
            let b = input[(idx + 1).min(last)];
            a + (b - a) * frac
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Queue {
    chunks: VecDeque<Vec<f32>>,
    pos: usize,
}

impl Queue {
    fn next_sample(&mut self) -> Option<f32> {
        let front = self.chunks.front()?;
        let s = front[self.pos];
        self.pos += 1;
        if self.pos >= front.len() {
            self.chunks.pop_front();
            self.pos = 0;
        }
        Some(s)
    }

    fn remaining(&self) -> usize {
        self.chunks.iter().map(Vec::len).sum::<usize>().saturating_sub(self.pos)
    }

    fn clear(&mut self) {
        self.chunks.clear();
        self.pos = 0;
    }
}

struct PlayShared {
    queue: Mutex<Queue>,
    /// Frames played since the last `play()` / `stop()`.
    played: AtomicU64,
    error: Mutex<Option<String>>,
}

pub struct Player {
    shared: Arc<PlayShared>,
    sample_rate: u32,
    device_name: String,
    stop: Option<Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

fn build_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    shared: Arc<PlayShared>,
    drained: Option<DrainSink>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample + FromSample<f32>,
{
    let channels = config.channels.max(1) as usize;
    let err_shared = shared.clone();
    device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            let mut q = lock(&shared.queue);
            let had_audio = !q.chunks.is_empty();
            let mut consumed = 0u64;
            for frame in data.chunks_mut(channels) {
                let s = match q.next_sample() {
                    Some(s) => {
                        consumed += 1;
                        s
                    }
                    None => 0.0,
                };
                let v = <T as FromSample<f32>>::from_sample_(s);
                for o in frame {
                    *o = v;
                }
            }
            let ran_dry = had_audio && q.chunks.is_empty();
            drop(q);
            if consumed > 0 {
                shared.played.fetch_add(consumed, Ordering::Relaxed);
            }
            if ran_dry {
                if let Some(sink) = &drained {
                    sink();
                }
            }
        },
        move |err| {
            *lock(&err_shared.error) = Some(err.to_string());
        },
        None,
    )
}

impl Player {
    /// Opens the default output. `drained` is told every time the queue plays out.
    pub fn new(drained: Option<DrainSink>) -> anyhow::Result<Self> {
        let shared = Arc::new(PlayShared {
            queue: Mutex::new(Queue::default()),
            played: AtomicU64::new(0),
            error: Mutex::new(None),
        });
        let (init_tx, init_rx) = mpsc::channel::<Result<(u32, String), String>>();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let thread_shared = shared.clone();

        let thread = thread::Builder::new()
            .name("audio-playback".into())
            .spawn(move || {
                let setup = (|| -> anyhow::Result<(cpal::Stream, u32, String)> {
                    let dev = cpal::default_host()
                        .default_output_device()
                        .ok_or_else(|| anyhow!("no audio output device found: check the system sound settings"))?;
                    let name = dev.name().unwrap_or_else(|_| "unknown device".into());
                    let supported = dev
                        .default_output_config()
                        .map_err(|e| anyhow!("cannot open the audio output \"{name}\": {e}"))?;
                    let rate = supported.sample_rate().0;
                    let format = supported.sample_format();
                    let config: cpal::StreamConfig = supported.into();
                    let stream = match format {
                        SampleFormat::F32 => build_output::<f32>(&dev, &config, thread_shared, drained),
                        SampleFormat::I16 => build_output::<i16>(&dev, &config, thread_shared, drained),
                        SampleFormat::U16 => build_output::<u16>(&dev, &config, thread_shared, drained),
                        SampleFormat::I32 => build_output::<i32>(&dev, &config, thread_shared, drained),
                        SampleFormat::I8 => build_output::<i8>(&dev, &config, thread_shared, drained),
                        SampleFormat::U8 => build_output::<u8>(&dev, &config, thread_shared, drained),
                        SampleFormat::F64 => build_output::<f64>(&dev, &config, thread_shared, drained),
                        other => return Err(anyhow!("unsupported output sample format {other:?}")),
                    }
                    .map_err(|e| anyhow!("cannot start playback on \"{name}\": {e}"))?;
                    stream
                        .play()
                        .map_err(|e| anyhow!("cannot start playback on \"{name}\": {e}"))?;
                    Ok((stream, rate, name))
                })();
                match setup {
                    Ok((stream, rate, name)) => {
                        let _ = init_tx.send(Ok((rate, name)));
                        let _ = stop_rx.recv();
                        drop(stream);
                    }
                    Err(e) => {
                        let _ = init_tx.send(Err(format!("{e:#}")));
                    }
                }
            })
            .context("cannot start the audio playback thread")?;

        match init_rx.recv() {
            Ok(Ok((sample_rate, device_name))) => {
                Ok(Self { shared, sample_rate, device_name, stop: Some(stop_tx), thread: Some(thread) })
            }
            Ok(Err(msg)) => {
                let _ = thread.join();
                Err(anyhow!(msg))
            }
            Err(_) => Err(anyhow!("audio playback thread ended unexpectedly")),
        }
    }

    /// Stops whatever plays and plays `path`.
    pub fn play(&self, path: &Path) -> anyhow::Result<()> {
        let samples = self.load(path)?;
        let mut q = lock(&self.shared.queue);
        q.clear();
        self.shared.played.store(0, Ordering::Relaxed);
        q.chunks.push_back(samples);
        Ok(())
    }

    /// Appends `path` to the queue (gapless streaming of tts.chunk files).
    pub fn enqueue(&self, path: &Path) -> anyhow::Result<()> {
        let samples = self.load(path)?;
        lock(&self.shared.queue).chunks.push_back(samples);
        Ok(())
    }

    /// Decodes and resamples to the device rate before the queue lock is taken.
    fn load(&self, path: &Path) -> anyhow::Result<Vec<f32>> {
        let (mono, rate) = read_wav_mono(path)?;
        let out = resample_linear(&mono, rate, self.sample_rate);
        if out.is_empty() {
            return Err(anyhow!("{} contains no audio", path.display()));
        }
        Ok(out)
    }

    pub fn stop(&self) {
        lock(&self.shared.queue).clear();
        self.shared.played.store(0, Ordering::Relaxed);
    }

    pub fn is_playing(&self) -> bool {
        !lock(&self.shared.queue).chunks.is_empty()
    }

    /// Seconds played since the last `play()` / `stop()` (keeps counting across enqueued chunks).
    pub fn position_s(&self) -> f32 {
        self.shared.played.load(Ordering::Relaxed) as f32 / self.sample_rate as f32
    }

    /// Seconds still queued.
    pub fn queued_s(&self) -> f32 {
        lock(&self.shared.queue).remaining() as f32 / self.sample_rate as f32
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn error(&self) -> Option<String> {
        lock(&self.shared.error).clone()
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        if let Some(tx) = self.stop.take() {
            let _ = tx.send(());
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
