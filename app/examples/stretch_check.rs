//! Offline check of the player's pitch-preserving speed change (`src/audio/stretch.rs`, WSOLA).
//!
//! `cargo run --example stretch_check [speech.wav] [out_dir]`
//!
//! For a harmonic test tone (200 Hz fundamental) and, when given, a speech WAV, at 0.5x / 0.75x / 1x / 1.25x / 1.5x / 2x:
//! * the length is the source length divided by the speed (within 2 %);
//! * the pitch is kept: the dominant frequency of the tone, and the median fundamental of the voiced speech frames,
//!   stay within 2 % of the source's;
//! * at 1x the output equals the source (apart from the 16 ms fade-in and fade-out at the ends);
//! * changing the speed while it plays keeps the waveform continuous (the tone's largest sample step stays put);
//! * the time it takes per second of audio (the playback callback runs it in real time).
//!
//! With `out_dir`, every result is written as a WAV there to listen to. Exits non-zero when a check fails.

#[path = "../src/audio/stretch.rs"]
mod stretch;

use std::f32::consts::PI;
use std::path::Path;
use std::time::Instant;

const SPEEDS: [f32; 6] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let speech = args.first().map(String::as_str);
    let out_dir = args.get(1).map(Path::new);
    let mut failed = 0;

    let rate = 48_000;
    let tone: Vec<f32> = (0..rate * 3)
        .map(|i| {
            let t = i as f32 / rate as f32;
            0.5 * (2.0 * PI * 200.0 * t).sin() + 0.25 * (2.0 * PI * 400.0 * t).sin() + 0.12 * (2.0 * PI * 600.0 * t).sin()
        })
        .collect();
    failed += check("tone 200 Hz", &tone, rate, Measure::Dominant, out_dir);
    failed += switching(&tone, rate);

    if let Some(path) = speech {
        match read_wav(Path::new(path)) {
            Ok((x, r)) => failed += check(path, &x, r, Measure::F0, out_dir),
            Err(e) => {
                eprintln!("cannot read {path}: {e}");
                failed += 1;
            }
        }
    }
    if failed > 0 {
        eprintln!("{failed} check(s) failed");
        std::process::exit(1);
    }
    println!("all checks passed");
}

#[derive(Clone, Copy)]
enum Measure {
    /// Strongest spectral peak (a steady tone).
    Dominant,
    /// Median autocorrelation fundamental of the voiced frames (speech).
    F0,
}

fn check(name: &str, src: &[f32], rate: u32, m: Measure, out_dir: Option<&Path>) -> u32 {
    let mut failed = 0;
    let base = pitch(src, rate, m);
    println!("{name}: {:.2} s at {rate} Hz, pitch {base:.1} Hz", src.len() as f32 / rate as f32);
    for speed in SPEEDS {
        let t0 = Instant::now();
        let out = stretch::stretch_all(src, rate, speed);
        let cost = t0.elapsed().as_secs_f32() / (out.len() as f32 / rate as f32);
        let want = src.len() as f32 / speed;
        let len_err = (out.len() as f32 - want) / want;
        let p = pitch(&out, rate, m);
        let p_err = (p - base) / base;
        let mut line = format!(
            "  {speed:>4}x  length {:.3} s (expected {:.3}, {:+.2} %)  pitch {p:.1} Hz ({:+.2} %)  {:.1} ms per audio second",
            out.len() as f32 / rate as f32,
            want / rate as f32,
            len_err * 100.0,
            p_err * 100.0,
            cost * 1000.0
        );
        let mut ok = len_err.abs() <= 0.02 && p_err.abs() <= 0.02;
        if speed == 1.0 {
            let edge = (rate as f32 * 0.016) as usize;
            let n = src.len().min(out.len()).saturating_sub(edge);
            let max_diff = (edge..n).map(|i| (src[i] - out[i]).abs()).fold(0.0f32, f32::max);
            line.push_str(&format!("  max |out - src| {max_diff:.2e}"));
            ok &= max_diff < 1e-4;
        }
        println!("{line}  {}", if ok { "ok" } else { "FAIL" });
        if !ok {
            failed += 1;
        }
        if let Some(dir) = out_dir {
            let stem = Path::new(name).file_stem().and_then(|s| s.to_str()).unwrap_or("clip").replace(' ', "_");
            let path = dir.join(format!("{stem}_{speed}x.wav"));
            if let Err(e) = write_wav(&path, &out, rate) {
                eprintln!("cannot write {}: {e}", path.display());
            }
        }
    }
    failed
}

/// Changing the speed while playing (every 0.1 s, through all speeds, in 10 ms callback-sized blocks) must not
/// break the waveform: on the tone, the largest step between two output samples stays near the source's.
fn switching(src: &[f32], rate: u32) -> u32 {
    let mut st = stretch::Stretcher::new(rate);
    let block = (rate / 100) as usize;
    let mut buf = vec![0.0f32; block];
    let mut out = Vec::new();
    for k in 0.. {
        let speed = SPEEDS[(k / 10) % SPEEDS.len()];
        let (n, flow) = st.process(src, speed, true, &mut buf);
        out.extend_from_slice(&buf[..n]);
        if flow == stretch::Flow::End {
            break;
        }
    }
    let step = |x: &[f32]| x.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
    let (a, b) = (step(src), step(&out));
    let ok = b <= a * 1.3;
    println!("  speed changes every 0.1 s: largest sample step {b:.4} (source {a:.4})  {}", if ok { "ok" } else { "FAIL" });
    u32::from(!ok)
}

fn pitch(x: &[f32], rate: u32, m: Measure) -> f32 {
    match m {
        Measure::Dominant => dominant(x, rate),
        Measure::F0 => median_f0(x, rate),
    }
}

/// The strongest frequency between 60 and 1000 Hz (Goertzel, 0.5 Hz steps, Hann window) over the middle second.
fn dominant(x: &[f32], rate: u32) -> f32 {
    let n = (rate as usize).min(x.len());
    let start = (x.len() - n) / 2;
    let seg: Vec<f32> = (0..n).map(|i| x[start + i] * (0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())).collect();
    let (mut best, mut best_p) = (0.0f32, 0.0f64);
    let mut f = 60.0f32;
    while f <= 1000.0 {
        let w = 2.0 * std::f64::consts::PI * f64::from(f) / f64::from(rate);
        let c = 2.0 * w.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for v in &seg {
            let s0 = f64::from(*v) + c * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        let power = s1 * s1 + s2 * s2 - c * s1 * s2;
        if power > best_p {
            best_p = power;
            best = f;
        }
        f += 0.5;
    }
    best
}

/// Median fundamental (70..400 Hz) of 40 ms frames whose normalised autocorrelation peak is above 0.6 (voiced),
/// with parabolic interpolation of the peak lag.
fn median_f0(x: &[f32], rate: u32) -> f32 {
    let frame = (rate as f32 * 0.04) as usize;
    let hop = frame / 2;
    let (lo, hi) = ((rate / 400) as usize, (rate / 70) as usize);
    let rms_all = (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
    let mut f0s = Vec::new();
    let mut i = 0;
    while i + frame + hi < x.len() {
        let a = &x[i..i + frame];
        let energy: f32 = a.iter().map(|v| v * v).sum();
        if (energy / frame as f32).sqrt() > 0.5 * rms_all {
            let corr = |lag: usize| -> f32 {
                let b = &x[i + lag..i + lag + frame];
                let xy: f32 = a.iter().zip(b).map(|(p, q)| p * q).sum();
                let yy: f32 = b.iter().map(|v| v * v).sum();
                xy / (energy * yy).sqrt().max(1e-9)
            };
            let cs: Vec<f32> = (lo..=hi + 1).map(corr).collect();
            if let Some((k, &c)) = cs.iter().enumerate().take(cs.len() - 1).skip(1).max_by(|p, q| p.1.total_cmp(q.1)) {
                if c > 0.6 {
                    let (y0, y2) = (cs[k - 1], cs[k + 1]);
                    let d = 0.5 * (y0 - y2) / (y0 - 2.0 * c + y2).min(-1e-9);
                    f0s.push(rate as f32 / ((lo + k) as f32 + d));
                }
            }
        }
        i += hop;
    }
    if f0s.is_empty() {
        return 0.0;
    }
    f0s.sort_by(f32::total_cmp);
    f0s[f0s.len() / 2]
}

fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), hound::Error> {
    let mut r = hound::WavReader::open(path)?;
    let spec = r.spec();
    let ch = spec.channels.max(1) as usize;
    let all: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.map(|v| v as f32 * scale)).collect::<Result<_, _>>()?
        }
    };
    Ok((all.chunks(ch).map(|f| f.iter().sum::<f32>() / f.len() as f32).collect(), spec.sample_rate))
}

fn write_wav(path: &Path, x: &[f32], rate: u32) -> Result<(), hound::Error> {
    let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(path, spec)?;
    for v in x {
        w.write_sample((v.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
    }
    w.finalize()
}
