//! Pitch-preserving playback speed: WSOLA (waveform-similarity overlap-add) on mono PCM, streaming.
//!
//! The output is built from Hann-windowed frames of the source, 50 % overlapped (`frame` about 32 ms, synthesis hop
//! `hop` = half a frame). Each new frame is read about `hop * speed` source samples after the previous one, then
//! nudged by up to `tol` (about 10 ms) to where the source looks most like the natural continuation of the previous
//! frame (normalised cross-correlation over the overlap). So whole pitch periods are dropped (faster) or repeated
//! (slower) and the waveform stays continuous: the pitch and the formants are kept, which is what speech needs. At
//! 1x the frames are taken exactly one hop apart and the Hann windows sum to one, so the output is the source itself
//! (apart from a 16 ms fade at the start and the end).
//!
//! The speed can change between any two frames without a reset (no click); [`Stretcher::reset`] (a seek) starts
//! over with a 16 ms fade-in. Self-contained (no crate imports), so `examples/stretch_check.rs` checks it offline.

#![allow(dead_code)]

/// What stopped [`Stretcher::process`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// `out` is full.
    Full,
    /// A live source has no more samples yet (the frame needs more than has arrived): call again when it grew.
    Starved,
    /// The whole source has been played (only when `complete`).
    End,
}

pub struct Stretcher {
    frame: usize,
    hop: usize,
    tol: usize,
    window: Vec<f32>,
    /// Overlap-add accumulator (`frame` samples; the first `hop` are final once a frame was added).
    acc: Vec<f32>,
    /// Finished output not handed out yet, from `ready_at`.
    ready: Vec<f32>,
    ready_at: usize,
    /// Nominal source position (samples) of the next frame.
    pos: f64,
    /// Start of the previous frame (`None` after a reset: no continuation to match).
    prev: Option<usize>,
    /// The tail of the last frame was flushed: the source is done.
    flushed: bool,
}

impl Stretcher {
    /// A stretcher for audio at `rate` Hz.
    pub fn new(rate: u32) -> Stretcher {
        let rate = rate.max(8000) as f64;
        let hop = ((rate * 0.016).round() as usize).max(64);
        let frame = hop * 2;
        let tol = ((rate * 0.010).round() as usize).max(32);
        // Periodic Hann: two copies half a frame apart sum to exactly 1.
        let window = (0..frame).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / frame as f64).cos()) as f32).collect();
        Stretcher { frame, hop, tol, window, acc: vec![0.0; frame], ready: Vec::with_capacity(frame), ready_at: 0, pos: 0.0, prev: None, flushed: false }
    }

    /// Start over at source sample `pos` (seek, loop, new track).
    pub fn reset(&mut self, pos: f64) {
        self.pos = pos.max(0.0);
        self.prev = None;
        self.acc.iter_mut().for_each(|v| *v = 0.0);
        self.ready.clear();
        self.ready_at = 0;
        self.flushed = false;
    }

    /// The source position (samples) being played, give or take a frame.
    pub fn position(&self) -> f64 {
        self.pos
    }

    /// Fills `out` from `src` played `speed` times as fast (0.25..4). `complete`: `src` will not grow (its end is
    /// the end); otherwise a frame that would read past the end waits for more ([`Flow::Starved`]). Returns the
    /// samples written and why it stopped.
    pub fn process(&mut self, src: &[f32], speed: f32, complete: bool, out: &mut [f32]) -> (usize, Flow) {
        let speed = f64::from(speed.clamp(0.25, 4.0));
        let mut n = 0;
        loop {
            if self.ready_at < self.ready.len() {
                let take = (self.ready.len() - self.ready_at).min(out.len() - n);
                out[n..n + take].copy_from_slice(&self.ready[self.ready_at..self.ready_at + take]);
                self.ready_at += take;
                n += take;
            }
            if n == out.len() {
                return (n, Flow::Full);
            }
            if self.flushed {
                return (n, Flow::End);
            }
            self.ready.clear();
            self.ready_at = 0;
            if self.pos >= src.len() as f64 {
                if !complete {
                    return (n, Flow::Starved);
                }
                // The last frame's second half is still in the accumulator.
                self.ready.extend_from_slice(&self.acc[..self.hop]);
                self.flushed = true;
                continue;
            }
            if !complete && (self.pos as usize) + self.tol + self.frame > src.len() {
                return (n, Flow::Starved);
            }
            self.next_frame(src, speed);
        }
    }

    /// Picks, windows and adds the next frame; `hop` finished samples go to `ready`.
    fn next_frame(&mut self, src: &[f32], speed: f64) {
        let (hop, frame) = (self.hop, self.frame);
        let nominal = self.pos.round() as usize;
        let start = match self.prev {
            // 1x: exactly the next hop (the frames sum back to the source).
            Some(prev) if (speed - 1.0).abs() < 1e-3 => {
                self.pos = (prev + hop) as f64;
                prev + hop
            }
            Some(prev) => self.best_match(src, prev + hop, nominal),
            None => nominal,
        };
        for i in 0..frame {
            self.acc[i] += self.window[i] * src.get(start + i).copied().unwrap_or(0.0);
        }
        self.ready.extend_from_slice(&self.acc[..hop]);
        self.acc.copy_within(hop.., 0);
        self.acc[frame - hop..].iter_mut().for_each(|v| *v = 0.0);
        self.prev = Some(start);
        self.pos += hop as f64 * speed;
    }

    /// The start within `nominal ± tol` whose first `hop` samples correlate best with `src[target..target + hop]`
    /// (the natural continuation of the previous frame): a coarse search on every 2nd offset over every 4th sample,
    /// then a fine one around the winner.
    fn best_match(&self, src: &[f32], target: usize, nominal: usize) -> usize {
        let len = self.hop;
        let at = |i: usize| src.get(i).copied().unwrap_or(0.0);
        let lo = nominal.saturating_sub(self.tol);
        let hi = nominal + self.tol;
        let score = |cand: usize, step: usize| -> f32 {
            let (mut xy, mut yy) = (0.0f32, 0.0f32);
            let mut i = 0;
            while i < len {
                let (x, y) = (at(target + i), at(cand + i));
                xy += x * y;
                yy += y * y;
                i += step;
            }
            if yy <= 1e-9 { 0.0 } else { xy / yy.sqrt() }
        };
        let mut best = nominal;
        let mut best_score = score(nominal, 4);
        let mut c = lo;
        while c <= hi {
            let v = score(c, 4);
            if v > best_score {
                best_score = v;
                best = c;
            }
            c += 2;
        }
        let mut fine = best;
        let mut fine_score = score(best, 1);
        for c in best.saturating_sub(3).max(lo)..=(best + 3).min(hi) {
            let v = score(c, 1);
            if v > fine_score {
                fine_score = v;
                fine = c;
            }
        }
        fine
    }
}

/// Stretches a whole clip (offline use: checks and tests). `speed` 2 gives about half as many samples.
pub fn stretch_all(src: &[f32], rate: u32, speed: f32) -> Vec<f32> {
    let mut st = Stretcher::new(rate);
    let mut out = Vec::with_capacity((src.len() as f32 / speed) as usize + 4096);
    let mut buf = vec![0.0f32; 4096];
    loop {
        let (n, flow) = st.process(src, speed, true, &mut buf);
        out.extend_from_slice(&buf[..n]);
        if flow == Flow::End {
            return out;
        }
    }
}
