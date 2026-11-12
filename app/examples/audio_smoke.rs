//! Records 3 seconds from the default microphone and plays it back.
//! Run: `cargo run --example audio_smoke` (needs audio devices; fails with a clear message without them).

#[path = "../src/audio/mod.rs"]
mod audio;

use std::time::{Duration, Instant};

fn main() {
    if let Err(e) = run() {
        eprintln!("audio_smoke: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let devices = audio::input_devices();
    println!("input devices: {devices:?}");

    let rec = audio::Recorder::start(None, None)?;
    println!("recording 3 s from \"{}\" at {} Hz ... speak now", rec.device_name(), rec.sample_rate());
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(3) {
        let bars = "#".repeat((rec.peak() * 40.0) as usize);
        println!("{:4.1}s  {:<40} peak {:.2}", rec.elapsed_s(), bars, rec.peak());
        std::thread::sleep(Duration::from_millis(250));
    }
    let path = std::env::temp_dir().join("mimvox-audio-smoke.wav");
    let info = rec.stop(&path)?;
    println!("saved {} ({:.2} s, peak {:.2})", info.path.display(), info.duration_s, info.peak);

    let player = audio::Player::new(None)?;
    println!("playing back on \"{}\"", player.device_name());
    player.play(&path, false)?;
    std::thread::sleep(Duration::from_millis(100));
    while player.is_playing() {
        std::thread::sleep(Duration::from_millis(100));
    }
    println!("done, played {:.2} s", player.position_s());
    Ok(())
}
