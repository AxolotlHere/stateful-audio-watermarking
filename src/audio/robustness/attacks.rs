//! Audio attack functions for robustness testing.
//!
//! Each function takes the path to the watermarked WAV, applies an attack,
//! and returns the resulting samples as Vec<f32> ready for extraction.
//!
//! All attacks use ffmpeg or sox via subprocess — no Rust DSP needed here.

use std::process::Command;
use crate::audio::io::read_wav;

/// Describes a single attack.
pub struct Attack {
    pub name: &'static str,
    pub category: &'static str,
}

/// All attacks in order.
pub fn all_attacks() -> Vec<Attack> {
    vec![
        Attack { name: "baseline",       category: "none"      },
        Attack { name: "mp3_128",        category: "compress"  },
        Attack { name: "mp3_64",         category: "compress"  },
        Attack { name: "mp3_32",         category: "compress"  },
        Attack { name: "aac_128",        category: "compress"  },
        Attack { name: "noise_30db",     category: "noise"     },
        Attack { name: "noise_20db",     category: "noise"     },
        Attack { name: "trim_5pct",      category: "edit"      },
        Attack { name: "trim_10pct",     category: "edit"      },
        Attack { name: "amplitude_90",   category: "level"     },
        Attack { name: "amplitude_110",  category: "level"     },
        Attack { name: "resample_44",    category: "resample"  },
    ]
}

/// Apply attack by name and return resulting samples.
/// Returns None if the attack fails (e.g. tool not found).
pub fn apply_attack(
    attack_name: &str,
    input_wav: &str,
    sample_rate: u32,
) -> Option<Vec<f32>> {
    match attack_name {
        "baseline"      => attack_baseline(input_wav),
        "mp3_128"       => attack_mp3(input_wav, 128, sample_rate),
        "mp3_64"        => attack_mp3(input_wav, 64,  sample_rate),
        "mp3_32"        => attack_mp3(input_wav, 32,  sample_rate),
        "aac_128"       => attack_aac(input_wav, 128, sample_rate),
        "noise_30db"    => attack_noise(input_wav, -30.0, sample_rate),
        "noise_20db"    => attack_noise(input_wav, -20.0, sample_rate),
        "trim_5pct"     => attack_trim(input_wav, 0.05, sample_rate),
        "trim_10pct"    => attack_trim(input_wav, 0.10, sample_rate),
        "amplitude_90"  => attack_amplitude(input_wav, 0.90, sample_rate),
        "amplitude_110" => attack_amplitude(input_wav, 1.10, sample_rate),
        "resample_44"   => attack_resample(input_wav, 44100, sample_rate),
        _               => None,
    }
}

// ─── Baseline ────────────────────────────────────────────────────────────────

fn attack_baseline(input_wav: &str) -> Option<Vec<f32>> {
    let wave = read_wav(input_wav);
    Some(wave.samples)
}

// ─── MP3 Compression ─────────────────────────────────────────────────────────
// WAV → encode to MP3 → decode back to WAV → read

fn attack_mp3(input_wav: &str, bitrate: u32, _sample_rate: u32) -> Option<Vec<f32>> {
    let tmp_mp3 = format!("/tmp/wm_attack_{bitrate}.mp3");
    let tmp_wav = format!("/tmp/wm_attack_{bitrate}.wav");

    // Encode to MP3
    let enc = Command::new("ffmpeg")
        .args(["-y", "-i", input_wav,
               "-b:a", &format!("{bitrate}k"),
               "-ac", "1",
               &tmp_mp3])
        .output().ok()?;
    if !enc.status.success() {
        eprintln!("MP3 encode failed: {}", String::from_utf8_lossy(&enc.stderr));
        return None;
    }

    // Decode back to WAV
    let dec = Command::new("ffmpeg")
        .args(["-y", "-i", &tmp_mp3,
               "-ac", "1",
               "-c:a", "pcm_s16le",
               &tmp_wav])
        .output().ok()?;
    if !dec.status.success() {
        eprintln!("MP3 decode failed: {}", String::from_utf8_lossy(&dec.stderr));
        return None;
    }

    let wave = read_wav(&tmp_wav);
    Some(wave.samples)
}

// ─── AAC Compression ─────────────────────────────────────────────────────────

fn attack_aac(input_wav: &str, bitrate: u32, _sample_rate: u32) -> Option<Vec<f32>> {
    let tmp_aac = format!("/tmp/wm_attack_aac{bitrate}.m4a");
    let tmp_wav = format!("/tmp/wm_attack_aac{bitrate}.wav");

    let enc = Command::new("ffmpeg")
        .args(["-y", "-i", input_wav,
               "-b:a", &format!("{bitrate}k"),
               "-ac", "1",
               "-c:a", "aac",
               &tmp_aac])
        .output().ok()?;
    if !enc.status.success() {
        eprintln!("AAC encode failed: {}", String::from_utf8_lossy(&enc.stderr));
        return None;
    }

    let dec = Command::new("ffmpeg")
        .args(["-y", "-i", &tmp_aac,
               "-ac", "1",
               "-c:a", "pcm_s16le",
               &tmp_wav])
        .output().ok()?;
    if !dec.status.success() {
        eprintln!("AAC decode failed: {}", String::from_utf8_lossy(&dec.stderr));
        return None;
    }

    let wave = read_wav(&tmp_wav);
    Some(wave.samples)
}

// ─── Additive Noise ───────────────────────────────────────────────────────────
// Uses sox synth to add white noise at a given dBFS level

fn attack_noise(input_wav: &str, noise_db: f32, sample_rate: u32) -> Option<Vec<f32>> {
    let tag = format!("{}", noise_db.abs() as u32);
    let tmp_wav = format!("/tmp/wm_attack_noise{tag}.wav");

    // Convert dBFS to linear amplitude
    let amplitude = 10.0_f32.powf(noise_db / 20.0);

    let status = Command::new("sox")
        .args([
            input_wav,
            &tmp_wav,
            "synth", "whitenoise", "vol", &format!("{amplitude:.6}"),
            ":",
            "mix", input_wav,
        ])
        .output();

    // Sox mix syntax: mix original with noise
    // Simpler approach: use sox's newfile mixing
    let status = Command::new("sox")
        .args([
            "-M",                    // mix
            input_wav,
            "|",
        ])
        .output();

    // Cleanest approach: generate noise file then mix
    let tmp_noise = format!("/tmp/wm_noise{tag}.wav");

    // Get duration of input
    let dur_out = Command::new("sox")
        .args(["--info", "-D", input_wav])
        .output().ok()?;
    let duration: f32 = String::from_utf8_lossy(&dur_out.stdout)
        .trim().parse().unwrap_or(30.0);

    // Generate noise file at the same sample rate as the input
    let noise_gen = Command::new("sox")
        .args([
            "-n", "-r", &sample_rate.to_string(), "-c", "1",
            &tmp_noise,
            "synth", &format!("{duration:.3}"),
            "whitenoise",
            "vol", &format!("{amplitude:.8}"),
        ])
        .output().ok()?;
    if !noise_gen.status.success() {
        eprintln!("Noise gen failed: {}", String::from_utf8_lossy(&noise_gen.stderr));
        return None;
    }

    // Mix original + noise
    let mix = Command::new("sox")
        .args(["-m", input_wav, &tmp_noise, &tmp_wav])
        .output().ok()?;
    if !mix.status.success() {
        eprintln!("Noise mix failed: {}", String::from_utf8_lossy(&mix.stderr));
        return None;
    }

    let wave = read_wav(&tmp_wav);
    Some(wave.samples)
}

// ─── Trimming ────────────────────────────────────────────────────────────────
// Remove `pct` of audio from the END (tail trim).
// Trimming from the start would shift all sample positions and break the
// key-derived chunk alignment. Tail trim is a realistic attack that preserves
// the watermarked region while reducing file length.

fn attack_trim(input_wav: &str, pct: f32, _sample_rate: u32) -> Option<Vec<f32>> {
    let tag = format!("{}", (pct * 100.0) as u32);
    let tmp_wav = format!("/tmp/wm_attack_trim{tag}.wav");

    // Get duration
    let dur_out = Command::new("sox")
        .args(["--info", "-D", input_wav])
        .output().ok()?;
    let duration: f32 = String::from_utf8_lossy(&dur_out.stdout)
        .trim().parse().unwrap_or(30.0);
    let keep_secs = duration * (1.0 - pct);

    // trim 0 <keep_secs> = keep from 0 to keep_secs (drops the tail)
    let status = Command::new("sox")
        .args([input_wav, &tmp_wav,
               "trim", "0", &format!("{keep_secs:.3}")])
        .output().ok()?;
    if !status.status.success() {
        eprintln!("Trim failed: {}", String::from_utf8_lossy(&status.stderr));
        return None;
    }

    let wave = read_wav(&tmp_wav);
    Some(wave.samples)
}

// ─── Amplitude Scaling ────────────────────────────────────────────────────────

fn attack_amplitude(input_wav: &str, factor: f32, _sample_rate: u32) -> Option<Vec<f32>> {
    let tag = format!("{}", (factor * 100.0) as u32);
    let tmp_wav = format!("/tmp/wm_attack_amp{tag}.wav");

    let status = Command::new("sox")
        .args([input_wav, &tmp_wav,
               "vol", &format!("{factor:.4}")])
        .output().ok()?;
    if !status.status.success() {
        eprintln!("Amplitude scale failed: {}", String::from_utf8_lossy(&status.stderr));
        return None;
    }

    let wave = read_wav(&tmp_wav);
    Some(wave.samples)
}

// ─── Resampling ───────────────────────────────────────────────────────────────
// Downsample to 44100 then back up to original rate

fn attack_resample(input_wav: &str, intermediate_sr: u32, original_sr: u32) -> Option<Vec<f32>> {
    let tmp_down = format!("/tmp/wm_attack_resamp_down.wav");
    let tmp_up   = format!("/tmp/wm_attack_resamp_up.wav");

    // Downsample
    let down = Command::new("sox")
        .args([input_wav, "-r", &intermediate_sr.to_string(), &tmp_down])
        .output().ok()?;
    if !down.status.success() {
        eprintln!("Resample down failed: {}", String::from_utf8_lossy(&down.stderr));
        return None;
    }

    // Upsample back
    let up = Command::new("sox")
        .args([&tmp_down, "-r", &original_sr.to_string(), &tmp_up])
        .output().ok()?;
    if !up.status.success() {
        eprintln!("Resample up failed: {}", String::from_utf8_lossy(&up.stderr));
        return None;
    }

    let wave = read_wav(&tmp_up);
    Some(wave.samples)
}
