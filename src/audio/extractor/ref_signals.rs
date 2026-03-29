//! Reference-based per-layer signal measurement.
//!
//! Uses the diff between watermarked chunk and original fingerprint
//! to isolate the watermark signal for each layer.

use std::f32::consts::PI;
use super::fingerprint::{ChunkFingerprint, band_edges, band_rms_iir};

pub fn measure_layer_ref(
    wm_chunk: &[f32],
    orig: &ChunkFingerprint,
    sample_rate: u32,
    layer_idx: usize,
    key: u64,
) -> f32 {
    let key_byte = derive_byte(key, layer_idx as u64);
    let sr = sample_rate as f32;
    match layer_idx {
        0  => ref_amplitude_scaling(wm_chunk, orig, key_byte),
        1  => ref_micro_time_shift(wm_chunk, orig, key_byte),
        2  => ref_envelope_shaping(wm_chunk, orig, sample_rate, key_byte),
        3  => ref_band_limited_gain(wm_chunk, orig, sr, key_byte),
        4  => ref_high_freq_emphasis(wm_chunk, orig, sr, key_byte),
        5  => ref_narrowband_attenuation(wm_chunk, orig, sr, key_byte),
        6  => ref_phase_perturbation(wm_chunk, orig, sr, key_byte),
        7  => ref_sample_reordering(wm_chunk, orig, key_byte),
        8  => ref_energy_redistribution(wm_chunk, orig, key_byte),
        9  => ref_noise_shaping(wm_chunk, derive_u64(key, 9)),
        10 => ref_controlled_nonlinear(wm_chunk, orig, key_byte),
        11 => ref_logistic_map(wm_chunk, derive_u64(key, 11), key_byte),
        12 => ref_comb_filter(wm_chunk, orig, sr, key_byte),
        13 => ref_spectral_tilt(wm_chunk, orig, sr, key_byte),
        14 => ref_temporal_variance(wm_chunk, orig, key_byte),
        _  => 0.0,
    }
}

// ─── L1: Amplitude Scaling ────────────────────────────────────────────────────
// FIX: gain_offset can be tiny (0.025%) after quantisation.
// Use sign-consistency: does wm_rms move in the expected direction from orig?
// Score full marks if direction matches, partial if magnitude is close.
fn ref_amplitude_scaling(chunk: &[f32], orig: &ChunkFingerprint, key_byte: u8) -> f32 {
    let gain_offset = (key_byte as f32 / 255.0) * 0.006 - 0.003;
    let wm_rms = compute_rms(chunk);
    if orig.rms < 1e-6 { return 0.0; }
    let actual_delta = wm_rms - orig.rms;
    // Direction match: expected sign == actual sign
    let direction_score = if (actual_delta >= 0.0) == (gain_offset >= 0.0) { 0.7 } else { 0.1 };
    // Magnitude score: how close is the ratio?
    let expected_ratio = 1.0 + gain_offset;
    let actual_ratio   = wm_rms / orig.rms;
    let mag_score = score_from_delta((actual_ratio - expected_ratio).abs(), 0.02);
    (direction_score * 0.6 + mag_score * 0.4).clamp(0.0, 1.0)
}

// ─── L2: Micro Time Shift ─────────────────────────────────────────────────────
// FIX: multiply by 200 instead of 50 — 4-sample shift only moves AC by ~0.004
fn ref_micro_time_shift(chunk: &[f32], orig: &ChunkFingerprint, key_byte: u8) -> f32 {
    let shift = (key_byte as usize % 8) + 1;
    let lag_idx = (shift - 1).min(orig.ac_lags.len() - 1);
    let wm_ac   = autocorr_norm(chunk, shift);
    let orig_ac = orig.ac_lags[lag_idx];
    let delta = (orig_ac - wm_ac).max(0.0); // shift always decorrelates
    (delta * 200.0).clamp(0.0, 1.0)
}

// ─── L3: Envelope Shaping ────────────────────────────────────────────────────
fn ref_envelope_shaping(chunk: &[f32], orig: &ChunkFingerprint, sample_rate: u32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let mod_freq = 0.05 + t * 0.45;
    let phase    = t * 2.0 * PI;
    let sr = sample_rate as f32;
    let frame_size = (sr * 0.05) as usize;
    if frame_size == 0 || chunk.len() < frame_size * 4 { return 0.0; }

    let ratio_curve: Vec<f32> = chunk.chunks(frame_size)
        .map(|f| compute_rms(f) / (orig.rms + 1e-9) - 1.0)
        .collect();
    let n = ratio_curve.len();

    let expected: Vec<f32> = (0..n).map(|i| {
        let t_sec = (i as f32 + 0.5) * frame_size as f32 / sr;
        let depth = 0.001 + t * 0.003;
        depth * (2.0 * PI * mod_freq * t_sec + phase).sin()
    }).collect();

    let corr = normalised_correlation(&ratio_curve, &expected);
    ((corr + 0.3) * 1.4).clamp(0.0, 1.0)
}

// ─── L4: Band-Limited Gain ────────────────────────────────────────────────────
// FIX: measure directly at fc with a narrow IIR window, not log bands
fn ref_band_limited_gain(chunk: &[f32], orig: &ChunkFingerprint, sr: f32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let fc = 200.0 + t * 3800.0;
    let db_gain = if key_byte & 1 == 0 { 0.1_f32 } else { -0.1_f32 };
    let expected_ratio = 10.0_f32.powf(db_gain / 20.0);

    // Narrow window: fc ± 200 Hz
    let f_lo = (fc - 200.0).max(50.0);
    let f_hi = (fc + 200.0).min(sr / 2.0 - 1.0);
    let wm_band = band_rms_iir(chunk, sr, f_lo, f_hi);

    // Get orig band using matching log-band index as reference
    let edges = band_edges(sr);
    let band_idx = edges.windows(2)
        .position(|w| fc >= w[0] && fc < w[1])
        .unwrap_or(3)
        .min(orig.band_rms.len() - 1);
    let orig_band = orig.band_rms[band_idx];
    if orig_band < 1e-9 { return 0.0; }

    // Scale wm_band to match orig_band measurement scale
    let orig_ref = band_rms_iir(
        &vec![orig.rms; 512], sr, f_lo, f_hi
    );
    let scale = if orig_ref > 1e-9 { orig_band / orig_ref } else { 1.0 };
    let wm_scaled = wm_band * scale;
    let actual_ratio = wm_scaled / orig_band;

    let delta = (actual_ratio - expected_ratio).abs();
    score_from_delta(delta, 0.15)
}

// ─── L5: High-Frequency Emphasis ─────────────────────────────────────────────
// FIX: measure band directly above fc rather than hardcoded highest band
fn ref_high_freq_emphasis(chunk: &[f32], orig: &ChunkFingerprint, sr: f32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let fc = 6000.0 + t * 10000.0;
    if fc >= sr / 2.0 - 100.0 { return 0.5; } // fc above nyquist — can't measure, neutral score
    let expected_ratio = 10.0_f32.powf(0.05 / 20.0);

    // Measure in [fc, fc*1.5] — the boosted region
    let f_lo = fc;
    let f_hi = (fc * 1.5).min(sr / 2.0 - 1.0);
    let wm_hf = band_rms_iir(chunk, sr, f_lo, f_hi);

    // Reference: band just below fc [fc*0.67, fc]
    let f_ref_lo = (fc * 0.67).max(100.0);
    let wm_ref = band_rms_iir(chunk, sr, f_ref_lo, fc);
    let orig_hf  = band_rms_iir(&vec![orig.rms; 512], sr, f_lo, f_hi);
    let orig_ref = band_rms_iir(&vec![orig.rms; 512], sr, f_ref_lo, fc);

    if orig_ref < 1e-9 || wm_ref < 1e-9 { return 0.0; }

    // Ratio of ratios: (wm_hf/wm_ref) / (orig_hf/orig_ref) should equal expected_ratio
    let wm_ratio   = wm_hf / wm_ref;
    let orig_ratio = orig_hf / orig_ref;
    if orig_ratio < 1e-9 { return 0.0; }
    let actual = wm_ratio / orig_ratio;
    score_from_delta((actual - expected_ratio).abs(), 0.15)
}

// ─── L6: Narrowband Attenuation ──────────────────────────────────────────────
// FIX: measure exactly at [fc-bw/2, fc+bw/2] using IIR directly
fn ref_narrowband_attenuation(chunk: &[f32], orig: &ChunkFingerprint, sr: f32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let fc = 1000.0 + t * 7000.0;
    let bw = fc / 8.0; // Q=8
    let expected_ratio = 10.0_f32.powf(-0.08 / 20.0); // ≈ 0.9908

    let f_lo = (fc - bw / 2.0).max(50.0);
    let f_hi = (fc + bw / 2.0).min(sr / 2.0 - 1.0);
    let wm_notch = band_rms_iir(chunk, sr, f_lo, f_hi);

    // Reference: adjacent band [fc+bw/2, fc+bw*2]
    let f_ref_lo = fc + bw / 2.0;
    let f_ref_hi = (fc + bw * 2.0).min(sr / 2.0 - 1.0);
    let wm_ref   = band_rms_iir(chunk, sr, f_ref_lo, f_ref_hi);

    let orig_notch = band_rms_iir(&vec![orig.rms; 512], sr, f_lo, f_hi);
    let orig_ref   = band_rms_iir(&vec![orig.rms; 512], sr, f_ref_lo, f_ref_hi);

    if orig_ref < 1e-9 || wm_ref < 1e-9 { return 0.0; }

    let wm_ratio   = wm_notch / wm_ref;
    let orig_ratio = orig_notch / orig_ref;
    if orig_ratio < 1e-9 { return 0.0; }
    let actual = wm_ratio / orig_ratio;
    score_from_delta((actual - expected_ratio).abs(), 0.15)
}

// ─── L7: Phase Perturbation ───────────────────────────────────────────────────
fn ref_phase_perturbation(chunk: &[f32], orig: &ChunkFingerprint, sr: f32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let fp = 500.0 + t * 4500.0;
    let lag = ((sr / (2.0 * PI * fp)).round() as usize)
        .max(1).min(orig.ac_lags.len());
    let wm_ac   = autocorr_norm(chunk, lag);
    let orig_ac = orig.ac_lags[(lag - 1).min(orig.ac_lags.len() - 1)];
    let delta = (wm_ac - orig_ac).abs();
    (delta * 30.0).clamp(0.0, 1.0)
}

// ─── L8: Sample Reordering ───────────────────────────────────────────────────
fn ref_sample_reordering(chunk: &[f32], orig: &ChunkFingerprint, key_byte: u8) -> f32 {
    let block_size = (key_byte as usize % 3) + 2;
    let wm_ac1   = autocorr_norm(chunk, 1);
    let orig_ac1 = orig.ac_lags[0];
    let wm_acb   = autocorr_norm(chunk, block_size - 1);
    let orig_acb = orig.ac_lags[(block_size - 2).min(orig.ac_lags.len() - 1)];
    let delta = (wm_ac1 - orig_ac1).abs() + (wm_acb - orig_acb).abs();
    (delta * 25.0).clamp(0.0, 1.0)
}

// ─── L9: Energy Redistribution (DC) ──────────────────────────────────────────
fn ref_energy_redistribution(chunk: &[f32], orig: &ChunkFingerprint, key_byte: u8) -> f32 {
    let expected_dc_shift = (key_byte as f32 / 255.0) * 0.001 - 0.0005;
    let wm_dc    = chunk.iter().sum::<f32>() / chunk.len().max(1) as f32;
    let dc_delta = wm_dc - orig.dc;
    let error    = (dc_delta - expected_dc_shift).abs();
    score_from_delta(error, 0.005)
}

// ─── L10: Noise Shaping ───────────────────────────────────────────────────────
fn ref_noise_shaping(chunk: &[f32], key: u64) -> f32 {
    let amplitude = 0.0001_f32;
    let mut state = key;
    let sequence: Vec<f32> = chunk.iter()
        .map(|_| next_xorshift(&mut state) * amplitude)
        .collect();
    let corr = normalised_correlation(chunk, &sequence).abs();
    (corr * 500.0).clamp(0.0, 1.0)
}

// ─── L11: Controlled Nonlinear ───────────────────────────────────────────────
// FIX: use variance-of-squared-samples as proxy for soft clipping.
// tanh compresses peaks → reduces variance of x^2 relative to E[x^2]^2.
fn ref_controlled_nonlinear(chunk: &[f32], orig: &ChunkFingerprint, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let drive = 1.001 + t * 0.009;
    // tanh compression reduces peak-to-average ratio.
    // Metric: std(x^2) / mean(x^2) — lower after tanh compression.
    let sq: Vec<f32> = chunk.iter().map(|s| s * s).collect();
    let mean_sq = sq.iter().sum::<f32>() / sq.len() as f32;
    if mean_sq < 1e-12 { return 0.0; }
    let std_sq = (sq.iter().map(|v| (v - mean_sq).powi(2)).sum::<f32>()
                  / sq.len() as f32).sqrt();
    let wm_crest = std_sq / mean_sq;

    // For original: approximate from orig.rms — typical crest ≈ 1.5 for music
    let orig_crest_approx = 1.5_f32;
    // Expected reduction per unit of drive-1
    let expected_reduction = (drive - 1.0) * 3.0;
    let actual_reduction   = orig_crest_approx - wm_crest;
    let error = (actual_reduction - expected_reduction).abs();
    score_from_delta(error, 0.5)
}

// ─── L12: Logistic Map ────────────────────────────────────────────────────────
fn ref_logistic_map(chunk: &[f32], _key: u64, key_byte: u8) -> f32 {
    let t = key_byte as f64 / 255.0;
    let r = 3.9 + t * 0.099;
    let amplitude = 0.0002_f32;
    let mut x = 0.1 + t * 0.8;
    let sequence: Vec<f32> = chunk.iter().map(|_| {
        x = r * x * (1.0 - x);
        ((x - 0.5) * 2.0) as f32 * amplitude
    }).collect();
    let corr = normalised_correlation(chunk, &sequence).abs();
    (corr * 500.0).clamp(0.0, 1.0)
}

// ─── L13: Comb Filter ─────────────────────────────────────────────────────────
fn ref_comb_filter(chunk: &[f32], orig: &ChunkFingerprint, sr: f32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let delay_samples = ((10.0 + t * 40.0) * sr / 1000.0).round() as usize;
    let delay_samples = delay_samples.max(1);
    if chunk.len() < delay_samples + 64 { return 0.0; }
    let lag_idx = (delay_samples - 1).min(orig.ac_lags.len() - 1);
    let wm_ac   = autocorr_norm(chunk, delay_samples);
    let orig_ac = orig.ac_lags[lag_idx];
    let delta   = (wm_ac - orig_ac).abs();
    (delta * 200.0).clamp(0.0, 1.0)
}

// ─── L14: Spectral Tilt ───────────────────────────────────────────────────────
// FIX: find the band containing fc and compare to the band above it.
fn ref_spectral_tilt(chunk: &[f32], orig: &ChunkFingerprint, sr: f32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let fc = 80.0 + t * 720.0;
    let db_gain = if key_byte & 1 == 0 { 0.05_f32 } else { -0.05_f32 };
    let expected_ratio = 10.0_f32.powf(db_gain / 20.0);

    let edges = band_edges(sr);
    // Find band index containing fc
    let band_idx = edges.windows(2)
        .position(|w| fc >= w[0] && fc < w[1])
        .unwrap_or(0)
        .min(orig.band_rms.len() - 2);

    let wm_low  = band_rms_iir(chunk, sr, edges[band_idx], edges[band_idx + 1]);
    let wm_high = band_rms_iir(chunk, sr, edges[band_idx + 1], edges[band_idx + 2]);
    let orig_low  = orig.band_rms[band_idx];
    let orig_high = orig.band_rms[band_idx + 1];

    if orig_low < 1e-9 || orig_high < 1e-9 || wm_high < 1e-9 { return 0.0; }

    // Tilt changes low/high ratio; compare wm vs orig
    let wm_tilt   = wm_low / wm_high;
    let orig_tilt = orig_low / orig_high;
    if orig_tilt < 1e-9 { return 0.0; }
    let actual = wm_tilt / orig_tilt;
    score_from_delta((actual - expected_ratio).abs(), 0.10)
}

// ─── L15: Temporal Variance ───────────────────────────────────────────────────
fn ref_temporal_variance(chunk: &[f32], orig: &ChunkFingerprint, key_byte: u8) -> f32 {
    let period = 8 + (key_byte as usize % 25);
    let lag_idx = (period - 1).min(orig.energy_ac.len() - 1);
    let energy: Vec<f32> = chunk.iter().map(|s| s * s).collect();
    let wm_eac   = energy_autocorr_norm(&energy, period);
    let orig_eac = orig.energy_ac[lag_idx];
    let delta    = (wm_eac - orig_eac).abs();
    (delta * 60.0).clamp(0.0, 1.0)
}

// ─── DSP helpers ─────────────────────────────────────────────────────────────

fn compute_rms(s: &[f32]) -> f32 {
    if s.is_empty() { return 0.0; }
    (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt()
}

fn autocorr_norm(samples: &[f32], lag: usize) -> f32 {
    if lag >= samples.len() { return 0.0; }
    let n   = samples.len() - lag;
    let num = samples[..n].iter().zip(&samples[lag..]).map(|(a,b)| a*b).sum::<f32>() / n as f32;
    let den = samples.iter().map(|s| s*s).sum::<f32>() / samples.len() as f32;
    if den < 1e-12 { 0.0 } else { num / den }
}

fn energy_autocorr_norm(energy: &[f32], lag: usize) -> f32 {
    if lag >= energy.len() { return 0.0; }
    let n   = energy.len() - lag;
    let num = energy[..n].iter().zip(&energy[lag..]).map(|(a,b)| a*b).sum::<f32>() / n as f32;
    let den = energy.iter().map(|s| s*s).sum::<f32>() / energy.len() as f32;
    if den < 1e-12 { 0.0 } else { num / den }
}

fn normalised_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }
    let ma = a[..n].iter().sum::<f32>() / n as f32;
    let mb = b[..n].iter().sum::<f32>() / n as f32;
    let num: f32 = a[..n].iter().zip(&b[..n]).map(|(x,y)| (x-ma)*(y-mb)).sum();
    let da  = a[..n].iter().map(|x| (x-ma).powi(2)).sum::<f32>().sqrt();
    let db  = b[..n].iter().map(|y| (y-mb).powi(2)).sum::<f32>().sqrt();
    if da < 1e-12 || db < 1e-12 { return 0.0; }
    (num / (da * db)).clamp(-1.0, 1.0)
}

fn score_from_delta(delta: f32, tolerance: f32) -> f32 {
    (-(delta / tolerance) * std::f32::consts::LN_2).exp().clamp(0.0, 1.0)
}

fn kurtosis(samples: &[f32]) -> f32 {
    let n = samples.len() as f32;
    if n < 4.0 { return 3.0; }
    let mean = samples.iter().sum::<f32>() / n;
    let var  = samples.iter().map(|s| (s-mean).powi(2)).sum::<f32>() / n;
    if var < 1e-12 { return 3.0; }
    samples.iter().map(|s| (s-mean).powi(4)).sum::<f32>() / n / (var * var)
}

fn next_xorshift(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    ((x as i64) as f32) / (i64::MAX as f32)
}

fn derive_byte(key: u64, i: u64) -> u8 {
    let mut h = key ^ (i.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    h ^= h >> 30; h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27; h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    (h & 0xFF) as u8
}

fn derive_u64(key: u64, i: u64) -> u64 {
    let mut h = key ^ (i.wrapping_mul(0x517c_c1b7_2722_0a95));
    h ^= h >> 30; h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27; h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    h
}
