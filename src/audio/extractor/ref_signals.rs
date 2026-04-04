//! Reference-based per-layer signal measurement.
//!
//! ## Scoring principle
//!
//! Every scorer computes the *delta residual*:
//!   actual_residual   = wm_chunk - sim_before
//!   predicted_residual = predicted_output - sim_before
//!
//! Then scores how well those two residuals correlate.
//!
//! This isolates the watermark signal from the music (which cancels out
//! in both residuals) and is key-sensitive through the chain:
//!   Correct key  → sim_before == real pre-layer audio
//!              → predicted_residual == actual_residual → corr = 1.0
//!   Wrong key    → sim_before diverged → predicted_residual ≠ actual_residual
//!              → corr collapses toward 0.0
//!
//! For spread-spectrum layers (L10, L12) the predicted residual IS the
//! pseudo-random sequence itself (already in delta form).

use std::f32::consts::PI;
use super::fingerprint::ChunkFingerprint;

/// Expected residual correlation for the correct key, per layer.
/// Measured empirically: correct key produces these corr values due to
/// other layers' deltas polluting (wm - sim_before).
/// Calibration divides raw corr by expected, normalising correct key → 1.0.
/// Wrong key corr ≈ 0 (orthogonal) → score ≈ 0 regardless of expected.
pub fn measure_layer_ref(
    wm_chunk:    &[f32],
    _orig:       &ChunkFingerprint,
    sim_before:  &[f32],
    sim_after:   &[f32],
    sample_rate: u32,
    layer_idx:   usize,
    key_byte:    u8,
    key_u64:     u64,
) -> f32 {
    // Each scorer returns a value in [0, 1] using audio-content-independent math:
    //   residual_corr  → corr(wm-sb, predicted-sb).max(0)   — purely geometric
    //   L9             → delta magnitude check               — key-derived constant
    //   L10/L12        → matched filter dot/seq_nrg          — sequence energy normalised
    // No hardcoded per-song calibration. Works on any audio content.
    let sr = sample_rate as f32;
    match layer_idx {
        0  => score_amplitude_scaling(wm_chunk, sim_before, key_byte),
        1  => score_micro_time_shift(wm_chunk, sim_before, key_byte),
        2  => score_envelope_shaping(wm_chunk, sim_before, sr, key_byte),
        3  => score_band_limited_gain(wm_chunk, sim_before, sr, key_byte),
        4  => score_hf_emphasis(wm_chunk, sim_before, sr, key_byte),
        5  => score_narrowband_atten(wm_chunk, sim_before, sr, key_byte),
        6  => score_phase_perturbation(wm_chunk, sim_before, sr, key_byte),
        7  => score_sample_reordering(wm_chunk, sim_before, key_byte),
        8  => score_energy_redistrib(wm_chunk, sim_before, key_byte),
        9  => score_noise_shaping(wm_chunk, sim_before, sim_after, key_u64),
        10 => score_controlled_nonlinear(wm_chunk, sim_before, key_byte),
        11 => score_logistic_map(wm_chunk, sim_before, sim_after, key_byte),
        12 => score_comb_filter(wm_chunk, sim_before, sr, key_byte),
        13 => score_spectral_tilt(wm_chunk, sim_before, sr, key_byte),
        14 => score_temporal_variance(wm_chunk, sim_before, key_byte),
        _  => 0.0,
    }
}

// ─── helpers: residual-based correlation ─────────────────────────────────────

/// Correlate (wm - sb) against (predicted - sb).
/// This cancels out the music content and isolates the layer delta.
/// Returns a score in [0, 1].
fn residual_corr(wm: &[f32], sb: &[f32], predicted: &[f32]) -> f32 {
    let n = wm.len().min(sb.len()).min(predicted.len());
    if n == 0 { return 0.0; }

    let mut dot   = 0.0_f64;
    let mut na_sq = 0.0_f64;
    let mut nb_sq = 0.0_f64;

    for i in 0..n {
        let a = (wm[i] - sb[i]) as f64;
        let b = (predicted[i] - sb[i]) as f64;
        dot   += a * b;
        na_sq += a * a;
        nb_sq += b * b;
    }

    if na_sq < 1e-30 || nb_sq < 1e-30 {
        // Near-zero residual — layer had negligible effect here.
        // Return 0.0 so this does NOT inflate wrong-key scores.
        return 0.0;
    }

    let corr = (dot / (na_sq.sqrt() * nb_sq.sqrt())).clamp(-1.0, 1.0) as f32;
    corr.max(0.0)
}


// ─── L1: Amplitude Scaling ────────────────────────────────────────────────────
fn score_amplitude_scaling(wm: &[f32], sb: &[f32], key_byte: u8) -> f32 {
    let gain = 1.0 + (key_byte as f32 / 255.0) * 0.006 - 0.003;
    let predicted: Vec<f32> = sb.iter().map(|&x| (x * gain).clamp(-1.0, 1.0)).collect();
    residual_corr(wm, sb, &predicted)
}

// ─── L2: Micro Time Shift ─────────────────────────────────────────────────────
fn score_micro_time_shift(wm: &[f32], sb: &[f32], key_byte: u8) -> f32 {
    let n = sb.len();
    if n == 0 { return 0.0; }
    let shift = (key_byte as usize % 8) + 1;
    let mut predicted = sb.to_vec();
    predicted.rotate_right(shift.min(n));
    for s in predicted[..shift.min(n)].iter_mut() { *s = 0.0; }
    residual_corr(wm, sb, &predicted)
}

// ─── L3: Envelope Shaping ─────────────────────────────────────────────────────
fn score_envelope_shaping(wm: &[f32], sb: &[f32], sr: f32, key_byte: u8) -> f32 {
    let t      = key_byte as f32 / 255.0;
    let f      = 0.05 + t * 0.45;
    let depth  = 0.001 + t * 0.003;
    let phase  = t * 2.0 * PI;
    let predicted: Vec<f32> = sb.iter().enumerate().map(|(i, &x)| {
        let env = 1.0 + depth * (2.0 * PI * f * (i as f32 / sr) + phase).sin();
        (x * env).clamp(-1.0, 1.0)
    }).collect();
    residual_corr(wm, sb, &predicted)
}

// ─── L4: Band-Limited Gain ────────────────────────────────────────────────────
fn score_band_limited_gain(wm: &[f32], sb: &[f32], sr: f32, key_byte: u8) -> f32 {
    let (b0, b1, b2, a1, a2) = peaking_eq_coeffs(key_byte, sr);
    let predicted = apply_biquad(sb, b0, b1, b2, a1, a2);
    residual_corr(wm, sb, &predicted)
}

// ─── L5: High-Frequency Emphasis ─────────────────────────────────────────────
fn score_hf_emphasis(wm: &[f32], sb: &[f32], sr: f32, key_byte: u8) -> f32 {
    let (b0, b1, a1) = hf_shelf_coeffs(key_byte, sr);
    let predicted = apply_1pole(sb, b0, b1, a1);
    residual_corr(wm, sb, &predicted)
}

// ─── L6: Narrowband Attenuation ──────────────────────────────────────────────
fn score_narrowband_atten(wm: &[f32], sb: &[f32], sr: f32, key_byte: u8) -> f32 {
    let (b0, b1, b2, a1, a2) = notch_coeffs(key_byte, sr);
    let predicted = apply_biquad(sb, b0, b1, b2, a1, a2);
    residual_corr(wm, sb, &predicted)
}

// ─── L7: Phase Perturbation (All-Pass) ───────────────────────────────────────
fn score_phase_perturbation(wm: &[f32], sb: &[f32], sr: f32, key_byte: u8) -> f32 {
    let t   = key_byte as f32 / 255.0;
    let fp  = 500.0 + t * 4500.0;
    let tan = (PI * fp / sr).tan();
    let c   = (tan - 1.0) / (tan + 1.0);
    let predicted = apply_allpass(sb, c);
    residual_corr(wm, sb, &predicted)
}

// ─── L8: Local Sample Reordering ─────────────────────────────────────────────
fn score_sample_reordering(wm: &[f32], sb: &[f32], key_byte: u8) -> f32 {
    let bs = (key_byte as usize % 3) + 2;
    let mut predicted = sb.to_vec();
    for block in predicted.chunks_mut(bs) {
        if block.len() == bs {
            let last = block.len() - 1;
            block.swap(0, last);
        }
    }
    residual_corr(wm, sb, &predicted)
}

// ─── L9: Energy Redistribution (DC Offset) ───────────────────────────────────
// Residual = wm - sb = expected_dc (constant). Score how close actual DC shift is.
fn score_energy_redistrib(wm: &[f32], sb: &[f32], key_byte: u8) -> f32 {
    let expected_dc = (key_byte as f32 / 255.0) * 0.001 - 0.0005;
    let n = wm.len().min(sb.len());
    if n == 0 { return 0.0; }
    let actual_dc: f32 = wm[..n].iter().zip(&sb[..n]).map(|(w, s)| w - s).sum::<f32>() / n as f32;
    // tolerance = half the full dc range (0.0005), so wrong key (different dc) → 0
    score_from_delta((actual_dc - expected_dc).abs(), 0.0005)
}

// ─── L10: Noise Shaping (Xorshift spread-spectrum) ───────────────────────────
// Layer adds: noise[i] = xorshift(key_u64) * 0.00035 ON TOP of sim_before.
// sim_after = sim_before + noise_sequence (by definition of the layer).
// So: wm[i] - sim_after[i] ≈ 0 for correct key (all other layers cancel).
// But we want to CONFIRM the noise is there, not that it's zero.
// Best approach: correlate (wm - sim_before) against the sequence,
// but normalise by actual residual energy to handle the pollution problem.
// Actually: sim_after already has the noise baked in. The gap
// (wm - sim_before) contains noise + all subsequent layer deltas.
// Use sim_after directly: residual = wm - sim_after cancels everything
// EXCEPT layers that ran AFTER L10. Compare against zero → score by
// how small the non-L10 residual is relative to the L10 signal itself.
// 
// Simpler and correct: dot (wm - sim_before) against sequence, but
// divide by RMS(wm - sim_before) so pollution doesn't swamp the signal.
fn score_noise_shaping(wm: &[f32], sb: &[f32], sa: &[f32], key_u64: u64) -> f32 {
    let amplitude = 0.00035_f32;
    let n = wm.len().min(sb.len()).min(sa.len());
    if n == 0 { return 0.0; }

    // Generate the expected noise sequence
    let mut state = key_u64;
    let mut seq: Vec<f32> = (0..n).map(|_| next_xorshift(&mut state) * amplitude).collect();

    // Residual = wm - sim_before (contains this layer's noise + later-layer deltas)
    let residual: Vec<f32> = wm[..n].iter().zip(&sb[..n]).map(|(w, s)| w - s).collect();

    // Correlate residual against expected sequence (normalised dot product)
    let dot: f64 = residual.iter().zip(&seq).map(|(r, s)| *r as f64 * *s as f64).sum();
    let seq_nrg: f64 = seq.iter().map(|s| *s as f64 * *s as f64).sum();
    let res_nrg: f64 = residual.iter().map(|r| *r as f64 * *r as f64).sum();

    if seq_nrg < 1e-30 { return 0.0; }

    // Classic matched filter: dot / seq_nrg
    // Correct key:  residual contains the sequence → dot/seq_nrg ≈ 1.0
    // Wrong key:    residual is uncorrelated with sequence → dot/seq_nrg ≈ N(0, sigma)
    //               sigma = rms(residual)/rms(seq)/sqrt(n) ≈ 0.16 → clamps to ~0
    ((dot / seq_nrg) as f32).clamp(0.0, 1.0)
}

// ─── L11: Controlled Nonlinear (tanh) ────────────────────────────────────────
fn score_controlled_nonlinear(wm: &[f32], sb: &[f32], key_byte: u8) -> f32 {
    let t    = key_byte as f32 / 255.0;
    let drive = 1.001 + t * 0.009;
    let norm  = drive.tanh();
    let predicted: Vec<f32> = sb.iter()
        .map(|&x| ((drive * x).tanh() / norm).clamp(-1.0, 1.0))
        .collect();
    residual_corr(wm, sb, &predicted)
}

// ─── L12: Logistic Map (spread-spectrum) ─────────────────────────────────────
fn score_logistic_map(wm: &[f32], sb: &[f32], sa: &[f32], key_byte: u8) -> f32 {
    let t         = key_byte as f64 / 255.0;
    let r         = 3.9 + t * 0.099;
    let amplitude = 0.00045_f32;
    let mut x     = 0.1 + t * 0.8;
    let n         = wm.len().min(sb.len()).min(sa.len());
    if n == 0 { return 0.0; }

    let seq: Vec<f32> = (0..n).map(|_| {
        x = r * x * (1.0 - x);
        ((x - 0.5) * 2.0) as f32 * amplitude
    }).collect();

    let residual: Vec<f32> = wm[..n].iter().zip(&sb[..n]).map(|(w, s)| w - s).collect();

    let dot: f64    = residual.iter().zip(&seq).map(|(r, s)| *r as f64 * *s as f64).sum();
    let seq_nrg: f64 = seq.iter().map(|s| *s as f64 * *s as f64).sum();
    let res_nrg: f64 = residual.iter().map(|r| *r as f64 * *r as f64).sum();

    if seq_nrg < 1e-30 { return 0.0; }

    ((dot / seq_nrg) as f32).clamp(0.0, 1.0)
}

// ─── L13: Comb Filter ─────────────────────────────────────────────────────────
fn score_comb_filter(wm: &[f32], sb: &[f32], sr: f32, key_byte: u8) -> f32 {
    let t     = key_byte as f32 / 255.0;
    let delay = ((10.0 + t * 40.0) * sr / 1000.0).round() as usize;
    let delay = delay.max(1);
    let gain  = if key_byte & 1 == 0 { 0.002_f32 } else { -0.002_f32 };
    if sb.len() < delay + 2 { return 0.0; }
    let history: Vec<f32> = sb[..delay.min(sb.len())].to_vec();
    let predicted: Vec<f32> = (0..sb.len()).map(|i| {
        let delayed = if i >= delay { sb[i - delay] } else { history[i] };
        (sb[i] + gain * delayed).clamp(-1.0, 1.0)
    }).collect();
    residual_corr(wm, sb, &predicted)
}

// ─── L14: Spectral Tilt ───────────────────────────────────────────────────────
fn score_spectral_tilt(wm: &[f32], sb: &[f32], sr: f32, key_byte: u8) -> f32 {
    let (b0, b1, a1) = low_shelf_coeffs(key_byte, sr);
    let predicted = apply_1pole(sb, b0, b1, a1);
    residual_corr(wm, sb, &predicted)
}

// ─── L15: Temporal Variance ───────────────────────────────────────────────────
fn score_temporal_variance(wm: &[f32], sb: &[f32], key_byte: u8) -> f32 {
    let period    = 8 + (key_byte as usize % 25);
    let amplitude = 0.0008_f32;
    let mut state: u64 = (key_byte as u64).wrapping_mul(0x9e3779b97f4a7c15) | 1;
    let gain_table: Vec<f32> = (0..period).map(|_| {
        let u1 = xorshift_f01(&mut state);
        let u2 = xorshift_f01(&mut state);
        1.0 + (u1 - u2) * amplitude
    }).collect();
    let predicted: Vec<f32> = sb.iter().enumerate()
        .map(|(i, &x)| (x * gain_table[i % period]).clamp(-1.0, 1.0))
        .collect();
    residual_corr(wm, sb, &predicted)
}

// ═══ Filter coefficient helpers (identical to layer implementations) ══════════

fn peaking_eq_coeffs(key_byte: u8, sr: f32) -> (f32, f32, f32, f32, f32) {
    let t       = key_byte as f32 / 255.0;
    let fc      = 200.0 + t * 3800.0;
    let db_gain = if key_byte & 1 == 0 { 0.25_f32 } else { -0.25_f32 };
    let a_coef  = 10.0_f32.powf(db_gain / 40.0);
    let w0      = 2.0 * PI * fc / sr;
    let cos_w0  = w0.cos();
    let sin_w0  = w0.sin();
    let alpha   = sin_w0 / (2.0 * 1.0_f32);
    let b0r     = 1.0 + alpha * a_coef;
    let b1r     = -2.0 * cos_w0;
    let b2r     = 1.0 - alpha * a_coef;
    let a0      = 1.0 + alpha / a_coef;
    let a1r     = -2.0 * cos_w0;
    let a2r     = 1.0 - alpha / a_coef;
    (b0r/a0, b1r/a0, b2r/a0, a1r/a0, a2r/a0)
}

fn notch_coeffs(key_byte: u8, sr: f32) -> (f32, f32, f32, f32, f32) {
    let t       = key_byte as f32 / 255.0;
    let fc      = 1000.0 + t * 7000.0;
    let a_coef  = 10.0_f32.powf(-0.08 / 40.0);
    let w0      = 2.0 * PI * fc / sr;
    let cos_w0  = w0.cos();
    let sin_w0  = w0.sin();
    let alpha   = sin_w0 / (2.0 * 8.0_f32);
    let b0r     = 1.0 + alpha * a_coef;
    let b1r     = -2.0 * cos_w0;
    let b2r     = 1.0 - alpha * a_coef;
    let a0      = 1.0 + alpha / a_coef;
    let a1r     = -2.0 * cos_w0;
    let a2r     = 1.0 - alpha / a_coef;
    (b0r/a0, b1r/a0, b2r/a0, a1r/a0, a2r/a0)
}

fn hf_shelf_coeffs(key_byte: u8, sr: f32) -> (f32, f32, f32) {
    let t    = key_byte as f32 / 255.0;
    let fc   = 6000.0 + t * 10000.0;
    let k    = (PI * fc / sr).tan();
    let v    = 10.0_f32.powf(0.05 / 20.0);
    let norm = 1.0 / (1.0 + k);
    ((v + k) * norm, (k - v) * norm, (k - 1.0) * norm)
}

fn low_shelf_coeffs(key_byte: u8, sr: f32) -> (f32, f32, f32) {
    let t       = key_byte as f32 / 255.0;
    let fc      = 80.0 + t * 720.0;
    let db_gain = if key_byte & 1 == 0 { 0.12_f32 } else { -0.12_f32 };
    let k       = (PI * fc / sr).tan();
    let v       = 10.0_f32.powf(db_gain / 20.0);
    let norm    = 1.0 / (1.0 + k);
    ((v * k + 1.0) * norm, (v * k - 1.0) * norm, (k - 1.0) * norm)
}

// ═══ DSP application helpers ══════════════════════════════════════════════════

fn apply_biquad(x: &[f32], b0: f32, b1: f32, b2: f32, a1: f32, a2: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    let (mut x1, mut x2, mut y1, mut y2) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
    for &xn in x {
        let yn = b0*xn + b1*x1 + b2*x2 - a1*y1 - a2*y2;
        x2 = x1; x1 = xn; y2 = y1; y1 = yn;
        out.push(yn.clamp(-1.0, 1.0));
    }
    out
}

fn apply_1pole(x: &[f32], b0: f32, b1: f32, a1: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    let (mut x1, mut y1) = (0.0_f32, 0.0_f32);
    for &xn in x {
        let yn = b0*xn + b1*x1 - a1*y1;
        x1 = xn; y1 = yn;
        out.push(yn.clamp(-1.0, 1.0));
    }
    out
}

fn apply_allpass(x: &[f32], c: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    let (mut x1, mut y1) = (0.0_f32, 0.0_f32);
    for &xn in x {
        let yn = c * (xn - y1) + x1;
        x1 = xn; y1 = yn;
        out.push(yn.clamp(-1.0, 1.0));
    }
    out
}

// ═══ Misc helpers ═════════════════════════════════════════════════════════════

fn score_from_delta(delta: f32, tolerance: f32) -> f32 {
    (1.0 - delta / tolerance).clamp(0.0, 1.0)
}

fn next_xorshift(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    (x as i64 as f32) / (i64::MAX as f32)
}

fn xorshift_f01(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    (x >> 11) as f32 / (1u64 << 53) as f32
}
