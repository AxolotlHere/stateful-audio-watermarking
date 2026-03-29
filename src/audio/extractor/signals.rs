//! Per-layer signal measurement functions.
//!
//! Each function takes the watermarked chunk and the key, regenerates the
//! expected signal from the key, then measures how well the chunk matches.
//!
//! Returns a score in [0.0, 1.0]:
//!   0.0 = no evidence of this layer
//!   1.0 = perfect match

use std::f32::consts::PI;

/// Dispatch to the correct measurement function by layer index (0-based).
pub fn measure_layer(
    chunk: &[f32],
    sample_rate: u32,
    layer_idx: usize,
    key: u64,
    _slot: usize,
) -> f32 {
    let key_byte = derive_byte(key, layer_idx as u64);
    match layer_idx {
        0  => measure_amplitude_scaling(chunk, key_byte),
        1  => measure_micro_time_shift(chunk, key_byte),
        2  => measure_envelope_shaping(chunk, sample_rate, key_byte),
        3  => measure_band_limited_gain(chunk, sample_rate, key_byte),
        4  => measure_high_freq_emphasis(chunk, sample_rate, key_byte),
        5  => measure_narrowband_attenuation(chunk, sample_rate, key_byte),
        6  => measure_phase_perturbation(chunk, sample_rate, key_byte),
        7  => measure_sample_reordering(chunk, key_byte),
        8  => measure_energy_redistribution(chunk, key_byte),
        9  => measure_noise_shaping(chunk, derive_u64(key, 9)),
        10 => measure_controlled_nonlinear(chunk, key_byte),
        11 => measure_logistic_map(chunk, key_byte),
        12 => measure_comb_filter(chunk, sample_rate, key_byte),
        13 => measure_spectral_tilt(chunk, sample_rate, key_byte),
        14 => measure_temporal_variance(chunk, key_byte),
        _  => 0.0,
    }
}

// ─── Layer 1: Amplitude Scaling ──────────────────────────────────────────────
// The embedded gain = 1.0 + gain_offset.
// We measure the chunk RMS and compare to the expected gain direction.
// Since we don't have the original, we use the sign of gain_offset as
// a soft indicator and compare RMS to a flat-spectrum reference.
fn measure_amplitude_scaling(chunk: &[f32], key_byte: u8) -> f32 {
    let gain_offset = (key_byte as f32 / 255.0) * 0.006 - 0.003;
    let expected_gain = 1.0 + gain_offset;
    let rms = compute_rms(chunk);
    if rms < 1e-6 { return 0.0; }
    // Score based on how well rms * (1/expected_gain) ≈ rms
    // i.e. the RMS is consistent with the expected scaling direction
    let ratio = rms / expected_gain;
    // Score: distance of ratio from rms, normalised
    let delta = (ratio - rms).abs() / (rms + 1e-9);
    // A gain offset of 0.003 → delta ≈ 0.003; we want that to score ~0.7
    score_from_delta(delta, 0.003)
}

// ─── Layer 2: Micro Time Shift ────────────────────────────────────────────────
// Autocorrelation at lag N should show a dip matching the shift.
fn measure_micro_time_shift(chunk: &[f32], key_byte: u8) -> f32 {
    let shift = (key_byte as usize % 8) + 1;
    if chunk.len() < shift + 64 { return 0.0; }
    // Compute normalised autocorrelation at lag=shift vs lag=0
    let ac0 = autocorr(chunk, 0);
    let ac_n = autocorr(chunk, shift);
    if ac0 < 1e-12 { return 0.0; }
    // A shift introduces a measurable asymmetry
    let ratio = (ac0 - ac_n.abs()) / ac0;
    (ratio * 80.0).clamp(0.0, 1.0)
}

// ─── Layer 3: Envelope Shaping ────────────────────────────────────────────────
// Compute the short-time RMS curve and correlate against the expected sinusoid.
fn measure_envelope_shaping(chunk: &[f32], sample_rate: u32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let mod_freq = 0.05 + t * 0.45;
    let phase    = t * 2.0 * PI;

    // Short-time RMS in 100ms frames
    let frame_size = (sample_rate as f32 * 0.1) as usize;
    if frame_size == 0 || chunk.len() < frame_size * 3 { return 0.0; }

    let rms_curve: Vec<f32> = chunk.chunks(frame_size)
        .map(|f| compute_rms(f))
        .collect();

    let n = rms_curve.len();
    let sr = sample_rate as f32;
    // Generate expected envelope at frame centres
    let expected: Vec<f32> = (0..n).map(|i| {
        let t_sec = (i as f32 + 0.5) * frame_size as f32 / sr;
        (2.0 * PI * mod_freq * t_sec + phase).sin()
    }).collect();

    normalised_correlation(&rms_curve, &expected)
}

// ─── Layer 4: Band-Limited Gain (Peaking EQ) ─────────────────────────────────
// Measure the energy ratio in a narrow band around fc vs adjacent bands.
fn measure_band_limited_gain(chunk: &[f32], sample_rate: u32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let fc = 200.0 + t * 3800.0;
    let db_gain = if key_byte & 1 == 0 { 0.1_f32 } else { -0.1_f32 };
    let sr = sample_rate as f32;

    // Band energy ratio: energy in [fc-100, fc+100] vs [fc-300, fc-100]
    let band_energy   = band_rms(chunk, sr, fc - 100.0, fc + 100.0);
    let ref_energy    = band_rms(chunk, sr, fc - 300.0, fc - 100.0);
    if ref_energy < 1e-9 { return 0.0; }

    let ratio = band_energy / ref_energy;
    let expected_ratio = 10.0_f32.powf(db_gain / 20.0); // ≈ 1.0057 for 0.1 dB

    let delta = (ratio - expected_ratio).abs();
    score_from_delta(delta, 0.01)
}

// ─── Layer 5: High-Frequency Emphasis ────────────────────────────────────────
fn measure_high_freq_emphasis(chunk: &[f32], sample_rate: u32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let fc = 6000.0 + t * 10000.0;
    let sr = sample_rate as f32;

    if fc >= sr / 2.0 { return 0.0; }
    let hf_energy  = band_rms(chunk, sr, fc, sr / 2.0);
    let mid_energy = band_rms(chunk, sr, fc / 2.0, fc);
    if mid_energy < 1e-9 { return 0.0; }

    let ratio = hf_energy / mid_energy;
    // The shelf boost is tiny; we look for consistency above 0.5
    score_from_delta((ratio - 1.0).abs(), 0.05)
}

// ─── Layer 6: Narrowband Attenuation ─────────────────────────────────────────
fn measure_narrowband_attenuation(chunk: &[f32], sample_rate: u32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let fc = 1000.0 + t * 7000.0;
    let sr = sample_rate as f32;
    let bw = fc / 8.0; // Q=8 → bandwidth = fc/Q

    let notch_energy = band_rms(chunk, sr, fc - bw / 2.0, fc + bw / 2.0);
    let ref_energy   = band_rms(chunk, sr, fc - bw * 2.0, fc - bw / 2.0);
    if ref_energy < 1e-9 { return 0.0; }

    // A notch → lower ratio than reference
    let ratio = notch_energy / ref_energy;
    let expected = 10.0_f32.powf(-0.08 / 20.0); // ≈ 0.9908
    let delta = (ratio - expected).abs();
    score_from_delta(delta, 0.02)
}

// ─── Layer 7: Phase Perturbation (All-Pass) ───────────────────────────────────
// All-pass changes phase but not magnitude.
// We measure the asymmetry of the autocorrelation function —
// the all-pass introduces a measurable group-delay signature.
fn measure_phase_perturbation(chunk: &[f32], sample_rate: u32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let fp = 500.0 + t * 4500.0;
    let sr = sample_rate as f32;
    // Expected lag of maximum group delay (≈ 1/(2π·fp))
    let expected_lag = (sr / (2.0 * PI * fp)).round() as usize;
    let expected_lag = expected_lag.max(1).min(chunk.len() / 4);

    let ac0   = autocorr(chunk, 0);
    let ac_lag = autocorr(chunk, expected_lag);
    if ac0 < 1e-12 { return 0.0; }

    // All-pass decorrelates at the pole lag
    let decorr = 1.0 - (ac_lag / ac0).abs();
    decorr.clamp(0.0, 1.0)
}

// ─── Layer 8: Sample Reordering ───────────────────────────────────────────────
// Block swaps introduce a lag-1 autocorrelation asymmetry.
fn measure_sample_reordering(chunk: &[f32], key_byte: u8) -> f32 {
    let block_size = (key_byte as usize % 3) + 2;
    // Measure odd/even sample correlation asymmetry caused by swaps
    let ac1 = autocorr(chunk, 1);
    let ac_bs = autocorr(chunk, block_size - 1);
    let ac0 = autocorr(chunk, 0);
    if ac0 < 1e-12 { return 0.0; }

    let asymmetry = (ac1.abs() - ac_bs.abs()).abs() / ac0;
    (asymmetry * 50.0).clamp(0.0, 1.0)
}

// ─── Layer 9: Energy Redistribution (DC Offset) ───────────────────────────────
fn measure_energy_redistribution(chunk: &[f32], key_byte: u8) -> f32 {
    let expected_dc = (key_byte as f32 / 255.0) * 0.001 - 0.0005;
    let actual_dc = chunk.iter().sum::<f32>() / chunk.len() as f32;
    let delta = (actual_dc - expected_dc).abs();
    score_from_delta(delta, 0.0005)
}

// ─── Layer 10: Noise Shaping (Spread-Spectrum Correlation) ───────────────────
// Re-generate the exact pseudo-random sequence from the key and correlate
// against the chunk. This is the most powerful extractor.
fn measure_noise_shaping(chunk: &[f32], key: u64) -> f32 {
    let amplitude = 0.0001_f32;
    let mut state = key;
    let sequence: Vec<f32> = chunk.iter().map(|_| {
        next_xorshift(&mut state) * amplitude
    }).collect();

    // Normalised cross-correlation between chunk and sequence
    let corr = normalised_correlation(chunk, &sequence);
    // Boost small but consistent correlations
    (corr * 20.0).clamp(0.0, 1.0)
}

// ─── Layer 11: Controlled Nonlinear (tanh) ───────────────────────────────────
// The tanh introduces even/odd harmonic distortion.
// We measure the 2nd harmonic of the dominant frequency as a proxy.
fn measure_controlled_nonlinear(chunk: &[f32], key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let drive = 1.001 + t * 0.009;
    // Expected THD ≈ drive^2 / 12 for small signals (Taylor expansion of tanh)
    let expected_thd = drive * drive / 12.0 - 1.0 / 12.0;
    // Measure actual signal kurtosis as a proxy for nonlinearity
    let kurt = kurtosis(chunk);
    // Gaussian signal has kurtosis ≈ 3.0; nonlinearity shifts it
    let delta = (kurt - 3.0).abs() - expected_thd * 100.0;
    score_from_delta(delta.abs(), 0.1)
}

// ─── Layer 12: Logistic Map ───────────────────────────────────────────────────
// Re-generate the logistic sequence and cross-correlate.
fn measure_logistic_map(chunk: &[f32], key_byte: u8) -> f32 {
    let t = key_byte as f64 / 255.0;
    let r  = 3.9 + t * 0.099;
    let amplitude = 0.0002_f32;
    let mut x = 0.1 + t * 0.8;

    let sequence: Vec<f32> = chunk.iter().map(|_| {
        x = r * x * (1.0 - x);
        ((x - 0.5) * 2.0) as f32 * amplitude
    }).collect();

    let corr = normalised_correlation(chunk, &sequence);
    (corr * 20.0).clamp(0.0, 1.0)
}

// ─── Layer 13: Comb Filter ────────────────────────────────────────────────────
// Feedforward comb creates a peak in autocorrelation at lag=D.
fn measure_comb_filter(chunk: &[f32], sample_rate: u32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let delay_ms = 10.0 + t * 40.0;
    let delay_samples = (delay_ms * sample_rate as f32 / 1000.0).round() as usize;
    let delay_samples = delay_samples.max(1);

    if chunk.len() < delay_samples + 64 { return 0.0; }
    let ac0 = autocorr(chunk, 0);
    let ac_d = autocorr(chunk, delay_samples);
    if ac0 < 1e-12 { return 0.0; }

    // Comb filter raises |AC(D)| slightly
    let ratio = ac_d.abs() / ac0;
    (ratio * 2.0).clamp(0.0, 1.0)
}

// ─── Layer 14: Spectral Tilt ──────────────────────────────────────────────────
fn measure_spectral_tilt(chunk: &[f32], sample_rate: u32, key_byte: u8) -> f32 {
    let t = key_byte as f32 / 255.0;
    let fc = 80.0 + t * 720.0;
    let db_gain = if key_byte & 1 == 0 { 0.05_f32 } else { -0.05_f32 };
    let sr = sample_rate as f32;

    let low_energy  = band_rms(chunk, sr, 20.0, fc);
    let high_energy = band_rms(chunk, sr, fc, fc * 4.0);
    if high_energy < 1e-9 { return 0.0; }

    let ratio = low_energy / high_energy;
    let expected = 10.0_f32.powf(db_gain / 20.0);
    let delta = (ratio - expected).abs();
    score_from_delta(delta, 0.01)
}

// ─── Layer 15: Temporal Variance ─────────────────────────────────────────────
// The periodic gain table creates a periodic pattern in sample-wise energy.
// We look for the autocorrelation peak at lag=period.
fn measure_temporal_variance(chunk: &[f32], key_byte: u8) -> f32 {
    let period = 8 + (key_byte as usize % 25);
    if chunk.len() < period * 4 { return 0.0; }

    // Compute squared sample energy sequence
    let energy: Vec<f32> = chunk.iter().map(|s| s * s).collect();
    let ac0 = autocorr(&energy, 0);
    let ac_p = autocorr(&energy, period);
    if ac0 < 1e-12 { return 0.0; }

    let ratio = (ac_p / ac0).abs();
    (ratio * 3.0).clamp(0.0, 1.0)
}

// ─── DSP Helpers ─────────────────────────────────────────────────────────────

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Unnormalised autocorrelation at a given lag.
fn autocorr(samples: &[f32], lag: usize) -> f32 {
    if lag >= samples.len() { return 0.0; }
    let n = samples.len() - lag;
    samples[..n].iter().zip(samples[lag..].iter()).map(|(a, b)| a * b).sum::<f32>() / n as f32
}

/// Pearson-style normalised cross-correlation in [-1, 1].
fn normalised_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }
    let mean_a = a[..n].iter().sum::<f32>() / n as f32;
    let mean_b = b[..n].iter().sum::<f32>() / n as f32;
    let num: f32 = a[..n].iter().zip(b[..n].iter())
        .map(|(x, y)| (x - mean_a) * (y - mean_b))
        .sum();
    let da: f32 = a[..n].iter().map(|x| (x - mean_a).powi(2)).sum::<f32>().sqrt();
    let db: f32 = b[..n].iter().map(|y| (y - mean_b).powi(2)).sum::<f32>().sqrt();
    if da < 1e-12 || db < 1e-12 { return 0.0; }
    (num / (da * db)).clamp(-1.0, 1.0)
}

/// Very simple band energy approximation using a running IIR bandpass.
/// bp(n) = sin(2π·fc/sr·n) gated to [f_lo, f_hi].
/// For extraction purposes we use a difference-of-lowpass approximation.
fn band_rms(samples: &[f32], sample_rate: f32, f_lo: f32, f_hi: f32) -> f32 {
    if samples.is_empty() { return 0.0; }
    let f_lo = f_lo.max(1.0).min(sample_rate / 2.0 - 1.0);
    let f_hi = f_hi.max(f_lo + 1.0).min(sample_rate / 2.0 - 1.0);

    // Simple difference-of-first-order LP filters
    let lp_coeff = |fc: f32| -> f32 {
        let x = (-2.0 * PI * fc / sample_rate).exp();
        1.0 - x
    };
    let a_hi = lp_coeff(f_hi);
    let a_lo = lp_coeff(f_lo);

    let mut lp_hi = 0.0_f32;
    let mut lp_lo = 0.0_f32;
    let mut sum_sq = 0.0_f32;

    for &s in samples {
        lp_hi = lp_hi + a_hi * (s - lp_hi);
        lp_lo = lp_lo + a_lo * (s - lp_lo);
        let bp = lp_hi - lp_lo;
        sum_sq += bp * bp;
    }
    (sum_sq / samples.len() as f32).sqrt()
}

/// Convert a delta (absolute error) to a [0,1] score.
/// `tolerance` is the delta at which score = 0.5.
fn score_from_delta(delta: f32, tolerance: f32) -> f32 {
    // Exponential decay: score = exp(-delta / tolerance * ln2)
    // → score = 0.5 when delta = tolerance
    (-(delta / tolerance) * std::f32::consts::LN_2).exp().clamp(0.0, 1.0)
}

/// Signal kurtosis (4th standardised moment).
fn kurtosis(samples: &[f32]) -> f32 {
    let n = samples.len() as f32;
    if n < 4.0 { return 3.0; }
    let mean = samples.iter().sum::<f32>() / n;
    let var  = samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
    if var < 1e-12 { return 3.0; }
    let m4 = samples.iter().map(|s| (s - mean).powi(4)).sum::<f32>() / n;
    m4 / (var * var)
}

fn next_xorshift(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x as i64) as f32) / (i64::MAX as f32)
}

/// Derive a sub-key byte from the master key (must match layers/mod.rs).
fn derive_byte(key: u64, i: u64) -> u8 {
    let mut h = key ^ (i.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    (h & 0xFF) as u8
}

fn derive_u64(key: u64, i: u64) -> u64 {
    let mut h = key ^ (i.wrapping_mul(0x517c_c1b7_2722_0a95));
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    h
}
