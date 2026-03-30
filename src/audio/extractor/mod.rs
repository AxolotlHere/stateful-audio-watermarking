//! `audio::extractor` — watermark verification given a key.
//!
//! Two modes:
//!   - `extract()`          — blind (no original needed, lower accuracy)
//!   - `extract_with_ref()` — reference-based (uses .wmpf fingerprint, high accuracy)

pub mod signals;
pub mod report;
pub mod fingerprint;
pub mod ref_signals;
pub mod key_test;

pub use report::{ExtractionResult, LayerResult};
pub use fingerprint::Fingerprint;
pub use key_test::{run_key_test, KeyTestResult};

use crate::audio::layers::{build_layers, permute_layers, KeyedChunker};
use signals::*;
use ref_signals::measure_layer_ref;

// ─── Reference-based extraction (the good one) ───────────────────────────────

/// Verify a watermarked file using the original fingerprint.
///
/// This is the primary extraction method. It captures the statistical
/// difference between the watermarked audio and the original fingerprint,
/// making even sub-0.1 dB changes reliably detectable.
pub fn extract_with_ref(
    wm_samples: &[f32],
    sample_rate: u32,
    key: u64,
    fp: &Fingerprint,
) -> ExtractionResult {
    let layers  = build_layers(key, sample_rate);
    let order   = permute_layers(key);
    let chunker = KeyedChunker::new(key, sample_rate);

    let mut layer_accum: Vec<Vec<f32>> = vec![Vec::new(); 15];
    let mut chunk_count = 0usize;
    let mut offset = 0usize;

    loop {
        if offset >= wm_samples.len() { break; }
        let end = (offset + chunker.chunk_size_samples()).min(wm_samples.len());
        let wm_chunk = &wm_samples[offset..end];
        offset = end;

        // Use the matching original fingerprint chunk (wrap if needed)
        let orig = &fp.chunks[chunk_count % fp.chunks.len()];
        chunk_count += 1;

        for &layer_idx in &order {
            let score = measure_layer_ref(wm_chunk, orig, sample_rate, layer_idx, key);
            layer_accum[layer_idx].push(score);
        }
    }

    build_result(key, sample_rate, chunk_count, layer_accum, &layers)
}

// ─── Blind extraction (no original) ─────────────────────────────────────────

/// Verify without the original audio. Less accurate for subtle layers.
pub fn extract(samples: &[f32], sample_rate: u32, key: u64) -> ExtractionResult {
    let layers  = build_layers(key, sample_rate);
    let order   = permute_layers(key);
    let chunker = KeyedChunker::new(key, sample_rate);

    let mut layer_accum: Vec<Vec<f32>> = vec![Vec::new(); 15];
    let mut chunk_count = 0usize;
    let mut offset = 0usize;

    loop {
        if offset >= samples.len() { break; }
        let end = (offset + chunker.chunk_size_samples()).min(samples.len());
        let chunk = &samples[offset..end];
        offset = end;
        chunk_count += 1;

        for (slot, &layer_idx) in order.iter().enumerate() {
            let score = measure_layer(chunk, sample_rate, layer_idx, key, slot);
            layer_accum[layer_idx].push(score);
        }
    }

    build_result(key, sample_rate, chunk_count, layer_accum, &layers)
}

// ─── Shared result builder ────────────────────────────────────────────────────

fn build_result(
    key: u64,
    sample_rate: u32,
    chunk_count: usize,
    layer_accum: Vec<Vec<f32>>,
    layers: &[Box<dyn crate::audio::layers::Layer>],
) -> ExtractionResult {
    let layer_results: Vec<LayerResult> = (0..15).map(|idx| {
        let scores = &layer_accum[idx];
        let mean = if scores.is_empty() { 0.0 }
                   else { scores.iter().sum::<f32>() / scores.len() as f32 };
        LayerResult {
            layer_index: idx + 1,
            layer_name: layers[idx].name(),
            score: mean,
            detected: mean >= detection_threshold(idx),
        }
    }).collect();

    let detected_count = layer_results.iter().filter(|r| r.detected).count();
    let confidence = weighted_confidence(&layer_results);

    ExtractionResult {
        key,
        chunk_count,
        sample_rate,
        watermark_detected: confidence >= 0.60,
        confidence,
        layer_results,
        detected_count,
    }
}

fn detection_threshold(layer_idx: usize) -> f32 {
    match layer_idx {
        6  => 0.50,
        9  => 0.30,
        11 => 0.30,
        3 | 4 | 5 => 0.40,
        13 => 0.40,
        _  => 0.25,
    }
}

fn weighted_confidence(results: &[LayerResult]) -> f32 {
    // KEY-SENSITIVE layers only drive the confidence score.
    // Key-insensitive layers (EQ bands, kurtosis, generic AC) are excluded
    // because they score similarly regardless of which key is tested —
    // causing a 94% false-positive rate with wrong keys.
    //
    // Index: 0=L1 ... 14=L15
    // KEY-SENSITIVE  (weight > 0): L2, L8, L9, L10, L12, L15
    // KEY-INSENSITIVE (weight = 0): L1, L3, L4, L5, L6, L7, L11, L13, L14
    const WEIGHTS: [f32; 15] = [
        0.0,  // L1  AmplitudeScaling   — RMS ratio, key-insensitive
        1.0,  // L2  MicroTimeShift     — shift=f(key), AC changes at specific lag
        0.0,  // L3  EnvelopeShaping    — RMS curve, weakly key-sensitive
        0.0,  // L4  BandLimitedGain    — ratio ~1.01, always passes
        0.0,  // L5  HFEmphasis         — ratio ~1.006, always passes
        0.0,  // L6  NarrowbandAtten    — ratio ~0.99, always passes
        0.0,  // L7  PhasePerturbation  — AC lag similar across keys
        1.0,  // L8  SampleReordering   — block_size=f(key)
        2.0,  // L9  EnergyRedistrib    — dc_offset exact match to key
        3.0,  // L10 NoiseShaping       — xorshift seq unique per key (spread-spectrum)
        0.0,  // L11 ControlledNonlin   — kurtosis, not key-specific
        3.0,  // L12 LogisticMap        — chaotic seq unique per key (spread-spectrum)
        0.0,  // L13 CombFilter         — AC at lag, similar across keys
        0.0,  // L14 SpectralTilt       — band ratio, key-insensitive
        1.5,  // L15 TemporalVariance   — period=f(key), energy AC at specific lag
    ];
    // Total weight = 1+1+2+3+3+1.5 = 11.5
    let total: f32 = WEIGHTS.iter().sum();
    if total < 1e-6 { return 0.0; }
    let scored: f32 = results.iter().zip(WEIGHTS.iter())
        .map(|(r, &w)| r.score * w)
        .sum();

    // Hard gate: if the two strongest key-sensitive layers (L10, L12) both fail,
    // cap confidence at 35% regardless of other scores.
    // This prevents wrong keys from coasting on generic audio properties.
    let l10_score = results[9].score;   // index 9 = L10
    let l12_score = results[11].score;  // index 11 = L12
    let raw = (scored / total).clamp(0.0, 1.0);
    if l10_score < 0.20 && l12_score < 0.20 {
        raw.min(0.35)
    } else {
        raw
    }
}
