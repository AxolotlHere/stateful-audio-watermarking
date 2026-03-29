//! `audio::extractor` — watermark verification given a key.
//!
//! Two modes:
//!   - `extract()`          — blind (no original needed, lower accuracy)
//!   - `extract_with_ref()` — reference-based (uses .wmpf fingerprint, high accuracy)

pub mod signals;
pub mod report;
pub mod fingerprint;
pub mod ref_signals;

pub use report::{ExtractionResult, LayerResult};
pub use fingerprint::Fingerprint;

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
    const WEIGHTS: [f32; 15] = [
        0.5, 0.4, 0.6, 0.8, 0.7, 0.8, 0.9,
        0.5, 0.4, 0.9, 0.6, 0.9, 0.7, 0.8, 0.6,
    ];
    let total: f32 = WEIGHTS.iter().sum();
    let scored: f32 = results.iter().zip(WEIGHTS.iter()).map(|(r,&w)| r.score * w).sum();
    (scored / total).clamp(0.0, 1.0)
}
