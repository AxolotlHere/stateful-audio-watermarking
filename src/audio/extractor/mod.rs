//! `audio::extractor` — watermark verification using chained layer replay.
//!
//! The extractor replays the exact same chain used during embedding:
//! same key + same chunk_index + same audio states = same parameters.
//! A wrong key diverges immediately at layer 0's audio state,
//! producing wrong parameters for every subsequent layer.

pub mod signals;
pub mod report;
pub mod fingerprint;
pub mod ref_signals;
pub mod key_test;

pub use report::{ExtractionResult, LayerResult};
pub use fingerprint::Fingerprint;

use crate::audio::layers::{
    apply_single_layer, permute_layers, KeyedChunker,
    chain::{ChainState, audio_hash},
};
use ref_signals::measure_layer_ref;

// ─── Reference-based extraction ──────────────────────────────────────────────

/// Verify a watermarked file using the original fingerprint.
///
/// Replays the chained embedding pipeline with the candidate key.
/// For the correct key, the chain reproduces the exact same parameters
/// used during embedding, and all layer measurements match.
/// For a wrong key, the chain diverges at layer 0 and all measurements fail.
pub fn extract_with_ref(
    wm_samples: &[f32],
    sample_rate: u32,
    key: u64,
    fp: &Fingerprint,
) -> ExtractionResult {
    use crate::audio::layers::build_layers;
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

        let orig = &fp.chunks[chunk_count % fp.chunks.len()];
        let chunk_idx = chunk_count;
        chunk_count += 1;

        // Replay the chain sequentially.
        // sim_before at each slot = audio after all prior layers with this key.
        // Correct key → sim_before matches real pre-layer audio → high residual corr.
        // Wrong key   → sim_before diverges at slot 0 → residuals uncorrelated.
        let mut state = ChainState::new(key, chunk_idx);
        let mut sim_chunk: Vec<f32> = orig.orig_samples.clone();

        for (slot, &layer_idx) in order.iter().enumerate() {
            let key_byte = state.derive_byte(slot);
            let key_u64  = state.derive_u64(slot);

            let sim_before = sim_chunk.clone();
            apply_single_layer(&mut sim_chunk, sample_rate, layer_idx, key_byte, key_u64);

            let score = measure_layer_ref(
                wm_chunk,
                orig,
                &sim_before,
                &sim_chunk,
                sample_rate,
                layer_idx,
                key_byte,
                key_u64,
            );
            layer_accum[layer_idx].push(score);

            state = state.advance(&sim_chunk, sample_rate);
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
        watermark_detected: confidence >= 0.55,
        confidence,
        layer_results,
        detected_count,
    }
}

fn detection_threshold(layer_idx: usize) -> f32 {
    match layer_idx {
        9 | 11 => 0.35,  // L10, L12 — spread-spectrum, high weight
        _      => 0.30,
    }
}

fn weighted_confidence(results: &[LayerResult]) -> f32 {
    // All 15 layers contribute — none zeroed.
    // Weights reflect how strongly each layer's score is key-discriminating
    // WITH the chain (wrong key → wrong chain → wrong expected values everywhere).
    const WEIGHTS: [f32; 15] = [
        0.6,  // L1  AmplitudeScaling
        0.9,  // L2  MicroTimeShift
        0.7,  // L3  EnvelopeShaping
        0.8,  // L4  BandLimitedGain
        0.2,  // L5  HFEmphasis        ← low: near-zero residual on real music
        0.8,  // L6  NarrowbandAtten
        0.9,  // L7  PhasePerturbation
        0.7,  // L8  SampleReordering
        1.0,  // L9  EnergyRedistrib
        1.2,  // L10 NoiseShaping      ← spread-spectrum
        0.7,  // L11 ControlledNonlin
        1.2,  // L12 LogisticMap       ← spread-spectrum
        0.8,  // L13 CombFilter
        0.8,  // L14 SpectralTilt
        0.7,  // L15 TemporalVariance
    ];
    let total: f32 = WEIGHTS.iter().sum();
    let scored: f32 = results.iter().zip(WEIGHTS.iter())
        .map(|(r, &w)| r.score * w)
        .sum();
    (scored / total).clamp(0.0, 1.0)
}
