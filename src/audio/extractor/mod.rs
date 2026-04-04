//! `audio::extractor` — watermark verification using chained layer replay.
//!
//! Detection is intentionally sparse and key-dependent:
//!   - the candidate key defines chunk alignment and chunk selection,
//!   - only keyed bands and autocorrelation lags are checked,
//!   - simulated and actual residuals must align strongly,
//!   - confidence is based on strong-match counts, not global averaging.

pub mod signals;
pub mod report;
pub mod fingerprint;
pub mod ref_signals;
pub mod key_test;

pub use report::{ExtractionResult, LayerResult};
pub use fingerprint::Fingerprint;

use crate::audio::layers::{apply_single_layer, permute_layers, KeyedChunker};
use crate::audio::layers::chain::{ChainState, splitmix64};
use fingerprint::{analyze_chunk, ChunkFingerprint, N_AC_LAGS, N_BANDS, N_ENERGY_LAGS};
use ref_signals::measure_layer_ref;

const BAND_SELECTIONS: usize = 3;
const AC_SELECTIONS: usize = 4;
const ENERGY_SELECTIONS: usize = 4;

const MIN_ALIGNMENT_SCORE: f32 = 0.18;
const MIN_REGION_RATIO: f32 = 0.25;
const MIN_STRONG_LAYER_HITS: usize = 2;
const MIN_SPREAD_HITS: usize = 1;
const SPREAD_RESCUE_MEAN: f32 = 0.45;
const SPREAD_RESCUE_ALIGNMENT: f32 = 0.04;
const SPREAD_RESCUE_REGION_RATIO: f32 = 0.08;
const SPREAD_PEAK_RESCUE: f32 = 0.28;
const MIN_RESCUE_SUPPORT: f32 = 0.10;

/// Verify a watermarked file using the original fingerprint.
///
/// Only sparsely selected, key-addressed chunks are evaluated. Wrong keys
/// choose the wrong offset and the wrong subset of chunks, then fail the
/// residual-alignment and keyed-region gates.
pub fn extract_with_ref(
    wm_samples: &[f32],
    sample_rate: u32,
    key: u64,
    fp: &Fingerprint,
) -> ExtractionResult {
    use crate::audio::layers::build_layers;

    let layers = build_layers(key, sample_rate);
    let order = permute_layers(key);
    let chunker = KeyedChunker::new(key, sample_rate);

    let mut layer_pass_counts = [0usize; 15];
    let mut layer_observed = [0usize; 15];
    let mut selected_regions = 0usize;
    let mut strong_region_hits = 0usize;
    let mut chunk_support_sum = 0.0_f32;

    let mut offset = chunker.chunk_offset_samples();
    let total_chunks = chunker.chunk_count(wm_samples.len());

    for chunk_idx in 0..total_chunks {
        if offset >= wm_samples.len() {
            break;
        }
        let end = (offset + chunker.chunk_size_samples()).min(wm_samples.len());
        let wm_chunk = &wm_samples[offset..end];
        offset = end;

        if !chunker.should_watermark_chunk(chunk_idx, total_chunks) {
            continue;
        }

        let Some(orig) = fp.chunks.get(chunk_idx) else {
            continue;
        };

        let shared_len = wm_chunk.len().min(orig.orig_samples.len());
        for (region_idx, (start, end)) in chunker
            .watermark_windows(chunk_idx, orig.orig_samples.len())
            .into_iter()
            .filter(|&(_, end)| end <= shared_len)
            .enumerate()
        {
            selected_regions += 1;

            let wm_region = &wm_chunk[start..end];
            let orig_region = &orig.orig_samples[start..end];
            let mut region_fp = analyze_chunk(orig_region, sample_rate);
            region_fp.orig_samples = orig_region.to_vec();

            let mut state = ChainState::new(key, keyed_region_seed(chunk_idx, region_idx));
            let mut sim_region = orig_region.to_vec();
            let mut per_layer_scores = [0.0_f32; 15];

            for (slot, &layer_idx) in order.iter().enumerate() {
                let key_byte = state.derive_byte(slot);
                let key_u64 = state.derive_u64(slot);

                let sim_before = sim_region.clone();
                apply_single_layer(&mut sim_region, sample_rate, layer_idx, key_byte, key_u64);

                per_layer_scores[layer_idx] = measure_layer_ref(
                    wm_region,
                    &region_fp,
                    &sim_before,
                    &sim_region,
                    sample_rate,
                    layer_idx,
                    key_byte,
                    key_u64,
                );

                state = state.advance(&sim_region, sample_rate);
            }

            chunker.blend_region_edges(orig_region, &mut sim_region);

            let alignment = residual_alignment(wm_region, orig_region, &sim_region);
            let region_ratio = keyed_region_ratio(
                key,
                keyed_region_seed(chunk_idx, region_idx),
                &region_fp,
                wm_region,
                &sim_region,
                sample_rate,
            );
            let mean_layer_score = per_layer_scores.iter().sum::<f32>() / per_layer_scores.len() as f32;
            let spread_mean = (per_layer_scores[9] + per_layer_scores[11]) * 0.5;
            let spread_peak = per_layer_scores[9].max(per_layer_scores[11]);
            let strong_layer_hits = per_layer_scores
                .iter()
                .enumerate()
                .filter(|(idx, score)| **score >= strong_layer_threshold(*idx))
                .count();
            let spread_hits = [9usize, 11usize]
                .iter()
                .filter(|&&idx| per_layer_scores[idx] >= strong_layer_threshold(idx))
                .count();

            let chunk_support =
                0.35 * alignment + 0.25 * region_ratio + 0.25 * spread_mean + 0.15 * mean_layer_score;
            chunk_support_sum += chunk_support;

            let chunk_accepted = chunk_support >= 0.35
                || (
                    alignment >= MIN_ALIGNMENT_SCORE
                    && region_ratio >= MIN_REGION_RATIO
                    && strong_layer_hits >= MIN_STRONG_LAYER_HITS
                    && spread_hits >= MIN_SPREAD_HITS
                )
                || (
                    spread_mean >= SPREAD_RESCUE_MEAN
                    && spread_hits >= MIN_SPREAD_HITS
                    && strong_layer_hits >= 1
                    && (alignment >= SPREAD_RESCUE_ALIGNMENT
                        || region_ratio >= SPREAD_RESCUE_REGION_RATIO)
                )
                || (
                    spread_peak >= SPREAD_PEAK_RESCUE
                    && spread_hits >= MIN_SPREAD_HITS
                    && chunk_support >= MIN_RESCUE_SUPPORT
                );

            for idx in 0..15 {
                layer_observed[idx] += 1;
                if chunk_accepted && per_layer_scores[idx] >= relaxed_layer_threshold(idx) {
                    layer_pass_counts[idx] += 1;
                }
            }

            if chunk_accepted {
                strong_region_hits += 1;
            }
        }
    }

    build_result(
        key,
        sample_rate,
        total_chunks,
        selected_regions,
        strong_region_hits,
        chunk_support_sum,
        &layer_pass_counts,
        &layer_observed,
        &layers,
    )
}

fn build_result(
    key: u64,
    sample_rate: u32,
    chunk_count: usize,
    selected_chunks: usize,
    strong_chunk_hits: usize,
    chunk_support_sum: f32,
    layer_pass_counts: &[usize; 15],
    layer_observed: &[usize; 15],
    layers: &[Box<dyn crate::audio::layers::Layer>],
) -> ExtractionResult {
    let layer_results: Vec<LayerResult> = (0..15)
        .map(|idx| {
            let score = if layer_observed[idx] == 0 {
                0.0
            } else {
                layer_pass_counts[idx] as f32 / layer_observed[idx] as f32
            };
            LayerResult {
                layer_index: idx + 1,
                layer_name: layers[idx].name(),
                score,
                detected: score >= detection_threshold(idx),
            }
        })
        .collect();

    let detected_count = layer_results.iter().filter(|r| r.detected).count();
    let confidence = overall_confidence(
        &layer_results,
        strong_chunk_hits,
        selected_chunks,
        chunk_support_sum,
    );

    ExtractionResult {
        key,
        chunk_count,
        sample_rate,
        watermark_detected: confidence >= 0.62,
        confidence,
        layer_results,
        detected_count,
    }
}

fn strong_layer_threshold(layer_idx: usize) -> f32 {
    // Raw (uncalibrated) residual corr thresholds.
    // Layers with strong geometric effects (L2 shift, L7 allpass, L11 tanh)
    // hit 0.6-1.0; spread-spectrum (L10/L12) use matched-filter which also
    // reaches 0.5-1.0 for correct key. Most subtle layers sit at 0.05-0.30.
    match layer_idx {
        1 | 6 | 8 | 10 => 0.45,   // strong geometric layers
        9 | 11 => 0.40,            // spread-spectrum matched filter
        3 | 7 | 12 | 13 | 14 => 0.05, // subtle layers that never reach 0.20 raw corr
        _ => 0.20,                 // subtle filter/gain layers
    }
}

fn relaxed_layer_threshold(layer_idx: usize) -> f32 {
    match layer_idx {
        9 | 11 => 0.30,
        _ => 0.05,
    }
}

fn detection_threshold(layer_idx: usize) -> f32 {
    match layer_idx {
        9 | 11 => 0.35,
        _ => 0.45,
    }
}

fn overall_confidence(
    results: &[LayerResult],
    strong_chunk_hits: usize,
    selected_chunks: usize,
    chunk_support_sum: f32,
) -> f32 {
    if selected_chunks == 0 {
        return 0.0;
    }

    let chunk_ratio = strong_chunk_hits as f32 / selected_chunks as f32;
    let strong_ratio = chunk_ratio * chunk_ratio;
    let support_ratio = chunk_support_sum / selected_chunks as f32;
    let spread_ratio = (results[9].score + results[11].score) * 0.5;
    let layer_ratio = results.iter().filter(|r| r.detected).count() as f32 / results.len() as f32;

    (0.40 * support_ratio + 0.30 * strong_ratio + 0.15 * spread_ratio + 0.15 * layer_ratio)
        .clamp(0.0, 1.0)
}

fn residual_alignment(wm_chunk: &[f32], orig_chunk: &[f32], sim_chunk: &[f32]) -> f32 {
    let n = wm_chunk.len().min(orig_chunk.len()).min(sim_chunk.len());
    if n == 0 {
        return 0.0;
    }

    let actual: Vec<f32> = wm_chunk[..n]
        .iter()
        .zip(&orig_chunk[..n])
        .map(|(w, o)| w - o)
        .collect();
    let predicted: Vec<f32> = sim_chunk[..n]
        .iter()
        .zip(&orig_chunk[..n])
        .map(|(s, o)| s - o)
        .collect();

    normalised_correlation(&actual, &predicted).max(0.0)
}

fn keyed_region_ratio(
    key: u64,
    chunk_idx: usize,
    orig: &ChunkFingerprint,
    wm_chunk: &[f32],
    sim_chunk: &[f32],
    sample_rate: u32,
) -> f32 {
    let actual = analyze_chunk(wm_chunk, sample_rate);
    let predicted = analyze_chunk(sim_chunk, sample_rate);

    let band_idxs = keyed_indices(key, chunk_idx, 0x41f3, N_BANDS, BAND_SELECTIONS);
    let ac_idxs = keyed_indices(key, chunk_idx, 0x7a1d, N_AC_LAGS, AC_SELECTIONS);
    let energy_idxs = keyed_indices(key, chunk_idx, 0xbb67, N_ENERGY_LAGS, ENERGY_SELECTIONS);

    let mut strong_hits = 0usize;
    let mut total_checks = 0usize;

    for idx in band_idxs {
        if compare_delta(actual.band_rms[idx], predicted.band_rms[idx], orig.band_rms[idx], 0.20) {
            strong_hits += 1;
        }
        total_checks += 1;
    }

    for idx in ac_idxs {
        if compare_delta(actual.ac_lags[idx], predicted.ac_lags[idx], orig.ac_lags[idx], 0.30) {
            strong_hits += 1;
        }
        total_checks += 1;
    }

    for idx in energy_idxs {
        if compare_delta(
            actual.energy_ac[idx],
            predicted.energy_ac[idx],
            orig.energy_ac[idx],
            0.30,
        ) {
            strong_hits += 1;
        }
        total_checks += 1;
    }

    if total_checks == 0 {
        0.0
    } else {
        strong_hits as f32 / total_checks as f32
    }
}

fn compare_delta(actual: f32, predicted: f32, baseline: f32, min_ratio: f32) -> bool {
    let predicted_delta = predicted - baseline;
    let actual_delta = actual - baseline;

    let expected_mag = predicted_delta.abs();
    if expected_mag < 1e-6 {
        return false;
    }

    if predicted_delta.signum() != actual_delta.signum() {
        return false;
    }

    let ratio = (actual_delta.abs() / expected_mag).min(expected_mag / actual_delta.abs().max(1e-6));
    ratio >= min_ratio
}

fn keyed_indices(key: u64, chunk_idx: usize, salt: u64, total: usize, picks: usize) -> Vec<usize> {
    let picks = picks.min(total);
    let mut state = splitmix64(
        key ^ (chunk_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ salt,
    );
    let mut used = vec![false; total];
    let mut out = Vec::with_capacity(picks);

    while out.len() < picks {
        state = splitmix64(state ^ out.len() as u64);
        let idx = (state as usize) % total;
        if !used[idx] {
            used[idx] = true;
            out.push(idx);
        }
    }

    out
}

fn normalised_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let mean_a = a[..n].iter().sum::<f32>() / n as f32;
    let mean_b = b[..n].iter().sum::<f32>() / n as f32;

    let mut num = 0.0_f32;
    let mut den_a = 0.0_f32;
    let mut den_b = 0.0_f32;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    if den_a < 1e-12 || den_b < 1e-12 {
        return 0.0;
    }

    (num / (den_a.sqrt() * den_b.sqrt())).clamp(-1.0, 1.0)
}

fn keyed_region_seed(chunk_idx: usize, region_idx: usize) -> usize {
    chunk_idx.wrapping_mul(8).wrapping_add(region_idx)
}
