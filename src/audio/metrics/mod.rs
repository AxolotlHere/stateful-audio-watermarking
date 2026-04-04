//! `audio::metrics` — patent-oriented watermark quality metrics.

mod report;

pub use report::{BerMetric, LayerDistortionMetric, MetricsReport};

use crate::audio::extractor::Fingerprint;
use crate::audio::io::read_wav;
use crate::audio::layers::{apply_single_layer, build_layers, permute_layers, ChainState, KeyedChunker};

const L10_NOISE_SHAPING_AMPLITUDE: f32 = 0.00035;
const L12_LOGISTIC_MAP_AMPLITUDE: f32 = 0.00045;

pub fn run_metrics(
    original_wav: &str,
    watermarked_wav: &str,
    fingerprint_path: &str,
    key: u64,
) -> Result<MetricsReport, String> {
    let original = read_wav(original_wav);
    let watermarked = read_wav(watermarked_wav);

    if original.sample_rate != watermarked.sample_rate {
        return Err(format!(
            "Sample-rate mismatch: original={} Hz, watermarked={} Hz",
            original.sample_rate,
            watermarked.sample_rate,
        ));
    }

    let aligned_samples = original.samples.len().min(watermarked.samples.len());
    if aligned_samples == 0 {
        return Err("No overlapping samples between original and watermarked audio".to_string());
    }

    let original_samples = &original.samples[..aligned_samples];
    let watermarked_samples = &watermarked.samples[..aligned_samples];

    let fp = Fingerprint::load(fingerprint_path)
        .map_err(|e| format!("Could not load fingerprint '{fingerprint_path}': {e}"))?;
    let full = distortion_stats(original_samples, watermarked_samples);
    let active = active_region_distortion_stats(
        original_samples,
        watermarked_samples,
        original.sample_rate,
        key,
    );
    let ber_metrics = compute_spread_presence_ber(watermarked_samples, original.sample_rate, key, &fp);
    let layer_distortion = profile_layer_distortion(original_samples, original.sample_rate, key);

    Ok(MetricsReport {
        key,
        sample_rate: original.sample_rate,
        aligned_samples,
        active_region_samples: active.samples,
        full_original_peak: full.original_peak,
        full_original_rms: full.original_rms,
        full_error_rms: full.error_rms,
        full_snr_db: full.snr_db,
        full_psnr_db: full.psnr_db,
        active_original_peak: active.original_peak,
        active_original_rms: active.original_rms,
        active_error_rms: active.error_rms,
        active_snr_db: active.snr_db,
        active_psnr_db: active.psnr_db,
        ber_metrics,
        layer_distortion,
    })
}

fn compute_spread_presence_ber(
    wm_samples: &[f32],
    sample_rate: u32,
    key: u64,
    fp: &Fingerprint,
) -> Vec<BerMetric> {
    let order = permute_layers(key);
    let chunker = KeyedChunker::new(key, sample_rate);

    let mut l10_bits = 0usize;
    let mut l10_errors = 0usize;
    let mut l10_corr_sum = 0.0_f32;

    let mut l12_bits = 0usize;
    let mut l12_errors = 0usize;
    let mut l12_corr_sum = 0.0_f32;

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

        let Some(orig_chunk) = fp.chunks.get(chunk_idx) else {
            continue;
        };

        let shared_len = wm_chunk.len().min(orig_chunk.orig_samples.len());
        for (region_idx, (start, end)) in chunker
            .watermark_windows(chunk_idx, shared_len)
            .into_iter()
            .enumerate()
        {
            let wm_region = &wm_chunk[start..end];
            let orig_region = &orig_chunk.orig_samples[start..end];
            let mut sim_region = orig_region.to_vec();
            let mut state = ChainState::new(key, keyed_region_seed(chunk_idx, region_idx));

            for (slot, &layer_idx) in order.iter().enumerate() {
                let key_byte = state.derive_byte(slot);
                let key_u64 = state.derive_u64(slot);
                let sim_before = sim_region.clone();

                apply_single_layer(&mut sim_region, sample_rate, layer_idx, key_byte, key_u64);

                match layer_idx {
                    9 => {
                        let corr = matched_filter_corr_l10(wm_region, &sim_before, key_u64);
                        l10_bits += 1;
                        l10_corr_sum += corr;
                        if corr <= 0.0 {
                            l10_errors += 1;
                        }
                    }
                    11 => {
                        let corr = matched_filter_corr_l12(wm_region, &sim_before, key_byte);
                        l12_bits += 1;
                        l12_corr_sum += corr;
                        if corr <= 0.0 {
                            l12_errors += 1;
                        }
                    }
                    _ => {}
                }

                state = state.advance(&sim_region, sample_rate);
            }
        }
    }

    vec![
        BerMetric {
            layer_index: 10,
            layer_name: "NoiseShaping",
            total_bits: l10_bits,
            error_bits: l10_errors,
            ber: safe_ratio(l10_errors, l10_bits),
            mean_corr: safe_mean(l10_corr_sum, l10_bits),
        },
        BerMetric {
            layer_index: 12,
            layer_name: "LogisticMapModulation",
            total_bits: l12_bits,
            error_bits: l12_errors,
            ber: safe_ratio(l12_errors, l12_bits),
            mean_corr: safe_mean(l12_corr_sum, l12_bits),
        },
    ]
}

fn matched_filter_corr_l10(wm: &[f32], sim_before: &[f32], key_u64: u64) -> f32 {
    let n = wm.len().min(sim_before.len());
    if n == 0 {
        return 0.0;
    }

    let mut state = key_u64;
    let seq: Vec<f32> = (0..n)
        .map(|_| next_xorshift(&mut state) * L10_NOISE_SHAPING_AMPLITUDE)
        .collect();
    matched_filter_corr(wm, sim_before, &seq)
}

fn matched_filter_corr_l12(wm: &[f32], sim_before: &[f32], key_byte: u8) -> f32 {
    let n = wm.len().min(sim_before.len());
    if n == 0 {
        return 0.0;
    }

    let t = key_byte as f64 / 255.0;
    let r = 3.9 + t * 0.099;
    let mut x = 0.1 + t * 0.8;
    let seq: Vec<f32> = (0..n)
        .map(|_| {
            x = r * x * (1.0 - x);
            ((x - 0.5) * 2.0) as f32 * L12_LOGISTIC_MAP_AMPLITUDE
        })
        .collect();
    matched_filter_corr(wm, sim_before, &seq)
}

fn matched_filter_corr(wm: &[f32], sim_before: &[f32], seq: &[f32]) -> f32 {
    let n = wm.len().min(sim_before.len()).min(seq.len());
    if n == 0 {
        return 0.0;
    }

    let residual: Vec<f32> = wm[..n]
        .iter()
        .zip(&sim_before[..n])
        .map(|(w, s)| w - s)
        .collect();
    let dot: f64 = residual
        .iter()
        .zip(&seq[..n])
        .map(|(r, s)| *r as f64 * *s as f64)
        .sum();
    let seq_nrg: f64 = seq[..n].iter().map(|s| *s as f64 * *s as f64).sum();

    if seq_nrg < 1e-30 {
        return 0.0;
    }

    (dot / seq_nrg) as f32
}

fn profile_layer_distortion(
    original: &[f32],
    sample_rate: u32,
    key: u64,
) -> Vec<LayerDistortionMetric> {
    let chunker = KeyedChunker::new(key, sample_rate);
    let order = permute_layers(key);
    let layers = build_layers(key, sample_rate);
    let total_chunks = chunker.chunk_count(original.len());
    let mut offset = chunker.chunk_offset_samples();

    let mut layer_delta_sq = [0.0_f64; 15];
    let mut layer_cum_sq = [0.0_f64; 15];
    let mut total_active_samples = 0usize;

    for chunk_idx in 0..total_chunks {
        if offset >= original.len() {
            break;
        }

        let end = (offset + chunker.chunk_size_samples()).min(original.len());
        if chunker.should_watermark_chunk(chunk_idx, total_chunks) {
            let chunk = &original[offset..end];
            for (region_idx, (start, stop)) in chunker
                .watermark_windows(chunk_idx, chunk.len())
                .into_iter()
                .enumerate()
            {
                let original_region = &chunk[start..stop];
                let mut sim_region = original_region.to_vec();
                let mut state = ChainState::new(key, keyed_region_seed(chunk_idx, region_idx));
                let mut prev_stage = original_region.to_vec();

                total_active_samples += original_region.len();

                for (slot, &layer_idx) in order.iter().enumerate() {
                    let key_byte = state.derive_byte(slot);
                    let key_u64 = state.derive_u64(slot);
                    apply_single_layer(&mut sim_region, sample_rate, layer_idx, key_byte, key_u64);

                    let mut stage_output = sim_region.clone();
                    chunker.blend_region_edges(original_region, &mut stage_output);

                    layer_delta_sq[layer_idx] += sum_sq_diff(&prev_stage, &stage_output);
                    layer_cum_sq[layer_idx] += sum_sq_diff(original_region, &stage_output);

                    prev_stage = stage_output;
                    state = state.advance(&sim_region, sample_rate);
                }
            }
        }
        offset = end;
    }

    order.iter()
        .enumerate()
        .map(|(slot, &layer_idx)| LayerDistortionMetric {
            slot_index: slot,
            layer_index: layer_idx + 1,
            layer_name: layers[layer_idx].name(),
            incremental_error_rms: rms_from_sum(layer_delta_sq[layer_idx], total_active_samples),
            cumulative_error_rms: rms_from_sum(layer_cum_sq[layer_idx], total_active_samples),
        })
        .collect()
}

#[derive(Clone, Copy)]
struct DistortionStats {
    samples: usize,
    original_peak: f32,
    original_rms: f32,
    error_rms: f32,
    snr_db: f32,
    psnr_db: f32,
}

fn active_region_distortion_stats(
    original: &[f32],
    watermarked: &[f32],
    sample_rate: u32,
    key: u64,
) -> DistortionStats {
    let chunker = KeyedChunker::new(key, sample_rate);
    let aligned_samples = original.len().min(watermarked.len());
    let total_chunks = chunker.chunk_count(aligned_samples);
    let mut offset = chunker.chunk_offset_samples();

    let mut sum_sig_sq = 0.0_f64;
    let mut sum_err_sq = 0.0_f64;
    let mut peak = 0.0_f32;
    let mut count = 0usize;

    for chunk_idx in 0..total_chunks {
        if offset >= aligned_samples {
            break;
        }

        let end = (offset + chunker.chunk_size_samples()).min(aligned_samples);
        if chunker.should_watermark_chunk(chunk_idx, total_chunks) {
            let chunk_len = end - offset;
            for (start, stop) in chunker.watermark_windows(chunk_idx, chunk_len) {
                let abs_start = offset + start;
                let abs_end = offset + stop;
                for i in abs_start..abs_end.min(aligned_samples) {
                    let s = original[i];
                    let e = original[i] - watermarked[i];
                    sum_sig_sq += (s as f64) * (s as f64);
                    sum_err_sq += (e as f64) * (e as f64);
                    peak = peak.max(s.abs());
                    count += 1;
                }
            }
        }
        offset = end;
    }

    stats_from_sums(sum_sig_sq, sum_err_sq, peak, count)
}

fn distortion_stats(original: &[f32], watermarked: &[f32]) -> DistortionStats {
    let n = original.len().min(watermarked.len());
    let mut sum_sig_sq = 0.0_f64;
    let mut sum_err_sq = 0.0_f64;
    let mut peak = 0.0_f32;

    for i in 0..n {
        let s = original[i];
        let e = original[i] - watermarked[i];
        sum_sig_sq += (s as f64) * (s as f64);
        sum_err_sq += (e as f64) * (e as f64);
        peak = peak.max(s.abs());
    }

    stats_from_sums(sum_sig_sq, sum_err_sq, peak, n)
}

fn stats_from_sums(sum_sig_sq: f64, sum_err_sq: f64, peak: f32, count: usize) -> DistortionStats {
    if count == 0 {
        return DistortionStats {
            samples: 0,
            original_peak: 0.0,
            original_rms: 0.0,
            error_rms: 0.0,
            snr_db: 0.0,
            psnr_db: 0.0,
        };
    }

    let original_rms = (sum_sig_sq / count as f64).sqrt() as f32;
    let error_rms = (sum_err_sq / count as f64).sqrt() as f32;
    DistortionStats {
        samples: count,
        original_peak: peak,
        original_rms,
        error_rms,
        snr_db: ratio_db(original_rms, error_rms),
        psnr_db: ratio_db(peak, error_rms),
    }
}

fn sum_sq_diff(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut sum = 0.0_f64;
    for i in 0..n {
        let err = a[i] - b[i];
        sum += (err as f64) * (err as f64);
    }
    sum
}

fn rms_from_sum(sum_sq: f64, count: usize) -> f32 {
    if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt() as f32
    }
}

fn ratio_db(signal: f32, noise: f32) -> f32 {
    if noise <= 1e-30 {
        return f32::INFINITY;
    }
    if signal <= 1e-30 {
        return f32::NEG_INFINITY;
    }
    20.0 * (signal / noise).log10()
}

fn safe_ratio(numer: usize, denom: usize) -> f32 {
    if denom == 0 {
        0.0
    } else {
        numer as f32 / denom as f32
    }
}

fn safe_mean(sum: f32, count: usize) -> f32 {
    if count == 0 {
        0.0
    } else {
        sum / count as f32
    }
}

fn next_xorshift(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x as i64 as f32) / (i64::MAX as f32)
}

fn keyed_region_seed(chunk_idx: usize, region_idx: usize) -> usize {
    chunk_idx
        .wrapping_mul(8)
        .wrapping_add(region_idx)
}
