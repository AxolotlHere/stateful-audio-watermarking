//! `audio::layers` — 15 heterogeneous watermark transformation layers
//! with chained parameter derivation.
//!
//! # How the chain works
//!
//! Each layer's key_byte is derived from:
//!   - The master key K
//!   - The chunk index (unique per chunk)
//!   - The audio state AFTER the previous layer ran (RMS + spectral centroid)
//!
//! This means every layer's parameters depend on every previous layer's
//! effect on the audio. A wrong key produces wrong parameters at layer 0,
//! which produce wrong audio state, which cascades into increasingly wrong
//! parameters for all subsequent layers.
//!
//! # Usage
//!
//! ```rust
//! use audio::layers::{permute_layers, KeyedChunker, apply_chained_layers};
//!
//! let key: u64 = 0xDEAD_BEEF_1234_5678;
//! let order   = permute_layers(key);
//! let chunker = KeyedChunker::new(key, sample_rate);
//!
//! for (chunk_idx, chunk) in chunker.iter_chunks_mut(&mut samples).enumerate() {
//!     apply_chained_layers(chunk, sample_rate, key, &order, chunk_idx);
//!     chunker.apply_boundary_fade(chunk);
//! }
//! ```

pub mod layer_trait;
pub mod chain;

pub mod layer1;
pub mod layer2;
pub mod layer3;
pub mod layer4;
pub mod layer5;
pub mod layer6;
pub mod layer7;
pub mod layer8;
pub mod layer9;
pub mod layer10;
pub mod layer11;
pub mod layer12;
pub mod layer13;
pub mod layer14;
pub mod layer15;

pub mod permutation;
pub mod chunker;

pub use layer_trait::Layer;
pub use permutation::permute_layers;
pub use chunker::KeyedChunker;
pub use chain::{ChainState, audio_hash};

use layer1::AmplitudeScalingLayer;
use layer2::MicroTimeShiftLayer;
use layer3::EnvelopeShapingLayer;
use layer4::BandLimitedGainLayer;
use layer5::HighFrequencyEmphasisLayer;
use layer6::NarrowbandAttenuationLayer;
use layer7::PhasePerturbationLayer;
use layer8::LocalSampleReorderingLayer;
use layer9::EnergyRedistributionLayer;
use layer10::NoiseShapingLayer;
use layer11::ControlledNonlinearLayer;
use layer12::LogisticMapLayer;
use layer13::CombFilterLayer;
use layer14::SpectralTiltLayer;
use layer15::TemporalVarianceLayer;

// ─── Chained pipeline ────────────────────────────────────────────────────────

/// Apply all 15 layers to `chunk` in the permuted order, with each layer's
/// parameters derived from the chain state that evolves after every layer.
///
/// This is the primary embedding function. The chain ensures every layer's
/// parameters depend on all previous layers' audio output — a wrong key
/// cascades into total parameter divergence.
pub fn apply_chained_layers(
    samples: &mut [f32],
    sample_rate: u32,
    key: u64,
    order: &[usize; 15],
    chunk_index: usize,
) {
    let mut state = ChainState::new(key, chunk_index);

    for (slot, &layer_idx) in order.iter().enumerate() {
        // Derive this layer's key_byte from the current chain state
        let key_byte = state.derive_byte(slot);
        let key_u64  = state.derive_u64(slot);

        // Build and apply the layer with chained parameters
        apply_single_layer(samples, sample_rate, layer_idx, key_byte, key_u64);

        // Advance chain using audio state AFTER this layer ran
        state = state.advance(samples, sample_rate);
    }
}

/// Apply a single layer by index using the given key_byte/key_u64.
/// Used by both the embedder and extractor.
pub fn apply_single_layer(
    samples: &mut [f32],
    sample_rate: u32,
    layer_idx: usize,
    key_byte: u8,
    key_u64: u64,
) {
    match layer_idx {
        0  => AmplitudeScalingLayer::new(key_byte).apply(samples, sample_rate),
        1  => MicroTimeShiftLayer::new(key_byte).apply(samples, sample_rate),
        2  => EnvelopeShapingLayer::new(key_byte).apply(samples, sample_rate),
        3  => BandLimitedGainLayer::new(key_byte, sample_rate).apply(samples, sample_rate),
        4  => HighFrequencyEmphasisLayer::new(key_byte, sample_rate).apply(samples, sample_rate),
        5  => NarrowbandAttenuationLayer::new(key_byte, sample_rate).apply(samples, sample_rate),
        6  => PhasePerturbationLayer::new(key_byte, sample_rate).apply(samples, sample_rate),
        7  => LocalSampleReorderingLayer::new(key_byte).apply(samples, sample_rate),
        8  => EnergyRedistributionLayer::new(key_byte).apply(samples, sample_rate),
        9  => NoiseShapingLayer::new(key_u64).apply(samples, sample_rate),
        10 => ControlledNonlinearLayer::new(key_byte).apply(samples, sample_rate),
        11 => LogisticMapLayer::new(key_byte).apply(samples, sample_rate),
        12 => CombFilterLayer::new(key_byte, sample_rate).apply(samples, sample_rate),
        13 => SpectralTiltLayer::new(key_byte, sample_rate).apply(samples, sample_rate),
        14 => TemporalVarianceLayer::new(key_byte).apply(samples, sample_rate),
        _  => {}
    }
}

// ─── Legacy API (kept for extractor compatibility) ───────────────────────────

/// Build all 15 layers with parameters derived directly from key (no chain).
/// Used only by the old blind extractor. Prefer `apply_chained_layers`.
pub fn build_layers(key: u64, sample_rate: u32) -> Vec<Box<dyn Layer>> {
    let bytes: Vec<u8> = (0u64..15)
        .map(|i| derive_byte(key, i))
        .collect();

    vec![
        Box::new(AmplitudeScalingLayer::new(bytes[0])),
        Box::new(MicroTimeShiftLayer::new(bytes[1])),
        Box::new(EnvelopeShapingLayer::new(bytes[2])),
        Box::new(BandLimitedGainLayer::new(bytes[3], sample_rate)),
        Box::new(HighFrequencyEmphasisLayer::new(bytes[4], sample_rate)),
        Box::new(NarrowbandAttenuationLayer::new(bytes[5], sample_rate)),
        Box::new(PhasePerturbationLayer::new(bytes[6], sample_rate)),
        Box::new(LocalSampleReorderingLayer::new(bytes[7])),
        Box::new(EnergyRedistributionLayer::new(bytes[8])),
        Box::new(NoiseShapingLayer::new(derive_u64(key, 9))),
        Box::new(ControlledNonlinearLayer::new(bytes[10])),
        Box::new(LogisticMapLayer::new(bytes[11])),
        Box::new(CombFilterLayer::new(bytes[12], sample_rate)),
        Box::new(SpectralTiltLayer::new(bytes[13], sample_rate)),
        Box::new(TemporalVarianceLayer::new(bytes[14])),
    ]
}

// ─── Key derivation helpers ──────────────────────────────────────────────────

fn derive_byte(key: u64, i: u64) -> u8 {
    (chain::splitmix64(key ^ i.wrapping_mul(0x9e37_79b9_7f4a_7c15)) & 0xFF) as u8
}

fn derive_u64(key: u64, i: u64) -> u64 {
    chain::splitmix64(key ^ i.wrapping_mul(0x517c_c1b7_2722_0a95))
}
