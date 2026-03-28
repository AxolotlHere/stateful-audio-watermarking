//! `audio::layers` — 15 heterogeneous watermark transformation layers.
//!
//! # Module layout
//!
//! ```
//! layers/
//!   layer_trait.rs   – the `Layer` trait
//!   layer1.rs  ..  layer15.rs  – individual implementations
//!   permutation.rs  – key-driven layer-order permutation engine
//!   chunker.rs      – key-derived chunk splitter
//!   mod.rs          – this file; public API surface
//! ```
//!
//! # Typical usage
//!
//! ```rust
//! use audio::layers::{build_layers, permute_layers, KeyedChunker};
//!
//! let key: u64 = 0xDEAD_BEEF_1234_5678;
//! let layers = build_layers(key, sample_rate);
//! let order  = permute_layers(key);          // [0..15] shuffled
//! let chunker = KeyedChunker::new(key, sample_rate);
//!
//! for chunk in chunker.chunks_mut(&mut samples) {
//!     for &idx in &order {
//!         layers[idx].apply(chunk, sample_rate);
//!     }
//! }
//! ```

pub mod layer_trait;

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

/// Construct all 15 layers using sub-keys derived from `key`.
///
/// Each layer receives a different byte or u64 slice of the key so that
/// their parameters are independent and spread across the key space.
pub fn build_layers(key: u64, sample_rate: u32) -> Vec<Box<dyn Layer>> {
    // Derive 15 independent bytes/u64 values from the master key.
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

/// Derive a pseudo-random byte from `key` at position `i`.
fn derive_byte(key: u64, i: u64) -> u8 {
    let mut h = key ^ (i.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    // One round of xorshift-mix
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    (h & 0xFF) as u8
}

/// Derive a full u64 from `key` at position `i`.
fn derive_u64(key: u64, i: u64) -> u64 {
    let mut h = key ^ (i.wrapping_mul(0x517c_c1b7_2722_0a95));
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    h
}
