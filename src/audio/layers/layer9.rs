//! Layer 9 – Energy Redistribution (DC Offset Nudge)
//!
//! WHAT  : Adds a tiny, key-derived DC offset (≤ 0.0005) to every sample.
//!         DC offset shifts the statistical mean of the waveform without
//!         audible tonal change.
//!
//! SAFE  : ±0.0005 DC offset is 66 dB below full scale; completely
//!         imperceptible and does not cause inter-sample clipping.

use super::layer_trait::Layer;

pub struct EnergyRedistributionLayer {
    dc_offset: f32,
}

impl EnergyRedistributionLayer {
    pub fn new(key_byte: u8) -> Self {
        // Map to [-0.0005, +0.0005]
        let dc_offset = (key_byte as f32 / 255.0) * 0.001 - 0.0005;
        Self { dc_offset }
    }
}

impl Layer for EnergyRedistributionLayer {
    fn name(&self) -> &'static str {
        "EnergyRedistribution"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        for s in samples.iter_mut() {
            *s = (*s + self.dc_offset).clamp(-1.0, 1.0);
        }
    }
}
