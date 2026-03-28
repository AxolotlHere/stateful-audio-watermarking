//! Layer 1 – Subtle Amplitude Scaling
//!
//! WHAT  : Multiplies every sample by a gain factor very close to 1.0.
//!         The factor is derived from the watermark key so it is unique
//!         per session but constant within a chunk.
//! SAFE  : |gain - 1.0| ≤ 0.003 → maximum 0.026 dB deviation; well below
//!         the ~1 dB JND (just-noticeable difference) for loudness.

use super::layer_trait::Layer;

pub struct AmplitudeScalingLayer {
    /// Gain offset in [-0.003, +0.003]; derived from key at construction.
    gain_offset: f32,
}

impl AmplitudeScalingLayer {
    /// `key_byte` – any byte derived from the watermark key (0–255).
    pub fn new(key_byte: u8) -> Self {
        // Map 0..=255 → [-0.003, +0.003]
        let gain_offset = (key_byte as f32 / 255.0) * 0.006 - 0.003;
        Self { gain_offset }
    }
}

impl Layer for AmplitudeScalingLayer {
    fn name(&self) -> &'static str {
        "AmplitudeScaling"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let gain = 1.0 + self.gain_offset;
        for s in samples.iter_mut() {
            *s = (*s * gain).clamp(-1.0, 1.0);
        }
    }
}
