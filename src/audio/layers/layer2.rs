//! Layer 2 – Fractional Micro Delay
//!
//! WHAT  : Applies a keyed fractional delay smaller than half a sample
//!         using linear interpolation. This preserves waveform continuity
//!         instead of inserting zeros and dropping content.
//!
//! SAFE  : Delay is only 0.05–0.45 samples, so the perturbation is a very
//!         small local timing nudge with bounded amplitude error.

use super::layer_trait::Layer;

pub struct MicroTimeShiftLayer {
    frac_delay: f32,
}

impl MicroTimeShiftLayer {
    /// `key_byte` – 0..=255; mapped to a 0.05..0.45 sample delay.
    pub fn new(key_byte: u8) -> Self {
        let t = key_byte as f32 / 255.0;
        let frac_delay = 0.05 + t * 0.40;
        Self { frac_delay }
    }
}

impl Layer for MicroTimeShiftLayer {
    fn name(&self) -> &'static str {
        "MicroTimeShift"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        if samples.is_empty() {
            return;
        }

        let d = self.frac_delay;
        let mut prev = samples[0];
        for s in samples.iter_mut() {
            let x = *s;
            *s = ((1.0 - d) * x + d * prev).clamp(-1.0, 1.0);
            prev = x;
        }
    }
}
