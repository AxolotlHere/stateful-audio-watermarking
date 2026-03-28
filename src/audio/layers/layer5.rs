//! Layer 5 – High-Frequency Emphasis (1st-Order High Shelf)
//!
//! WHAT  : Applies a first-order high-shelf filter with +0.05 dB boost
//!         above a key-derived shelf frequency (6 kHz – 16 kHz).
//!
//! SAFE  : 0.05 dB is imperceptible; the shelf is gradual (1st order,
//!         6 dB/oct roll-off), avoiding ringing or resonance.

use super::layer_trait::Layer;
use std::f32::consts::PI;

pub struct HighFrequencyEmphasisLayer {
    b0: f32,
    b1: f32,
    a1: f32,
}

impl HighFrequencyEmphasisLayer {
    pub fn new(key_byte: u8, sample_rate: u32) -> Self {
        let t = key_byte as f32 / 255.0;
        let fc = 6000.0 + t * 10000.0; // 6–16 kHz
        let db_gain = 0.05_f32; // always a gentle boost
        let sr = sample_rate as f32;

        // 1st-order high-shelf (bilinear transform approach)
        let k = (PI * fc / sr).tan();
        let v = 10.0_f32.powf(db_gain / 20.0); // linear gain
        // Boost case
        let norm = 1.0 / (1.0 + k);
        let b0 = (v + k) * norm;
        let b1 = (k - v) * norm;
        let a1 = (k - 1.0) * norm;

        Self { b0, b1, a1 }
    }
}

impl Layer for HighFrequencyEmphasisLayer {
    fn name(&self) -> &'static str {
        "HighFrequencyEmphasis"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let mut x1 = 0.0_f32;
        let mut y1 = 0.0_f32;

        for s in samples.iter_mut() {
            let x0 = *s;
            let y0 = self.b0 * x0 + self.b1 * x1 - self.a1 * y1;
            x1 = x0;
            y1 = y0;
            *s = y0.clamp(-1.0, 1.0);
        }
    }
}
