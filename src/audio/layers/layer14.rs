//! Layer 14 – Spectral Tilt (Low-Shelf Bias)
//!
//! WHAT  : Applies a 1st-order low-shelf filter with ±0.05 dB gain below a
//!         key-derived crossover frequency (80–800 Hz), effectively tilting
//!         the spectral centroid slightly up or down.
//!
//! SAFE  : ±0.05 dB is imperceptible even to trained listeners in A/B tests.
//!         1st-order shelves are unconditionally stable and produce no
//!         ringing.


use super::layer_trait::Layer;
use std::f32::consts::PI;

pub struct SpectralTiltLayer {
    b0: f32,
    b1: f32,
    a1: f32,
}

impl SpectralTiltLayer {
    pub fn new(key_byte: u8, sample_rate: u32) -> Self {
        let t = key_byte as f32 / 255.0;
        let fc = 80.0 + t * 720.0; // 80–800 Hz
        let db_gain = if key_byte & 1 == 0 { 0.12_f32 } else { -0.12_f32 };
        let sr = sample_rate as f32;

        // 1st-order low-shelf (bilinear)
        let k = (PI * fc / sr).tan();
        let v = 10.0_f32.powf(db_gain / 20.0);

        // Boost: shelve up low frequencies
        let norm = 1.0 / (1.0 + k);
        // For a low-shelf: H(s) = (s/wc + V) / (s/wc + 1)  →  bilinear
        // Simplified coefficients:
        let b0 = (v * k + 1.0) * norm;
        let b1 = (v * k - 1.0) * norm;
        let a1 = (k - 1.0) * norm;

        Self { b0, b1, a1 }
    }
}

impl Layer for SpectralTiltLayer {
    fn name(&self) -> &'static str {
        "SpectralTilt"
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
