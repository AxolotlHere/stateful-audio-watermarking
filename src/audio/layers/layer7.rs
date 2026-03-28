//! Layer 7 – Phase Perturbation (1st-Order All-Pass)
//!
//! WHAT  : Inserts a 1st-order all-pass filter with a key-derived pole
//!         frequency.  All-pass filters preserve magnitude; only phase is
//!         shifted (0° to −180° across the audio band).
//!
//! SAFE  : Magnitude response is identically 1.0 at all frequencies —
//!         zero amplitude distortion whatsoever.  Phase shift is
//!         perceptually inaudible for broadband content.

use super::layer_trait::Layer;
use std::f32::consts::PI;

pub struct PhasePerturbationLayer {
    coeff: f32, // all-pass coefficient
}

impl PhasePerturbationLayer {
    pub fn new(key_byte: u8, sample_rate: u32) -> Self {
        let t = key_byte as f32 / 255.0;
        // Pole frequency: 500 Hz – 5 kHz
        let fp = 500.0 + t * 4500.0;
        let sr = sample_rate as f32;
        // 1st-order all-pass: H(z) = (z^-1 - c) / (1 - c*z^-1)
        // c = (tan(π·fp/sr) - 1) / (tan(π·fp/sr) + 1)
        let tan_val = (PI * fp / sr).tan();
        let coeff = (tan_val - 1.0) / (tan_val + 1.0);
        Self { coeff }
    }
}

impl Layer for PhasePerturbationLayer {
    fn name(&self) -> &'static str {
        "PhasePerturbation"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let c = self.coeff;
        let mut x1 = 0.0_f32;
        let mut y1 = 0.0_f32;

        for s in samples.iter_mut() {
            let x0 = *s;
            // y[n] = c*(x[n] - y[n-1]) + x[n-1]
            let y0 = c * (x0 - y1) + x1;
            x1 = x0;
            y1 = y0;
            // All-pass preserves |H|=1, so clamp is just safety
            *s = y0.clamp(-1.0, 1.0);
        }
    }
}
