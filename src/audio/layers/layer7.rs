//! Layer 7 – Sparse Phase Perturbation
//!
//! WHAT  : Applies a very weak first-order all-pass phase shift and blends
//!         only a small portion of the phase-rotated signal back into the
//!         dry path. This keeps the perturbation phase-dominant without
//!         forcing a full-band waveform rewrite.
//!
//! SAFE  : The dry path dominates; the wet mix is only 8–18%, so the layer
//!         acts like a subtle keyed phase nudge instead of a strong reshape.

use super::layer_trait::Layer;
use std::f32::consts::PI;

pub struct PhasePerturbationLayer {
    coeff: f32,
    wet_mix: f32,
}

impl PhasePerturbationLayer {
    pub fn new(key_byte: u8, sample_rate: u32) -> Self {
        let t = key_byte as f32 / 255.0;
        let fp = 700.0 + t * 2900.0;
        let sr = sample_rate as f32;
        let tan_val = (PI * fp / sr).tan();
        let coeff = (tan_val - 1.0) / (tan_val + 1.0);
        let wet_mix = 0.08 + t * 0.10;
        Self { coeff, wet_mix }
    }
}

impl Layer for PhasePerturbationLayer {
    fn name(&self) -> &'static str {
        "PhasePerturbation"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let c = self.coeff;
        let wet = self.wet_mix;
        let dry = 1.0 - wet;
        let mut x1 = 0.0_f32;
        let mut y1 = 0.0_f32;

        for s in samples.iter_mut() {
            let x0 = *s;
            let ap = c * (x0 - y1) + x1;
            x1 = x0;
            y1 = ap;
            *s = (dry * x0 + wet * ap).clamp(-1.0, 1.0);
        }
    }
}
