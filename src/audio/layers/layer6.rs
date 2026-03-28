//! Layer 6 – Narrowband Attenuation (Biquad Notch)
//!
//! WHAT  : Applies a very shallow notch (−0.08 dB) at a key-derived
//!         frequency with a narrow bandwidth (Q = 8).
//!
//! SAFE  : −0.08 dB notch is inaudible; high Q keeps the affected
//!         bandwidth under 200 Hz at 4 kHz centre.

use super::layer_trait::Layer;
use std::f32::consts::PI;

pub struct NarrowbandAttenuationLayer {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl NarrowbandAttenuationLayer {
    pub fn new(key_byte: u8, sample_rate: u32) -> Self {
        let t = key_byte as f32 / 255.0;
        let fc = 1000.0 + t * 7000.0; // 1–8 kHz
        let db_gain = -0.08_f32;
        let q = 8.0_f32;
        let sr = sample_rate as f32;

        let a_coef = 10.0_f32.powf(db_gain / 40.0);
        let w0 = 2.0 * PI * fc / sr;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let b0 = 1.0 + alpha * a_coef;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a_coef;
        let a0 = 1.0 + alpha / a_coef;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a_coef;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }
}

impl Layer for NarrowbandAttenuationLayer {
    fn name(&self) -> &'static str {
        "NarrowbandAttenuation"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let (mut x1, mut x2) = (0.0_f32, 0.0_f32);
        let (mut y1, mut y2) = (0.0_f32, 0.0_f32);

        for s in samples.iter_mut() {
            let x0 = *s;
            let y0 = self.b0 * x0 + self.b1 * x1 + self.b2 * x2
                - self.a1 * y1
                - self.a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            *s = y0.clamp(-1.0, 1.0);
        }
    }
}
