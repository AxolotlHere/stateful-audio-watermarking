//! Layer 4 – Band-Limited Gain Tweak (Biquad Peaking EQ)
//!
//! WHAT  : Applies a peaking EQ biquad with ±0.1 dB gain at a
//!         key-derived centre frequency (200 Hz – 4 kHz), Q = 1.0.
//!
//! SAFE  : ±0.1 dB is below the typical 0.5 dB spectral JND.
//!         Biquad is unconditionally stable for |gain| < 6 dB.

use super::layer_trait::Layer;
use std::f32::consts::PI;

pub struct BandLimitedGainLayer {
    // Biquad coefficients (Direct Form I)
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
}

impl BandLimitedGainLayer {
    /// `key_byte` – used to choose centre frequency and sign of gain.
    pub fn new(key_byte: u8, sample_rate: u32) -> Self {
        let t = key_byte as f32 / 255.0;
        // Centre frequency: 200 Hz – 4 kHz
        let fc = 200.0 + t * 3800.0;
        // dB gain: alternates sign based on key parity → ±0.1 dB
        let db_gain = if key_byte & 1 == 0 { 0.1_f32 } else { -0.1_f32 };
        let q = 1.0_f32;
        let sr = sample_rate as f32;

        // Standard Audio EQ Cookbook peaking EQ coefficients
        let a_coef = 10.0_f32.powf(db_gain / 40.0); // sqrt(10^(dBGain/20))
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

impl Layer for BandLimitedGainLayer {
    fn name(&self) -> &'static str {
        "BandLimitedGain"
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
