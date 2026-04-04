//! Layer 11 – Masked Micro-Gain Modulation
//!
//! WHAT  : Applies a keyed zero-mean gain perturbation whose strength is
//!         masked by the local sample magnitude. Louder samples get slightly
//!         stronger modulation; near-silence is barely touched.
//!
//! SAFE  : This avoids harmonic generation from nonlinear waveshaping and
//!         keeps the perturbation bounded and signal-adaptive.

use super::layer_trait::Layer;

pub struct ControlledNonlinearLayer {
    seed: u64,
    amplitude: f32,
}

impl ControlledNonlinearLayer {
    pub fn new(key_byte: u8) -> Self {
        let t = key_byte as f32 / 255.0;
        let seed = (key_byte as u64).wrapping_mul(0x517c_c1b7_2722_0a95) | 1;
        let amplitude = 0.0015 + t * 0.0020;
        Self { seed, amplitude }
    }
}

impl Layer for ControlledNonlinearLayer {
    fn name(&self) -> &'static str {
        "ControlledNonlinear"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let mut state = self.seed;
        for s in samples.iter_mut() {
            let seq = xorshift_signed(&mut state);
            let mask = s.abs() / (s.abs() + 0.05);
            let gain = 1.0 + seq * self.amplitude * mask;
            *s = (*s * gain).clamp(-1.0, 1.0);
        }
    }
}

#[inline]
fn xorshift_signed(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
}
