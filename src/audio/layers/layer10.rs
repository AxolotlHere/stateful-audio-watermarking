//! Layer 10 – Noise Shaping (Pseudo-Random Low-Amplitude Dither)
//!
//! WHAT  : Adds a deterministic pseudo-random sequence scaled to ≤ −80 dBFS
//!         (amplitude ≤ 0.0001).  The sequence is seeded from the watermark
//!         key, making it unique and reproducible.
//!
//! SAFE  : −80 dBFS is below the noise floor of virtually all recordings
//!         and well below human hearing threshold in normal listening.

use super::layer_trait::Layer;

pub struct NoiseShapingLayer {
    seed: u64,
    amplitude: f32,
}

impl NoiseShapingLayer {
    pub fn new(key: u64) -> Self {
        Self {
            seed: key,
            amplitude: 0.0001, // −80 dBFS
        }
    }

    /// Xorshift64 — minimal, deterministic PRNG.
    #[inline]
    fn next_rand(state: &mut u64) -> f32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        // Map u64 to [-1, 1)
        ((x as i64) as f32) / (i64::MAX as f32)
    }
}

impl Layer for NoiseShapingLayer {
    fn name(&self) -> &'static str {
        "NoiseShaping"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let mut state = self.seed;
        for s in samples.iter_mut() {
            let noise = Self::next_rand(&mut state) * self.amplitude;
            *s = (*s + noise).clamp(-1.0, 1.0);
        }
    }
}
