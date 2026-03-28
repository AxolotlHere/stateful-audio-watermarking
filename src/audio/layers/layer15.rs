//! Layer 15 – Temporal Variance Modulation (Sample-Wise Gain Dithering)
//!
//! WHAT  : Applies a deterministic per-sample gain drawn from a quantised
//!         triangular distribution centred on 1.0, with peak deviation
//!         ≤ 0.0008.  The pattern repeats every `period` samples (8–32),
//!         where `period` is key-derived.
//!
//! SAFE  : 0.0008 amplitude deviation is −62 dBFS below full scale.
//!         Triangular (TPDF) distribution is the gold standard for dither
//!         in professional audio mastering — no audible artifact.

use super::layer_trait::Layer;

pub struct TemporalVarianceLayer {
    /// The gain table: one entry per position in the repeating period.
    gain_table: Vec<f32>,
}

impl TemporalVarianceLayer {
    pub fn new(key_byte: u8) -> Self {
        // Period: 8–32 samples (key-derived)
        let period = 8 + (key_byte as usize % 25);
        let amplitude = 0.0008_f32;

        // Build deterministic TPDF-like table via xorshift seeded from key_byte
        let mut state: u64 = (key_byte as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let gain_table: Vec<f32> = (0..period)
            .map(|_| {
                // Two uniform samples → triangular distribution
                let u1 = xorshift(&mut state);
                let u2 = xorshift(&mut state);
                1.0 + (u1 - u2) * amplitude
            })
            .collect();

        Self { gain_table }
    }
}

#[inline]
fn xorshift(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    // Map to [0, 1)
    (x >> 11) as f32 / (1u64 << 53) as f32
}

impl Layer for TemporalVarianceLayer {
    fn name(&self) -> &'static str {
        "TemporalVariance"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let period = self.gain_table.len();
        for (i, s) in samples.iter_mut().enumerate() {
            let gain = self.gain_table[i % period];
            *s = (*s * gain).clamp(-1.0, 1.0);
        }
    }
}
