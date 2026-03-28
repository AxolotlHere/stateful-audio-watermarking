//! Layer 13 – Comb Filter Coloring (Feedforward)
//!
//! WHAT  : Applies a single-tap feedforward comb filter:
//!         y[n] = x[n] + g * x[n − D]
//!         where D (delay in samples) and g (gain, ≤ 0.002) are key-derived.
//!
//! SAFE  : With |g| ≤ 0.002, the comb peaks/troughs are ±0.017 dB — far
//!         below the spectral JND.  Feedforward combs are always stable.


use super::layer_trait::Layer;

pub struct CombFilterLayer {
    delay_samples: usize,
    gain: f32,
}

impl CombFilterLayer {
    /// `key_byte` – determines delay and gain sign.
    pub fn new(key_byte: u8, sample_rate: u32) -> Self {
        let t = key_byte as f32 / 255.0;
        // Delay: 10–50 ms expressed in samples
        let delay_ms = 10.0 + t * 40.0;
        let delay_samples = (delay_ms * sample_rate as f32 / 1000.0).round() as usize;
        let delay_samples = delay_samples.max(1);
        // Gain: ±0.002
        let gain = if key_byte & 1 == 0 { 0.002 } else { -0.002 };
        Self {
            delay_samples,
            gain,
        }
    }
}

impl Layer for CombFilterLayer {
    fn name(&self) -> &'static str {
        "CombFilterColoring"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let d = self.delay_samples;
        // We need a delay line; clone the first `d` samples as history.
        let history: Vec<f32> = samples[..d.min(samples.len())].to_vec();

        for i in 0..samples.len() {
            let delayed = if i >= d {
                samples[i - d]
            } else {
                history[i]
            };
            samples[i] = (samples[i] + self.gain * delayed).clamp(-1.0, 1.0);
        }
    }
}
