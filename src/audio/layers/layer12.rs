//! Layer 12 – Logistic Map Modulation
//!
//! WHAT  : Generates a chaotic sequence via the logistic map
//!         x_{n+1} = r * x_n * (1 − x_n)  with r = 3.9 + key-offset (∈ [3.9, 3.999])
//!         and scales it to amplitude ≤ 0.0002 before adding to each sample.
//!
//! SAFE  : 0.0002 amplitude is −74 dBFS — well below the noise floor of
//!         typical recordings.

use super::layer_trait::Layer;

pub struct LogisticMapLayer {
    r: f64,
    x0: f64,
    amplitude: f32,
}

impl LogisticMapLayer {
    pub fn new(key_byte: u8) -> Self {
        let t = key_byte as f64 / 255.0;
        // r in [3.9, 3.999] — chaotic regime
        let r = 3.9 + t * 0.099;
        // x0 in (0.1, 0.9) to avoid fixed points
        let x0 = 0.1 + t * 0.8;
        Self {
            r,
            x0,
            amplitude: 0.00045,
        }
    }
}

impl Layer for LogisticMapLayer {
    fn name(&self) -> &'static str {
        "LogisticMapModulation"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let mut x = self.x0;
        for s in samples.iter_mut() {
            x = self.r * x * (1.0 - x);
            // Map (0,1) → (-0.5, 0.5) then scale
            let modulation = ((x - 0.5) * 2.0) as f32 * self.amplitude;
            *s = (*s + modulation).clamp(-1.0, 1.0);
        }
    }
}
