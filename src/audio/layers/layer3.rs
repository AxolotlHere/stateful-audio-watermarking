//! Layer 3 – Envelope Shaping (Mild Sinusoidal Modulation)
//!
//! WHAT  : Multiplies the chunk by a very slow sinusoidal envelope whose
//!         frequency is key-derived (0.05–0.5 Hz) and whose depth is ≤ 0.4 %.
//!
//! SAFE  : Depth of 0.004 (0.4 %) is three orders of magnitude below the
//!         loudness JND; the modulation frequency is sub-tremolo.


use super::layer_trait::Layer;
use std::f32::consts::PI;

pub struct EnvelopeShapingLayer {
    /// Modulation frequency in Hz (0.05–0.5).
    mod_freq: f32,
    /// Modulation depth (0.0–0.004).
    depth: f32,
    /// Phase offset in radians.
    phase: f32,
}

impl EnvelopeShapingLayer {
    pub fn new(key_byte: u8) -> Self {
        let t = key_byte as f32 / 255.0;
        let mod_freq = 0.05 + t * 0.45; // 0.05..0.5 Hz
        let depth = 0.001 + t * 0.003; // 0.001..0.004
        let phase = t * 2.0 * PI;
        Self {
            mod_freq,
            depth,
            phase,
        }
    }
}

impl Layer for EnvelopeShapingLayer {
    fn name(&self) -> &'static str {
        "EnvelopeShaping"
    }

    fn apply(&self, samples: &mut [f32], sample_rate: u32) {
        let sr = sample_rate as f32;
        for (i, s) in samples.iter_mut().enumerate() {
            let t = i as f32 / sr;
            let env = 1.0 + self.depth * (2.0 * PI * self.mod_freq * t + self.phase).sin();
            *s = (*s * env).clamp(-1.0, 1.0);
        }
    }
}
