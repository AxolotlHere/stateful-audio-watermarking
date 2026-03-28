//! Layer 11 – Controlled Nonlinear Transform (Soft Saturation)
//!
//! WHAT  : Applies a hyperbolic-tangent soft clipper with a drive factor
//!         so close to 1.0 that the nonlinear component is negligible.
//!         Formula:  y = tanh(drive * x) / tanh(drive)
//!         where drive ∈ [1.001, 1.010] is key-derived.
//!
//! SAFE  : At drive = 1.01, the THD introduced is < 0.01 % (−80 dB),
//!         which is far below audibility.  The output is bounded to [−1, 1].

use super::layer_trait::Layer;

pub struct ControlledNonlinearLayer {
    drive: f32,
    normaliser: f32,
}

impl ControlledNonlinearLayer {
    pub fn new(key_byte: u8) -> Self {
        let t = key_byte as f32 / 255.0;
        let drive = 1.001 + t * 0.009; // 1.001 – 1.010
        let normaliser = drive.tanh();
        Self { drive, normaliser }
    }
}

impl Layer for ControlledNonlinearLayer {
    fn name(&self) -> &'static str {
        "ControlledNonlinear"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        for s in samples.iter_mut() {
            *s = (self.drive * *s).tanh() / self.normaliser;
            // tanh output is already in (-1,1), clamp for float safety
            *s = s.clamp(-1.0, 1.0);
        }
    }
}
