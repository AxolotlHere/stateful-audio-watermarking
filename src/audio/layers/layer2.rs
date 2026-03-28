//! Layer 2 – Micro Time Shift
//!
//! WHAT  : Shifts the entire chunk forward by N samples (1–8 samples)
//!         by prepending zeros and dropping the tail.  The shift amount N
//!         is key-derived and constant within a chunk.
//!
//! SAFE  : 1–8 samples at 44.1 kHz = 0.02–0.18 ms; completely inaudible.
//!         Clipping cannot occur because we insert silence, not signal.

use super::layer_trait::Layer;

pub struct MicroTimeShiftLayer {
    shift_samples: usize,
}

impl MicroTimeShiftLayer {
    /// `key_byte` – 0..=255; mapped to 1..=8 samples.
    pub fn new(key_byte: u8) -> Self {
        let shift_samples = (key_byte as usize % 8) + 1;
        Self { shift_samples }
    }
}

impl Layer for MicroTimeShiftLayer {
    fn name(&self) -> &'static str {
        "MicroTimeShift"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let n = self.shift_samples.min(samples.len());
        //Rotate right by n
        //overwrite tail positions with 0 for cleanliness.
        samples.rotate_right(n);
        for s in samples[..n].iter_mut() {
            *s = 0.0;
        }
    }
}
