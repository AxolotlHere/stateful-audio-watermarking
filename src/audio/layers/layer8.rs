//! Layer 8 – Local Sample Reordering (Micro-Block Swap)
//!
//! WHAT  : Divides the chunk into blocks of `block_size` samples
//!         (2–4 samples).  Within each block, the first and last sample are
//!         swapped.  The block size is key-derived.
//!
//! SAFE  : At 44.1 kHz, a 4-sample block spans ~0.09 ms.  Swapping two
//!         samples within such a tiny window is completely inaudible.

use super::layer_trait::Layer;
pub struct LocalSampleReorderingLayer {
    block_size: usize,
}

impl LocalSampleReorderingLayer {
    pub fn new(key_byte: u8) -> Self {
        // block_size: 2, 3, or 4 samples
        let block_size = (key_byte as usize % 3) + 2;
        Self { block_size }
    }
}

impl Layer for LocalSampleReorderingLayer {
    fn name(&self) -> &'static str {
        "LocalSampleReordering"
    }

    fn apply(&self, samples: &mut [f32], _sample_rate: u32) {
        let bs = self.block_size;
        for block in samples.chunks_mut(bs) {
            if block.len() == bs {
                // Swap first and last within the block
                let last = block.len() - 1;
                block.swap(0, last);
            }
        }
    }
}
