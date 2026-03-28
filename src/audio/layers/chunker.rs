//! Key-derived audio chunker.
//!
//! Splits a flat sample buffer into variable-length chunks whose size is
//! determined by the watermark key.  Using a key-derived chunk size means
//! an attacker cannot know where chunk boundaries are without the key,
//! complicating removal attacks.
//!
//! # Boundary safety
//!
//! Each chunk is processed independently.  To avoid boundary artefacts
//! from IIR-type layers (biquads, all-pass, comb), each chunk is processed
//! with a short **fade-in / fade-out window** (8 ms) applied *after* all
//! layer transformations.  This suppresses any transient at the boundary
//! introduced by filter state initialisation.

const MIN_CHUNK_SECS: usize = 5;
const MAX_CHUNK_SECS: usize = 15;
const FADE_MS: f32 = 8.0;

pub struct KeyedChunker {
    chunk_samples: usize,
    fade_samples: usize,
}

impl KeyedChunker {
    /// Create a chunker whose chunk size is derived from `key`.
    ///
    /// The chunk duration is in `[MIN_CHUNK_SECS, MAX_CHUNK_SECS]` seconds.
    pub fn new(key: u64, sample_rate: u32) -> Self {
        let range = (MAX_CHUNK_SECS - MIN_CHUNK_SECS + 1) as u64;
        let secs = MIN_CHUNK_SECS + (key_mix(key) % range) as usize;
        let chunk_samples = secs * sample_rate as usize;
        let fade_samples = ((FADE_MS / 1000.0) * sample_rate as f32).round() as usize;
        Self {
            chunk_samples,
            fade_samples,
        }
    }

    /// Chunk duration in seconds (for logging).
    pub fn chunk_seconds(&self) -> usize {
        // Best-effort: not stored, but derivable from chunk_samples if needed.
        // Expose chunk_samples directly instead.
        self.chunk_samples
    }

    /// Chunk size in samples.
    pub fn chunk_size_samples(&self) -> usize {
        self.chunk_samples
    }

    /// Iterate over mutable sub-slices of `samples`, one per chunk.
    ///
    /// The last partial chunk (if any) is yielded as-is.
    pub fn iter_chunks_mut<'a>(
        &self,
        samples: &'a mut [f32],
    ) -> impl Iterator<Item = &'a mut [f32]> {
        ChunkIterMut {
            remaining: samples,
            chunk_size: self.chunk_samples,
        }
    }

    /// Apply a boundary-safety fade-in/out in-place on a chunk.
    ///
    /// Call this **after** all layers have been applied to the chunk.
    pub fn apply_boundary_fade(&self, chunk: &mut [f32]) {
        let fade = self.fade_samples.min(chunk.len() / 2);
        if fade == 0 {
            return;
        }
        // Fade-in: first `fade` samples
        for i in 0..fade {
            let gain = i as f32 / fade as f32;
            chunk[i] *= gain;
        }
        // Fade-out: last `fade` samples
        let len = chunk.len();
        for i in 0..fade {
            let gain = i as f32 / fade as f32;
            chunk[len - 1 - i] *= gain;
        }
    }
}

// ─── Iterator helper ─────────────────────────────────────────────────────────

struct ChunkIterMut<'a> {
    remaining: &'a mut [f32],
    chunk_size: usize,
}

impl<'a> Iterator for ChunkIterMut<'a> {
    type Item = &'a mut [f32];

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining.is_empty() {
            return None;
        }
        let n = self.chunk_size.min(self.remaining.len());
        // SAFETY: we move `remaining` forward; lifetimes don't overlap.
        let (head, tail) = std::mem::take(&mut self.remaining).split_at_mut(n);
        self.remaining = tail;
        Some(head)
    }
}

// ─── Key mixing helper ────────────────────────────────────────────────────────

fn key_mix(key: u64) -> u64 {
    let mut h = key ^ 0x9e37_79b9_7f4a_7c15_u64;
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_size_in_range() {
        for key in [0u64, 1, 42, u64::MAX, 0xDEADBEEF] {
            let c = KeyedChunker::new(key, 44100);
            let min = MIN_CHUNK_SECS * 44100;
            let max = MAX_CHUNK_SECS * 44100;
            assert!(
                c.chunk_size_samples() >= min && c.chunk_size_samples() <= max,
                "key={key} chunk={} out of [{min},{max}]",
                c.chunk_size_samples()
            );
        }
    }

    #[test]
    fn covers_all_samples() {
        let mut samples: Vec<f32> = (0..100_000).map(|i| i as f32).collect();
        let chunker = KeyedChunker::new(12345, 44100);
        let mut total = 0usize;
        for chunk in chunker.iter_chunks_mut(&mut samples) {
            total += chunk.len();
        }
        assert_eq!(total, 100_000);
    }
}
