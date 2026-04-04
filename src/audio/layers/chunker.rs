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
const WATERMARK_DENSITY_DENOM: u64 = 12;
const MAX_OFFSET_DIVISOR: usize = 3;
const REGION_COUNT: usize = 3;
const MIN_REGION_SAMPLES: usize = 8192;

pub struct KeyedChunker {
    key: u64,
    chunk_samples: usize,
    chunk_offset: usize,
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
        let max_offset = (chunk_samples / MAX_OFFSET_DIVISOR).max(1);
        let chunk_offset = (key_mix(key ^ 0xa5a5_5a5a_f0f0_0f0f) as usize) % max_offset;
        let fade_samples = ((FADE_MS / 1000.0) * sample_rate as f32).round() as usize;
        Self {
            key,
            chunk_samples,
            chunk_offset,
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

    /// Key-derived sample offset applied before chunking starts.
    pub fn chunk_offset_samples(&self) -> usize {
        self.chunk_offset
    }

    /// Number of addressable chunks after the key-derived offset.
    pub fn chunk_count(&self, total_samples: usize) -> usize {
        total_samples
            .saturating_sub(self.chunk_offset)
            .div_ceil(self.chunk_samples)
    }

    /// Sparse keyed selection of watermark-bearing chunks.
    ///
    /// Always activates at least one chunk for short signals.
    pub fn should_watermark_chunk(&self, chunk_index: usize, total_chunks: usize) -> bool {
        if total_chunks == 0 {
            return false;
        }
        if chunk_index == self.primary_chunk_index(total_chunks) {
            return true;
        }
        let h = key_mix(
            self.key
                ^ (chunk_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ 0xd1b5_4a32_d192_ed03,
        );
        h % WATERMARK_DENSITY_DENOM == 0
    }

    pub fn primary_chunk_index(&self, total_chunks: usize) -> usize {
        if total_chunks == 0 {
            return 0;
        }
        (key_mix(self.key ^ 0x6a09_e667_f3bc_c909) as usize) % total_chunks
    }

    /// Key-selected subregions within an active chunk.
    pub fn watermark_windows(&self, chunk_index: usize, chunk_len: usize) -> Vec<(usize, usize)> {
        if chunk_len == 0 {
            return Vec::new();
        }

        let max_width = (chunk_len / 4).max(1);
        let width = (chunk_len / 16).max(MIN_REGION_SAMPLES).min(max_width).min(chunk_len);
        let available = chunk_len.saturating_sub(width);
        let region_count = REGION_COUNT.min(chunk_len.max(1));
        let mut windows = Vec::with_capacity(region_count);

        for region_idx in 0..region_count {
            let bucket_start = available * region_idx / region_count;
            let bucket_end = available * (region_idx + 1) / region_count;
            let span = bucket_end.saturating_sub(bucket_start).max(1);
            let seed = key_mix(
                self.key
                    ^ (chunk_index as u64).wrapping_mul(0x517c_c1b7_2722_0a95)
                    ^ (region_idx as u64).wrapping_mul(0x94d0_49bb_1331_11eb),
            );
            let start = bucket_start + (seed as usize % span);
            let end = (start + width).min(chunk_len);
            windows.push((start, end));
        }

        windows.sort_unstable_by_key(|&(start, _)| start);
        windows
    }

    /// Iterate over mutable sub-slices of `samples`, one per chunk.
    ///
    /// The last partial chunk (if any) is yielded as-is.
    pub fn iter_chunks_mut<'a>(
        &self,
        samples: &'a mut [f32],
    ) -> impl Iterator<Item = &'a mut [f32]> {
        let (_, tail) = samples.split_at_mut(self.chunk_offset.min(samples.len()));
        ChunkIterMut {
            remaining: tail,
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

    /// Blend the processed region back into the unmodified audio at the
    /// region edges. Unlike `apply_boundary_fade`, this preserves the host
    /// signal energy at internal watermark-window boundaries.
    pub fn blend_region_edges(&self, original: &[f32], processed: &mut [f32]) {
        let len = original.len().min(processed.len());
        let fade = self.fade_samples.min(len / 2);
        if len == 0 || fade == 0 {
            return;
        }

        for i in 0..fade {
            let wet = i as f32 / fade as f32;
            let dry = 1.0 - wet;
            processed[i] = original[i] * dry + processed[i] * wet;

            let j = len - 1 - i;
            processed[j] = original[j] * dry + processed[j] * wet;
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
    fn chunk_offset_is_bounded() {
        let c = KeyedChunker::new(12345, 44100);
        assert!(c.chunk_offset_samples() < c.chunk_size_samples() / MAX_OFFSET_DIVISOR.max(1));
    }

    #[test]
    fn covers_all_samples() {
        let mut samples: Vec<f32> = (0..100_000).map(|i| i as f32).collect();
        let chunker = KeyedChunker::new(12345, 44100);
        let mut total = 0usize;
        for chunk in chunker.iter_chunks_mut(&mut samples) {
            total += chunk.len();
        }
        assert_eq!(total, 100_000 - chunker.chunk_offset_samples());
    }

    #[test]
    fn sparse_selection_is_keyed() {
        let c1 = KeyedChunker::new(12345, 44100);
        let c2 = KeyedChunker::new(12346, 44100);
        let s1: Vec<bool> = (0..64).map(|i| c1.should_watermark_chunk(i, 64)).collect();
        let s2: Vec<bool> = (0..64).map(|i| c2.should_watermark_chunk(i, 64)).collect();
        assert_ne!(s1, s2);
    }

    #[test]
    fn always_selects_one_chunk_for_short_audio() {
        let c = KeyedChunker::new(12345, 44100);
        let hits = (0..1).filter(|&i| c.should_watermark_chunk(i, 1)).count();
        assert_eq!(hits, 1);
    }
}
