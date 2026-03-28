//! Key-driven permutation engine.
//!
//! Uses a single 64-bit key to produce a reproducible, uniform random
//! permutation of the 15 layer indices via the Fisher-Yates shuffle.


pub const NUM_LAYERS: usize = 15;

/// Returns a permutation of `[0, 1, …, NUM_LAYERS-1]` that is uniquely
/// determined by `key`.
///
/// The same key always produces the same order (deterministic).
/// Different keys almost certainly produce different orders.
pub fn permute_layers(key: u64) -> [usize; NUM_LAYERS] {
    let mut indices: [usize; NUM_LAYERS] = core::array::from_fn(|i| i);
    let mut rng = Xorshift64::new(key);

    // Fisher-Yates shuffle
    for i in (1..NUM_LAYERS).rev() {
        let j = rng.next_usize_bounded(i + 1);
        indices.swap(i, j);
    }

    indices
}


struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state (xorshift is undefined for 0)
        let state = if seed == 0 { 0xdeadbeef_cafebabe } else { seed };
        // Warm up with one mix so low-entropy keys spread out
        let mut s = Self { state };
        s.next();
        s
    }

    #[inline]
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a value in `[0, bound)` with low bias (bound ≤ 2^32).
    #[inline]
    fn next_usize_bounded(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_key_same_order() {
        let a = permute_layers(0xABCD_1234_5678_EF01);
        let b = permute_layers(0xABCD_1234_5678_EF01);
        assert_eq!(a, b);
    }

    #[test]
    fn different_keys_different_order() {
        let a = permute_layers(1);
        let b = permute_layers(2);
        assert_ne!(a, b);
    }

    #[test]
    fn permutation_is_complete() {
        let p = permute_layers(42);
        let mut seen = [false; NUM_LAYERS];
        for &i in p.iter() {
            assert!(!seen[i], "duplicate index {i}");
            seen[i] = true;
        }
    }
}
