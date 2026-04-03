//! Chained layer parameter derivation.
//!
//! Instead of each layer drawing its key_byte independently from the master key,
//! each layer's parameters depend on:
//!   1. The master key K
//!   2. The layer's position in the permutation
//!   3. The chunk index (makes each chunk's chain unique)
//!   4. The audio state AFTER the previous layer ran
//!      (RMS + spectral centroid of the processed samples)
//!
//! This creates a cryptographic chain: to reproduce layer N's parameters,
//! you must know the exact audio state produced by layers 0..N-1,
//! which requires the key. A wrong key diverges at layer 0 and every
//! subsequent layer gets increasingly wrong parameters.
//!
//! # Chain state
//!
//! The chain state is a single u64 that evolves as:
//!
//! ```
//! state[0]   = splitmix64(key XOR chunk_index)
//! state[s+1] = splitmix64(state[s] XOR audio_hash(samples_after_layer_s))
//! ```
//!
//! # Audio hash
//!
//! audio_hash combines RMS and spectral centroid into a single u64:
//!
//! ```
//! audio_hash = bits(rms) XOR (bits(centroid) << 32) XOR bits(centroid)
//! ```
//!
//! Both are computed quickly (O(N)) with no allocations.

use std::f32::consts::PI;

/// Evolving chain state passed between layers.
#[derive(Clone, Copy, Debug)]
pub struct ChainState(pub u64);

impl ChainState {
    /// Initialise chain for a specific chunk.
    /// Every chunk gets a unique starting state even with the same key.
    pub fn new(key: u64, chunk_index: usize) -> Self {
        let mixed = key ^ splitmix64(chunk_index as u64 ^ 0xc0ffee_deadbeef);
        Self(splitmix64(mixed))
    }

    /// Derive an 8-bit parameter for the current layer slot.
    /// The `slot` is the position in the permuted order (0..14),
    /// not the layer index — so the same layer in a different position
    /// gets different parameters.
    pub fn derive_byte(&self, slot: usize) -> u8 {
        let h = splitmix64(self.0 ^ (slot as u64).wrapping_mul(0x9e3779b97f4a7c15));
        (h & 0xFF) as u8
    }

    /// Derive a full u64 sub-key for the current layer slot.
    pub fn derive_u64(&self, slot: usize) -> u64 {
        splitmix64(self.0 ^ (slot as u64).wrapping_mul(0x517cc1b727220a95))
    }

    /// Advance the chain using the audio state after a layer ran.
    ///
    /// Computes RMS and spectral centroid of `samples` and folds them
    /// into the current state to produce the next state.
    pub fn advance(&self, samples: &[f32], sample_rate: u32) -> Self {
        let hash = audio_hash(samples, sample_rate);
        Self(splitmix64(self.0 ^ hash))
    }
}

// ─── Audio hash ──────────────────────────────────────────────────────────────

/// Compute a u64 hash from the audio state (RMS + spectral centroid).
/// Both metrics change detectably after every layer runs.
pub fn audio_hash(samples: &[f32], sample_rate: u32) -> u64 {
    let rms       = compute_rms(samples);
    let centroid  = spectral_centroid(samples, sample_rate);

    // Pack both floats into a u64 and mix thoroughly
    let rms_bits      = rms.to_bits() as u64;
    let centroid_bits = centroid.to_bits() as u64;

    // XOR with rotation to avoid cancellation when rms ≈ centroid
    let combined = rms_bits
        ^ centroid_bits.rotate_left(32)
        ^ centroid_bits.rotate_right(17);

    splitmix64(combined)
}

/// RMS of a sample slice.
fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() { return 0.0; }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

/// Spectral centroid via first-moment approximation using a running
/// difference-of-lowpass filterbank (O(N), no FFT, no alloc).
///
/// Centroid = sum(freq_i * energy_i) / sum(energy_i)
/// Approximated with 8 log-spaced IIR bands.
fn spectral_centroid(samples: &[f32], sample_rate: u32) -> f32 {
    if samples.is_empty() { return 0.0; }
    let sr = sample_rate as f32;
    let nyquist = sr / 2.0;

    // 8 log-spaced band edges from 50 Hz to Nyquist
    let n_bands = 8usize;
    let lo = 50.0_f32.ln();
    let hi = nyquist.ln();
    let edges: Vec<f32> = (0..=n_bands)
        .map(|i| (lo + (hi - lo) * i as f32 / n_bands as f32).exp())
        .collect();

    // Pre-compute LP coefficients
    let coeffs: Vec<f32> = edges.iter()
        .map(|&fc| 1.0 - (-2.0 * PI * fc / sr).exp())
        .collect();

    let mut lp = vec![0.0_f32; n_bands + 1];
    let mut band_energy = vec![0.0_f32; n_bands];

    for &s in samples {
        for k in 0..=n_bands {
            lp[k] += coeffs[k] * (s - lp[k]);
        }
        for k in 0..n_bands {
            let bp = lp[k + 1] - lp[k];
            band_energy[k] += bp * bp;
        }
    }

    let total_energy: f32 = band_energy.iter().sum();
    if total_energy < 1e-12 { return 0.0; }

    // Band centre frequencies
    let centres: Vec<f32> = (0..n_bands)
        .map(|k| (edges[k] * edges[k + 1]).sqrt())
        .collect();

    let weighted: f32 = band_energy.iter()
        .zip(centres.iter())
        .map(|(&e, &f)| e * f)
        .sum();

    weighted / total_energy
}

// ─── splitmix64 ──────────────────────────────────────────────────────────────

#[inline]
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_is_key_sensitive() {
        let s1 = ChainState::new(0xDEADBEEF12345678, 0);
        let s2 = ChainState::new(0xDEADBEEF12345679, 0); // one bit different
        assert_ne!(s1.0, s2.0);
    }

    #[test]
    fn chain_is_chunk_sensitive() {
        let s1 = ChainState::new(0xDEADBEEF12345678, 0);
        let s2 = ChainState::new(0xDEADBEEF12345678, 1);
        assert_ne!(s1.0, s2.0);
    }

    #[test]
    fn advance_changes_state() {
        let s = ChainState::new(42, 0);
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin()).collect();
        let s2 = s.advance(&samples, 44100);
        assert_ne!(s.0, s2.0);
    }

    #[test]
    fn different_audio_different_state() {
        let s = ChainState::new(42, 0);
        let a: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin()).collect();
        let b: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.002).sin()).collect();
        assert_ne!(s.advance(&a, 44100).0, s.advance(&b, 44100).0);
    }
}
