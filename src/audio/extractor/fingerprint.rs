//! `audio::extractor::fingerprint`
//!
//! At **embed time**: snapshot per-chunk statistics from the ORIGINAL audio
//! and save them to a small `.wmpf` (watermark fingerprint) file.
//!
//! At **verify time**: load the fingerprint, compute the same statistics on
//! the watermarked audio, and compare — the difference is the watermark signal.
//!
//! # File format  (.wmpf — binary, little-endian)
//!
//! ```text
//! [0..8]   magic   "WMPF\x00\x01\x00\x00"
//! [8..16]  key     u64
//! [16..20] sr      u32  sample_rate
//! [20..24] chunks  u32  number of chunks
//! [24..28] chunk_samples u32
//! per chunk (CHUNK_STAT_BYTES each):
//!   rms         f32
//!   dc          f32
//!   band_rms    f32 × 8   (8 frequency bands)
//!   ac_lags     f32 × 16  (autocorrelation at lags 1..=16)
//!   energy_ac   f32 × 32  (energy autocorr at lags 1..=32, for temporal variance)
//! ```

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::f32::consts::PI;

const MAGIC: &[u8; 8] = b"WMPF\x00\x01\x00\x00";
const N_BANDS: usize = 8;
const N_AC_LAGS: usize = 16;
const N_ENERGY_LAGS: usize = 32;

/// Per-chunk statistics captured from the original audio.
#[derive(Debug, Clone)]
pub struct ChunkFingerprint {
    pub rms: f32,
    pub dc: f32,
    /// Energy in 8 frequency bands covering 20 Hz – Nyquist.
    pub band_rms: [f32; N_BANDS],
    /// Normalised autocorrelation at lags 1..=N_AC_LAGS.
    pub ac_lags: [f32; N_AC_LAGS],
    /// Normalised energy autocorrelation at lags 1..=N_ENERGY_LAGS.
    pub energy_ac: [f32; N_ENERGY_LAGS],
    /// Full original PCM samples — used for chain simulation during extraction.
    /// The extractor simulates layers on THIS (not the watermarked audio) so
    /// that the chain state produced by the correct key is deterministic and
    /// wrong keys diverge immediately at layer 0.
    pub orig_samples: Vec<f32>,
}

/// Full fingerprint file — one [`ChunkFingerprint`] per audio chunk.
pub struct Fingerprint {
    pub key: u64,
    pub sample_rate: u32,
    pub chunk_samples: u32,
    pub chunks: Vec<ChunkFingerprint>,
}

impl Fingerprint {
    /// Capture fingerprint from original (un-watermarked) samples.
    pub fn capture(samples: &[f32], sample_rate: u32, key: u64, chunk_samples: usize) -> Self {
        let sr = sample_rate as f32;
        let band_edges = band_edges(sr);

        let mut chunks = Vec::new();
        let mut offset = 0;
        loop {
            if offset >= samples.len() { break; }
            let end = (offset + chunk_samples).min(samples.len());
            let chunk = &samples[offset..end];
            offset = end;
            chunks.push(compute_chunk_fp(chunk, sr, &band_edges));
        }

        Self { key, sample_rate, chunk_samples: chunk_samples as u32, chunks }
    }

    /// Save to a `.wmpf` file.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);

        w.write_all(MAGIC)?;
        w.write_all(&self.key.to_le_bytes())?;
        w.write_all(&self.sample_rate.to_le_bytes())?;
        w.write_all(&(self.chunks.len() as u32).to_le_bytes())?;
        w.write_all(&self.chunk_samples.to_le_bytes())?;

        for c in &self.chunks {
            w.write_all(&c.rms.to_le_bytes())?;
            w.write_all(&c.dc.to_le_bytes())?;
            for &v in &c.band_rms  { w.write_all(&v.to_le_bytes())?; }
            for &v in &c.ac_lags   { w.write_all(&v.to_le_bytes())?; }
            for &v in &c.energy_ac { w.write_all(&v.to_le_bytes())?; }
            w.write_all(&(c.orig_samples.len() as u32).to_le_bytes())?;
            for &v in &c.orig_samples { w.write_all(&v.to_le_bytes())?; }
        }
        w.flush()?;
        Ok(())
    }

    /// Load from a `.wmpf` file.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData, "Not a valid .wmpf fingerprint file"));
        }

        let key          = read_u64(&mut r)?;
        let sample_rate  = read_u32(&mut r)?;
        let n_chunks     = read_u32(&mut r)? as usize;
        let chunk_samples = read_u32(&mut r)?;

        let mut chunks = Vec::with_capacity(n_chunks);
        for _ in 0..n_chunks {
            let rms = read_f32(&mut r)?;
            let dc  = read_f32(&mut r)?;
            let mut band_rms  = [0f32; N_BANDS];
            let mut ac_lags   = [0f32; N_AC_LAGS];
            let mut energy_ac = [0f32; N_ENERGY_LAGS];
            for v in &mut band_rms  { *v = read_f32(&mut r)?; }
            for v in &mut ac_lags   { *v = read_f32(&mut r)?; }
            for v in &mut energy_ac { *v = read_f32(&mut r)?; }
            let n_orig = read_u32(&mut r)? as usize;
            let mut orig_samples = vec![0f32; n_orig];
            for v in &mut orig_samples { *v = read_f32(&mut r)?; }
            chunks.push(ChunkFingerprint { rms, dc, band_rms, ac_lags, energy_ac, orig_samples });
        }

        Ok(Self { key, sample_rate, chunk_samples, chunks })
    }
}

// ─── Computation ─────────────────────────────────────────────────────────────

fn compute_chunk_fp(chunk: &[f32], sr: f32, band_edges: &[f32]) -> ChunkFingerprint {
    let rms = compute_rms(chunk);
    let dc  = chunk.iter().sum::<f32>() / chunk.len().max(1) as f32;

    // Band RMS
    let mut band_rms = [0f32; N_BANDS];
    for i in 0..N_BANDS {
        band_rms[i] = band_rms_iir(chunk, sr, band_edges[i], band_edges[i + 1]);
    }

    // Normalised autocorrelation
    let ac0 = autocorr_raw(chunk, 0);
    let mut ac_lags = [0f32; N_AC_LAGS];
    for (i, v) in ac_lags.iter_mut().enumerate() {
        *v = if ac0 > 1e-12 { autocorr_raw(chunk, i + 1) / ac0 } else { 0.0 };
    }

    // Energy autocorrelation
    let energy: Vec<f32> = chunk.iter().map(|s| s * s).collect();
    let eac0 = autocorr_raw(&energy, 0);
    let mut energy_ac = [0f32; N_ENERGY_LAGS];
    for (i, v) in energy_ac.iter_mut().enumerate() {
        *v = if eac0 > 1e-12 { autocorr_raw(&energy, i + 1) / eac0 } else { 0.0 };
    }

    ChunkFingerprint { rms, dc, band_rms, ac_lags, energy_ac, orig_samples: chunk.to_vec() }
}

/// 8 log-spaced band edges from 20 Hz to Nyquist.
pub fn band_edges(sample_rate: f32) -> Vec<f32> {
    let nyquist = sample_rate / 2.0;
    let lo = 20.0_f32.ln();
    let hi = nyquist.ln();
    (0..=N_BANDS)
        .map(|i| (lo + (hi - lo) * i as f32 / N_BANDS as f32).exp())
        .collect()
}

pub fn band_rms_iir(samples: &[f32], sr: f32, f_lo: f32, f_hi: f32) -> f32 {
    if samples.is_empty() { return 0.0; }
    let f_lo = f_lo.clamp(1.0, sr / 2.0 - 1.0);
    let f_hi = f_hi.clamp(f_lo + 1.0, sr / 2.0 - 0.5);
    let a_hi = 1.0 - (-2.0 * PI * f_hi / sr).exp();
    let a_lo = 1.0 - (-2.0 * PI * f_lo / sr).exp();
    let mut lp_hi = 0.0f32;
    let mut lp_lo = 0.0f32;
    let mut sum_sq = 0.0f32;
    for &s in samples {
        lp_hi += a_hi * (s - lp_hi);
        lp_lo += a_lo * (s - lp_lo);
        sum_sq += (lp_hi - lp_lo).powi(2);
    }
    (sum_sq / samples.len() as f32).sqrt()
}

fn autocorr_raw(samples: &[f32], lag: usize) -> f32 {
    if lag >= samples.len() { return 0.0; }
    let n = samples.len() - lag;
    samples[..n].iter().zip(&samples[lag..]).map(|(a, b)| a * b).sum::<f32>() / n as f32
}

fn compute_rms(s: &[f32]) -> f32 {
    if s.is_empty() { return 0.0; }
    (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt()
}

// ─── Binary helpers ───────────────────────────────────────────────────────────

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_f32(r: &mut impl Read) -> std::io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
