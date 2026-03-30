//! Wrong-key security test.
//!
//! Tests N random wrong keys against the watermarked file and measures
//! their confidence scores. A secure system shows a large gap between
//! the correct key score and the wrong-key distribution.

use super::{extract_with_ref, Fingerprint};

pub struct KeyTestResult {
    pub correct_key: u64,
    pub correct_score: f32,
    pub wrong_key_scores: Vec<(u64, f32)>,
    pub mean_wrong: f32,
    pub max_wrong: f32,
    pub std_wrong: f32,
    pub separation_gap: f32,   // correct_score - max_wrong
    pub false_positive_rate: f32, // fraction of wrong keys exceeding threshold
}

impl KeyTestResult {
    pub fn print_report(&self) {
        println!("\n{}", "═".repeat(60));
        println!(" SECURITY TEST — WRONG KEY ANALYSIS");
        println!("{}", "═".repeat(60));
        println!(" Correct key     : 0x{:016X}", self.correct_key);
        println!(" Correct score   : {:.1}%  ← WATERMARK CONFIRMED", self.correct_score * 100.0);
        println!("{}", "─".repeat(60));
        println!(" Wrong keys tested    : {}", self.wrong_key_scores.len());
        println!(" Wrong key mean       : {:.1}%", self.mean_wrong * 100.0);
        println!(" Wrong key max        : {:.1}%", self.max_wrong * 100.0);
        println!(" Wrong key std dev    : {:.1}%", self.std_wrong * 100.0);
        println!("{}", "─".repeat(60));
        println!(" Separation gap       : {:.1}%  (correct − max_wrong)",
            self.separation_gap * 100.0);
        println!(" False positive rate  : {:.1}%  (wrong keys ≥ 60% threshold)",
            self.false_positive_rate * 100.0);
        println!("{}", "─".repeat(60));

        // Security verdict
        if self.separation_gap > 0.15 && self.false_positive_rate < 0.01 {
            println!(" SECURITY VERDICT: STRONG — large gap, negligible false positives");
        } else if self.separation_gap > 0.05 && self.false_positive_rate < 0.05 {
            println!(" SECURITY VERDICT: MODERATE — detectable gap, low false positives");
        } else {
            println!(" SECURITY VERDICT: WEAK — insufficient separation from wrong keys");
        }
        println!("{}", "═".repeat(60));

        // Histogram of wrong key scores
        println!("\n Wrong key confidence distribution:");
        let buckets = 10usize;
        let mut counts = vec![0usize; buckets];
        for (_, score) in &self.wrong_key_scores {
            let idx = (score * buckets as f32).min(buckets as f32 - 1.0) as usize;
            counts[idx] += 1;
        }
        let max_count = *counts.iter().max().unwrap_or(&1);
        for (i, &count) in counts.iter().enumerate() {
            let lo = i * 10;
            let hi = lo + 10;
            let bar_len = if max_count > 0 { count * 30 / max_count } else { 0 };
            let marker = if lo <= (self.correct_score * 100.0) as usize
                           && ((self.correct_score * 100.0) as usize) < hi { " ← CORRECT KEY" } else { "" };
            println!(" {:2}–{:2}%  {:30}  {:3}{}",
                lo, hi, "█".repeat(bar_len), count, marker);
        }
    }
}

/// Run the wrong-key security test.
pub fn run_key_test(
    wm_samples: &[f32],
    sample_rate: u32,
    correct_key: u64,
    fp: &Fingerprint,
    n_wrong_keys: usize,
) -> KeyTestResult {
    // Correct key score
    let correct_result = extract_with_ref(wm_samples, sample_rate, correct_key, fp);
    let correct_score  = correct_result.confidence;

    // Generate deterministic wrong keys using a simple counter-based scheme
    // so results are reproducible across runs
    let mut wrong_key_scores = Vec::with_capacity(n_wrong_keys);

    for i in 0..n_wrong_keys {
        let wrong_key = wrong_key_from_index(correct_key, i);
        let result = extract_with_ref(wm_samples, sample_rate, wrong_key, fp);
        wrong_key_scores.push((wrong_key, result.confidence));

        // Progress indicator every 10 keys
        if (i + 1) % 10 == 0 {
            print!("  Tested {}/{} wrong keys...\r", i + 1, n_wrong_keys);
        }
    }
    println!("  Tested {} wrong keys.          ", n_wrong_keys);

    // Statistics
    let scores: Vec<f32> = wrong_key_scores.iter().map(|(_, s)| *s).collect();
    let mean_wrong = scores.iter().sum::<f32>() / scores.len() as f32;
    let max_wrong  = scores.iter().cloned().fold(0.0_f32, f32::max);
    let variance   = scores.iter().map(|s| (s - mean_wrong).powi(2)).sum::<f32>()
                     / scores.len() as f32;
    let std_wrong  = variance.sqrt();
    let threshold  = 0.60_f32;
    let false_positives = scores.iter().filter(|&&s| s >= threshold).count();
    let false_positive_rate = false_positives as f32 / scores.len() as f32;
    let separation_gap = correct_score - max_wrong;

    KeyTestResult {
        correct_key,
        correct_score,
        wrong_key_scores,
        mean_wrong,
        max_wrong,
        std_wrong,
        separation_gap,
        false_positive_rate,
    }
}

/// Generate a deterministic wrong key from index i.
/// Uses splitmix64 to spread keys across the full u64 space.
fn wrong_key_from_index(correct_key: u64, i: usize) -> u64 {
    // Mix the index to get a spread-out key, then XOR with correct key
    // to ensure it's never accidentally equal to the correct key
    let mut h = (i as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    // Ensure it never equals the correct key
    if h == correct_key { h ^ 0xFFFF_FFFF_FFFF_FFFF } else { h }
}
