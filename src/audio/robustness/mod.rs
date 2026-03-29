//! `audio::robustness` — automated watermark robustness testing.
//!
//! # Usage
//!
//! ```
//! cargo run -- robustness 0xDEADBEEF12345678
//! ```
//!
//! Runs all attacks on `output_watermarked.wav` using `original.wmpf`
//! as the reference fingerprint, then prints a full comparison table.

pub mod attacks;
pub mod report;

pub use report::{AttackResult, print_robustness_report, print_layer_survival};
pub use attacks::{all_attacks, apply_attack};

use crate::audio::extractor::{extract_with_ref, Fingerprint};

/// Run all attacks and return results.
pub fn run_robustness_test(
    watermarked_wav: &str,
    fingerprint_path: &str,
    key: u64,
    sample_rate: u32,
) -> Vec<AttackResult> {
    let fp = match Fingerprint::load(fingerprint_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot load fingerprint: {e}");
            return vec![];
        }
    };

    let attacks = all_attacks();
    let mut results = Vec::with_capacity(attacks.len());

    for attack in &attacks {
        print!(" Running {:.<40}", format!("{} ", attack.name));

        match apply_attack(attack.name, watermarked_wav, sample_rate) {
            None => {
                println!("FAILED (tool error)");
                results.push(AttackResult::failed(attack.name, attack.category));
            }
            Some(samples) => {
                // Trim or pad samples to avoid length mismatch panic
                let extraction = extract_with_ref(&samples, sample_rate, key, &fp);
                println!("{:.1}% ({}/15)",
                    extraction.confidence * 100.0,
                    extraction.detected_count);
                results.push(AttackResult {
                    attack_name: attack.name,
                    category: attack.category,
                    succeeded: true,
                    extraction: Some(extraction),
                });
            }
        }
    }

    results
}
