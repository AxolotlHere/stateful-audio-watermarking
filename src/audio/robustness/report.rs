//! Robustness report types and formatted output.

use crate::audio::extractor::ExtractionResult;

/// Result of a single attack.
pub struct AttackResult {
    pub attack_name: &'static str,
    pub category: &'static str,
    pub succeeded: bool,        // did the attack tool run successfully?
    pub extraction: Option<ExtractionResult>,
}

impl AttackResult {
    pub fn failed(attack_name: &'static str, category: &'static str) -> Self {
        Self { attack_name, category, succeeded: false, extraction: None }
    }

    pub fn confidence(&self) -> Option<f32> {
        self.extraction.as_ref().map(|e| e.confidence)
    }

    pub fn detected_count(&self) -> Option<usize> {
        self.extraction.as_ref().map(|e| e.detected_count)
    }

    pub fn watermark_detected(&self) -> bool {
        self.extraction.as_ref().map(|e| e.watermark_detected).unwrap_or(false)
    }
}

/// Print the full robustness comparison table.
pub fn print_robustness_report(results: &[AttackResult], baseline_confidence: f32) {
    println!("\n{}", "═".repeat(72));
    println!(" ROBUSTNESS REPORT");
    println!("{}", "═".repeat(72));
    println!(" {:<22} {:>10}  {:>8}  {:>8}  {}",
        "Attack", "Confidence", "Layers", "Survive", "Verdict");
    println!(" {}", "─".repeat(68));

    for r in results {
        match &r.extraction {
            None => {
                println!(" {:<22} {:>10}  {:>8}  {:>8}  {}",
                    r.attack_name, "ERROR", "—", "—", "⚠ attack failed");
            }
            Some(e) => {
                let survival = if baseline_confidence > 0.0 {
                    format!("{:.0}%", (e.confidence / baseline_confidence) * 100.0)
                } else {
                    "—".to_string()
                };
                let verdict = if e.watermark_detected {
                    "✓ SURVIVES"
                } else {
                    "✗ DESTROYED"
                };
                println!(" {:<22} {:>9.1}%  {:>5}/15  {:>8}  {}",
                    r.attack_name,
                    e.confidence * 100.0,
                    e.detected_count,
                    survival,
                    verdict,
                );
            }
        }
    }

    println!("{}", "─".repeat(72));

    // Summary stats
    let total    = results.iter().filter(|r| r.extraction.is_some()).count();
    let survived = results.iter().filter(|r| r.watermark_detected()).count();
    let failed_attacks = results.iter().filter(|r| !r.succeeded).count();

    println!(" Attacks run        : {}", total);
    println!(" Watermark survived : {}/{}", survived, total);
    if failed_attacks > 0 {
        println!(" Failed to run      : {} (tool not available)", failed_attacks);
    }

    // Per-category summary
    println!("\n Per-category survival:");
    for cat in &["compress", "noise", "edit", "level", "resample"] {
        let cat_results: Vec<&AttackResult> = results.iter()
            .filter(|r| r.category == *cat && r.extraction.is_some())
            .collect();
        if cat_results.is_empty() { continue; }
        let cat_survived = cat_results.iter().filter(|r| r.watermark_detected()).count();
        let avg_conf: f32 = cat_results.iter()
            .filter_map(|r| r.confidence())
            .sum::<f32>() / cat_results.len() as f32;
        println!("   {:<12} {}/{} survived, avg confidence {:.1}%",
            cat, cat_survived, cat_results.len(), avg_conf * 100.0);
    }

    println!("{}", "═".repeat(72));
}

/// Print per-layer survival across all attacks.
pub fn print_layer_survival(results: &[AttackResult]) {
    println!("\n{}", "═".repeat(72));
    println!(" LAYER SURVIVAL TABLE  (✓ = detected across attack)");
    println!("{}", "═".repeat(72));

    // Header: attack names
    let valid: Vec<&AttackResult> = results.iter()
        .filter(|r| r.extraction.is_some())
        .collect();

    print!(" {:<24}", "Layer");
    for r in &valid {
        print!(" {:>7}", truncate(r.attack_name, 7));
    }
    println!("  Total");
    println!(" {}", "─".repeat(68));

    for layer_idx in 0..15usize {
        let layer_name = valid.first()
            .and_then(|r| r.extraction.as_ref())
            .map(|e| e.layer_results[layer_idx].layer_name)
            .unwrap_or("?");

        print!(" L{:<2} {:<20}", layer_idx + 1, layer_name);
        let mut count = 0usize;
        for r in &valid {
            if let Some(e) = &r.extraction {
                let lr = &e.layer_results[layer_idx];
                if lr.detected { count += 1; print!("       ✓"); }
                else           { print!("       ✗"); }
            }
        }
        println!("  {}/{}", count, valid.len());
    }
    println!("{}", "═".repeat(72));
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max { s } else { &s[..max] }
}
