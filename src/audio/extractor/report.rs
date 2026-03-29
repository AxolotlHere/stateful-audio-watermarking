//! Result types for watermark extraction.

/// Result for a single layer's detection test.
#[derive(Debug, Clone)]
pub struct LayerResult {
    /// 1-based layer number.
    pub layer_index: usize,
    /// Human-readable layer name.
    pub layer_name: &'static str,
    /// Detection score in [0.0, 1.0]. Higher = more confident.
    pub score: f32,
    /// Whether this layer's score cleared its detection threshold.
    pub detected: bool,
}

/// Full extraction result returned to the caller.
#[derive(Debug)]
pub struct ExtractionResult {
    /// The key that was tested.
    pub key: u64,
    /// Number of chunks analysed.
    pub chunk_count: usize,
    /// Sample rate of the audio.
    pub sample_rate: u32,
    /// True if overall confidence ≥ 0.60.
    pub watermark_detected: bool,
    /// Weighted confidence in [0.0, 1.0].
    pub confidence: f32,
    /// Per-layer breakdown.
    pub layer_results: Vec<LayerResult>,
    /// How many of the 15 layers individually passed their threshold.
    pub detected_count: usize,
}

impl ExtractionResult {
    /// Print a formatted verification report to stdout.
    pub fn print_report(&self) {
        println!("\n{}", "═".repeat(60));
        println!(" WATERMARK EXTRACTION REPORT");
        println!("{}", "═".repeat(60));
        println!(" Key        : 0x{:016X}", self.key);
        println!(" Sample rate: {} Hz", self.sample_rate);
        println!(" Chunks     : {}", self.chunk_count);
        println!("{}", "─".repeat(60));

        println!(" {:<4} {:<24} {:>8}  {}", "L#", "Layer", "Score", "Status");
        println!(" {}", "─".repeat(55));

        for r in &self.layer_results {
            let bar = score_bar(r.score);
            let status = if r.detected { "✓ PASS" } else { "✗ FAIL" };
            println!(
                " L{:<3} {:<24} {:>6.1}%  {} {}",
                r.layer_index,
                r.layer_name,
                r.score * 100.0,
                bar,
                status,
            );
        }

        println!("{}", "─".repeat(60));
        println!(
            " Layers detected : {}/15",
            self.detected_count
        );
        println!(
            " Overall confidence : {:.1}%",
            self.confidence * 100.0
        );
        println!("{}", "─".repeat(60));

        if self.watermark_detected {
            println!(" VERDICT: ✓  WATERMARK CONFIRMED");
        } else {
            println!(" VERDICT: ✗  WATERMARK NOT DETECTED");
            println!(" (confidence {:.1}% < 60% threshold)", self.confidence * 100.0);
        }
        println!("{}", "═".repeat(60));
    }
}

fn score_bar(score: f32) -> String {
    let filled = (score * 10.0).round() as usize;
    let empty = 10usize.saturating_sub(filled);
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}
