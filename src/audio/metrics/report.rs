//! Formatted reporting for patent-oriented watermark metrics.

/// BER summary for one spread-spectrum layer treated as a presence-bit carrier.
#[derive(Debug, Clone)]
pub struct BerMetric {
    pub layer_index: usize,
    pub layer_name: &'static str,
    pub total_bits: usize,
    pub error_bits: usize,
    pub ber: f32,
    pub mean_corr: f32,
}

/// Distortion summary for one layer in the chained embed order.
#[derive(Debug, Clone)]
pub struct LayerDistortionMetric {
    pub slot_index: usize,
    pub layer_index: usize,
    pub layer_name: &'static str,
    pub incremental_error_rms: f32,
    pub cumulative_error_rms: f32,
}

/// Full metrics report.
#[derive(Debug, Clone)]
pub struct MetricsReport {
    pub key: u64,
    pub sample_rate: u32,
    pub aligned_samples: usize,
    pub active_region_samples: usize,
    pub full_original_peak: f32,
    pub full_original_rms: f32,
    pub full_error_rms: f32,
    pub full_snr_db: f32,
    pub full_psnr_db: f32,
    pub active_original_peak: f32,
    pub active_original_rms: f32,
    pub active_error_rms: f32,
    pub active_snr_db: f32,
    pub active_psnr_db: f32,
    pub ber_metrics: Vec<BerMetric>,
    pub layer_distortion: Vec<LayerDistortionMetric>,
}

impl MetricsReport {
    pub fn print_report(&self) {
        println!("\n{}", "═".repeat(60));
        println!(" PATENT METRICS REPORT");
        println!("{}", "═".repeat(60));
        println!(" Key             : 0x{:016X}", self.key);
        println!(" Sample rate     : {} Hz", self.sample_rate);
        println!(" Aligned samples : {}", self.aligned_samples);
        println!(" Active region samples : {}", self.active_region_samples);
        println!("{}", "─".repeat(60));
        println!(" Full-file metrics");
        println!(" Original RMS    : {:.6}", self.full_original_rms);
        println!(" Error RMS       : {:.6}", self.full_error_rms);
        println!(" Original peak   : {:.6}", self.full_original_peak);
        println!(" SNR             : {}", fmt_db(self.full_snr_db));
        println!(" PSNR            : {}", fmt_db(self.full_psnr_db));
        println!("{}", "─".repeat(60));
        println!(" Active watermark-region metrics");
        println!(" Original RMS    : {:.6}", self.active_original_rms);
        println!(" Error RMS       : {:.6}", self.active_error_rms);
        println!(" Original peak   : {:.6}", self.active_original_peak);
        println!(" SNR             : {}", fmt_db(self.active_snr_db));
        println!(" PSNR            : {}", fmt_db(self.active_psnr_db));
        println!("{}", "─".repeat(60));
        println!(" {:<4} {:<24} {:>8} {:>8} {:>10}", "L#", "Carrier", "Bits", "Errors", "PBER");
        println!(" {}", "─".repeat(57));

        for metric in &self.ber_metrics {
            println!(
                " L{:<3} {:<24} {:>8} {:>8} {:>8.2}% ",
                metric.layer_index,
                metric.layer_name,
                metric.total_bits,
                metric.error_bits,
                metric.ber * 100.0,
            );
            println!(
                " mean matched corr: {:.4}",
                metric.mean_corr,
            );
        }

        println!("{}", "─".repeat(60));
        println!(" PBER assumption : one presence bit per keyed region for L10/L12");
        println!(" Tx bit value    : 1 (sequence present), Rx bit = matched-filter sign");
        println!(" Patent note     : active-region SNR/PSNR isolates the actually watermarked spans");
        println!("{}", "─".repeat(60));
        println!(" Layer distortion ranking (active regions, chained order)");
        println!(" {:<5} {:<4} {:<24} {:>10} {:>10}", "Slot", "L#", "Layer", "Delta RMS", "Cum RMS");
        println!(" {}", "─".repeat(59));

        let mut ranked = self.layer_distortion.clone();
        ranked.sort_by(|a, b| b.incremental_error_rms.total_cmp(&a.incremental_error_rms));
        for metric in ranked {
            println!(
                " {:<5} L{:<3} {:<24} {:>10.6} {:>10.6}",
                metric.slot_index + 1,
                metric.layer_index,
                metric.layer_name,
                metric.incremental_error_rms,
                metric.cumulative_error_rms,
            );
        }
        println!("{}", "═".repeat(60));
    }
}

fn fmt_db(value: f32) -> String {
    if value.is_infinite() && value.is_sign_positive() {
        "inf dB".to_string()
    } else {
        format!("{value:.2} dB")
    }
}
