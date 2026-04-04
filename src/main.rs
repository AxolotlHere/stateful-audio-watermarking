mod audio;

use audio::io::{plot_chunks, plot_waveform, read_wav, write_wav};
use audio::layers::{apply_chained_layers, permute_layers, KeyedChunker};
use audio::extractor::{extract_with_ref, Fingerprint};
use audio::extractor::key_test::run_key_test;
use audio::metrics::run_metrics;
use audio::robustness::{run_robustness_test, print_robustness_report, print_layer_survival};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("embed");
    let key: u64 = args.get(2)
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0xDEAD_BEEF_1234_5678);

    match mode {
        "verify"      => run_verify(key),
        "metrics"     => run_metrics_mode(key, &args),
        "robustness"  => run_robustness(key),
        "security"    => run_security(key),
        _             => run_embed(key),
    }
}

fn run_embed(key: u64) {
    println!("═══ EMBED MODE ═══════════════════════════════════════════");

    let mut wave = read_wav("input_sample/Faint-1.wav");
    println!("Loaded {} samples @ {} Hz", wave.samples.len(), wave.sample_rate);

    let order   = permute_layers(key);
    let chunker = KeyedChunker::new(key, wave.sample_rate);

    println!("Key        : 0x{:016X}", key);
    println!("Chunk size : {} samples ({} s)",
        chunker.chunk_size_samples(),
        chunker.chunk_size_samples() / wave.sample_rate as usize);
    println!("Chunk offset: {} samples", chunker.chunk_offset_samples());

    // ── Capture fingerprint BEFORE watermarking ───────────────────────────
    print!("Capturing original fingerprint... ");
    let fp = Fingerprint::capture(
        &wave.samples,
        wave.sample_rate,
        key,
        chunker.chunk_size_samples(),
        chunker.chunk_offset_samples(),
    );
    fp.save("original.wmpf").expect("Failed to save fingerprint");
    println!("saved {} chunks → original.wmpf", fp.chunks.len());

    // ── Apply chained watermark ───────────────────────────────────────────
    let total_chunks = chunker.chunk_count(wave.samples.len());
    let mut chunk_count = 0usize;
    let mut watermarked_chunks = 0usize;
    for (chunk_idx, chunk) in chunker.iter_chunks_mut(&mut wave.samples).enumerate() {
        if chunker.should_watermark_chunk(chunk_idx, total_chunks) {
            for (region_idx, (start, end)) in chunker.watermark_windows(chunk_idx, chunk.len()).into_iter().enumerate() {
                let region_seed = keyed_region_seed(chunk_idx, region_idx);
                let original_region = chunk[start..end].to_vec();
                apply_chained_layers(&mut chunk[start..end], wave.sample_rate, key, &order, region_seed);
                chunker.blend_region_edges(&original_region, &mut chunk[start..end]);
            }
            watermarked_chunks += 1;
        }
        chunk_count += 1;
    }
    println!(
        "Watermarked {}/{} chunks (sparse keyed pipeline).",
        watermarked_chunks,
        chunk_count
    );

    let peak = wave.samples.iter().cloned().fold(0.0_f32, |a, s| a.max(s.abs()));
    println!("Peak: {:.6} ({})",
        peak, if peak <= 1.0 { "OK" } else { "WARNING — clipping!" });

    let offset = 2 * wave.sample_rate as usize;
    plot_waveform(&wave, "waveform_watermarked.png", offset, 20_000);
    plot_chunks(&wave.samples, wave.sample_rate, 15, "plots");
    write_wav(&wave, "output_watermarked.wav");

    println!("\nOutput: output_watermarked.wav + original.wmpf");
    println!("Metrics   : cargo run -- metrics 0x{:016X}", key);
    println!("Verify    : cargo run -- verify 0x{:016X}", key);
    println!("Security  : cargo run -- security 0x{:016X}", key);
    println!("Robustness: cargo run -- robustness 0x{:016X}", key);
}

fn run_verify(key: u64) {
    println!("═══ VERIFY MODE ══════════════════════════════════════════");
    let wave = read_wav("output_watermarked.wav");
    println!("Loaded {} samples @ {} Hz", wave.samples.len(), wave.sample_rate);

    match Fingerprint::load("original.wmpf") {
        Ok(fp) => {
            if fp.key != key {
                println!("WARNING: fingerprint key mismatch!");
                println!("  Fingerprint: 0x{:016X}", fp.key);
                println!("  Provided   : 0x{:016X}", key);
            }
            println!("Loaded fingerprint: {} chunks", fp.chunks.len());
            let result = extract_with_ref(&wave.samples, wave.sample_rate, key, &fp);
            result.print_report();
        }
        Err(e) => eprintln!("Could not load original.wmpf: {e}"),
    }
}

fn run_metrics_mode(key: u64, args: &[String]) {
    println!("═══ METRICS MODE ═════════════════════════════════════════");

    let original_wav = args.get(3).map(String::as_str).unwrap_or("input_sample/Faint-1.wav");
    let watermarked_wav = args.get(4).map(String::as_str).unwrap_or("output_watermarked.wav");
    let fingerprint_path = args.get(5).map(String::as_str).unwrap_or("original.wmpf");

    println!(" Original    : {}", original_wav);
    println!(" Watermarked : {}", watermarked_wav);
    println!(" Fingerprint : {}", fingerprint_path);

    match run_metrics(original_wav, watermarked_wav, fingerprint_path, key) {
        Ok(report) => report.print_report(),
        Err(err) => eprintln!("Metrics failed: {err}"),
    }
}

fn run_security(key: u64) {
    println!("═══ SECURITY TEST ════════════════════════════════════════");
    let wave = read_wav("output_watermarked.wav");
    let fp = match Fingerprint::load("original.wmpf") {
        Ok(f) => f,
        Err(e) => { eprintln!("Cannot load fingerprint: {e}"); return; }
    };
    println!(" Testing 100 wrong keys...\n");
    let result = run_key_test(&wave.samples, wave.sample_rate, key, &fp, 100);
    result.print_report();
}

fn run_robustness(key: u64) {
    println!("═══ ROBUSTNESS TEST ══════════════════════════════════════");
    let wave = read_wav("output_watermarked.wav");
    let sample_rate = wave.sample_rate;
    drop(wave);
    println!("\n Running attacks...\n");
    let results = run_robustness_test(
        "output_watermarked.wav", "original.wmpf", key, sample_rate);
    let baseline = results.first().and_then(|r| r.confidence()).unwrap_or(0.0);
    print_robustness_report(&results, baseline);
    print_layer_survival(&results);
}

fn keyed_region_seed(chunk_idx: usize, region_idx: usize) -> usize {
    chunk_idx
        .wrapping_mul(8)
        .wrapping_add(region_idx)
}
