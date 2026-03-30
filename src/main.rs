mod audio;

use audio::io::{plot_chunks, plot_waveform, read_wav, write_wav};
use audio::layers::{build_layers, permute_layers, KeyedChunker};
use audio::extractor::{extract, extract_with_ref, Fingerprint};
use audio::robustness::{
    run_robustness_test, print_robustness_report, print_layer_survival
};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("embed");
    let key: u64 = args.get(2)
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0xDEAD_BEEF_1234_5678);

    match mode {
        "verify"      => run_verify(key, true),
        "verify-blind"=> run_verify(key, false),
        "robustness"  => run_robustness(key),
        "security"    => run_security_test(key),
        _             => run_embed(key),
    }
}

fn run_embed(key: u64) {
    println!("═══ EMBED MODE ═══════════════════════════════════════════");

    let mut wave = read_wav("input_sample/Faint.wav");
    println!("Loaded {} samples @ {} Hz", wave.samples.len(), wave.sample_rate);

    let layers  = build_layers(key, wave.sample_rate);
    let order   = permute_layers(key);
    let chunker = KeyedChunker::new(key, wave.sample_rate);

    println!("Key        : 0x{:016X}", key);
    println!("Chunk size : {} samples ({} s)",
        chunker.chunk_size_samples(),
        chunker.chunk_size_samples() / wave.sample_rate as usize);

    print!("Capturing original fingerprint... ");
    let fp = Fingerprint::capture(
        &wave.samples, wave.sample_rate, key, chunker.chunk_size_samples());
    fp.save("original.wmpf").expect("Failed to save fingerprint");
    println!("saved {} chunks → original.wmpf", fp.chunks.len());

    let mut chunk_count = 0usize;
    for chunk in chunker.iter_chunks_mut(&mut wave.samples) {
        for &layer_idx in &order {
            layers[layer_idx].apply(chunk, wave.sample_rate);
        }
        chunker.apply_boundary_fade(chunk);
        chunk_count += 1;
    }
    println!("Watermarked {} chunks.", chunk_count);

    let peak = wave.samples.iter().cloned().fold(0.0_f32, |a, s| a.max(s.abs()));
    println!("Peak amplitude: {:.6} ({})",
        peak, if peak <= 1.0 { "OK — no clipping" } else { "WARNING — clipping!" });

    let offset = 2 * wave.sample_rate as usize;
    plot_waveform(&wave, "waveform_watermarked.png", offset, 20_000);
    plot_chunks(&wave.samples, wave.sample_rate, 15, "plots");
    write_wav(&wave, "output_watermarked.wav");

    println!("\nOutput: output_watermarked.wav + original.wmpf");
    println!("To verify    : cargo run -- verify 0x{:016X}", key);
    println!("To test robustness: cargo run -- robustness 0x{:016X}", key);
}

fn run_verify(key: u64, use_ref: bool) {
    if use_ref {
        println!("═══ VERIFY MODE (reference-based) ════════════════════════");
    } else {
        println!("═══ VERIFY MODE (blind) ═══════════════════════════════════");
    }

    let wave = read_wav("output_watermarked.wav");
    println!("Loaded {} samples @ {} Hz", wave.samples.len(), wave.sample_rate);

    let result = if use_ref {
        match Fingerprint::load("original.wmpf") {
            Ok(fp) => {
                if fp.key != key {
                    println!("WARNING: fingerprint key mismatch!");
                    println!("  Fingerprint: 0x{:016X}", fp.key);
                    println!("  Provided   : 0x{:016X}", key);
                }
                println!("Loaded fingerprint: {} chunks", fp.chunks.len());
                extract_with_ref(&wave.samples, wave.sample_rate, key, &fp)
            }
            Err(e) => {
                println!("Could not load original.wmpf: {e}");
                println!("Falling back to blind extraction...");
                extract(&wave.samples, wave.sample_rate, key)
            }
        }
    } else {
        extract(&wave.samples, wave.sample_rate, key)
    };

    result.print_report();
}

fn run_robustness(key: u64) {
    println!("═══ ROBUSTNESS TEST ══════════════════════════════════════");
    println!(" Watermarked file : output_watermarked.wav");
    println!(" Fingerprint      : original.wmpf");
    println!(" Key              : 0x{:016X}", key);

    // First get baseline sample rate
    let wave = read_wav("output_watermarked.wav");
    let sample_rate = wave.sample_rate;
    drop(wave);

    println!("\n Running attacks...\n");
    let results = run_robustness_test(
        "output_watermarked.wav",
        "original.wmpf",
        key,
        sample_rate,
    );

    // Baseline confidence (first result)
    let baseline_conf = results.first()
        .and_then(|r| r.confidence())
        .unwrap_or(0.0);

    print_robustness_report(&results, baseline_conf);
    print_layer_survival(&results);
}

// ─── appended: security test mode ────────────────────────────────────────────

fn run_security_test(key: u64) {
    println!("═══ SECURITY TEST — WRONG KEY ANALYSIS ══════════════════");
    use audio::extractor::run_key_test;

    let wave = audio::io::read_wav("output_watermarked.wav");
    let fp = match audio::extractor::Fingerprint::load("original.wmpf") {
        Ok(f) => f,
        Err(e) => { eprintln!("Cannot load fingerprint: {e}"); return; }
    };

    println!(" Testing 100 wrong keys (this takes ~30s)...\n");
    let result = run_key_test(&wave.samples, wave.sample_rate, key, &fp, 100);
    result.print_report();
}
