mod audio;

use audio::io::{plot_chunks, plot_waveform, read_wav, write_wav};
use audio::layers::{build_layers, permute_layers, KeyedChunker};
use audio::extractor::extract;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    // ── Mode selection ────────────────────────────────────────────────────────
    // cargo run                        → embed with default key
    // cargo run -- embed <key_hex>     → embed with custom key
    // cargo run -- verify <key_hex>    → verify output_watermarked.wav

    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("embed");
    let key: u64 = args.get(2)
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0xDEAD_BEEF_1234_5678);

    match mode {
        "verify" => run_verify(key),
        _        => run_embed(key),
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
    print!("Layer order: [");
    for (i, &idx) in order.iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("{}", idx + 1);
    }
    println!("]");

    // Plot original
    let offset = 2 * wave.sample_rate as usize;
    plot_waveform(&wave, "waveform_original.png", offset, 20_000);

    // Apply watermark
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
        peak, if peak <= 1.0 { "OK" } else { "CLIPPING!" });

    plot_waveform(&wave, "waveform_watermarked.png", offset, 20_000);
    plot_chunks(&wave.samples, wave.sample_rate, 15, "plots");
    write_wav(&wave, "output_watermarked.wav");

    println!("Written → output_watermarked.wav");
    println!("\nTo verify, run:");
    println!("  cargo run -- verify 0x{:016X}", key);
}

fn run_verify(key: u64) {
    println!("═══ VERIFY MODE ══════════════════════════════════════════");

    let wave = read_wav("output_watermarked.wav");
    println!("Loaded {} samples @ {} Hz", wave.samples.len(), wave.sample_rate);

    let result = extract(&wave.samples, wave.sample_rate, key);
    result.print_report();
}
