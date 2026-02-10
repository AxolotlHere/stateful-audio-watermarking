mod audio;

use audio::io::{plot_waveform, read_wav};

use crate::audio::io::plot_chunks;

fn main() {
    let wave = read_wav("input_sample/Faint.wav");
    println!(
        "Sample min/max: {:?} / {:?}",
        wave.samples.iter().cloned().fold(f32::INFINITY, f32::min),
        wave.samples
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max),
    );
    let offset = 2 * wave.sample_rate as usize;
    plot_waveform(&wave, "waveform.png", offset, 20_000);
    plot_chunks(&wave.samples, wave.sample_rate, 15, "plots");

    println!(
        "Loaded {} samples @ {} Hz",
        wave.samples.len(),
        wave.sample_rate
    );
}
