use hound;
use plotters::prelude::*;
use std::fs;
use std::path::Path;

const CHUNK_SECONDS: usize = 15;
const DOWNSAMPLE: usize = 256;

pub struct WaveData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub struct ChunkFeatures {
    pub rms: f32,
    // later:
    // pub centroid: f32,
    // pub spectrum: Vec<f32>,
}

pub struct FeatureSet {
    pub chunk_seconds: usize,
    pub features: Vec<ChunkFeatures>,
}

pub fn extract_rms_features(samples: &[f32], sample_rate: u32, chunk_seconds: usize) -> FeatureSet {
    let samples_per_chunk = chunk_seconds * sample_rate as usize;
    let total_chunks = samples.len() / samples_per_chunk;

    let mut features = Vec::with_capacity(total_chunks);

    for chunk_idx in 0..total_chunks {
        let start = chunk_idx * samples_per_chunk;
        let end = start + samples_per_chunk;

        let chunk = &samples[start..end];
        let rms = compute_rms_stereo(chunk, 1);

        features.push(ChunkFeatures { rms });
    }

    FeatureSet {
        chunk_seconds,
        features,
    }
}

pub fn compute_rms_stereo(samples: &[f32], channels: usize) -> f32 {
    if samples.is_empty() || channels == 0 {
        return 0.0;
    }

    let mut sum_sq = 0.0;
    let mut count = 0;

    for frame in samples.chunks(channels) {
        for &s in frame {
            sum_sq += s * s;
            count += 1;
        }
    }

    (sum_sq / count as f32).sqrt()
}

//Reset Directory
fn reset_plot_dir(dir: &str) {
    let path = Path::new(dir);
    if path.exists() {
        fs::remove_dir_all(path).expect("Failed to clear plot dir");
    }
    fs::create_dir_all(path).expect("Failed to create plot dir");
}

//Chunk at plot subdir
pub fn plot_chunks(samples: &[f32], sample_rate: u32, chunk_seconds: usize, out_dir: &str) {
    reset_plot_dir(out_dir);

    let samples_per_chunk = chunk_seconds * sample_rate as usize;
    let total_chunks = samples.len() / samples_per_chunk;

    println!("Generating {} plots…", total_chunks);

    for chunk_idx in 0..total_chunks {
        let start = chunk_idx * samples_per_chunk;
        let end = start + samples_per_chunk;
        let chunk = &samples[start..end];

        let filename = format!("{}/chunk_{:03}.png", out_dir, chunk_idx);

        let root = BitMapBackend::new(&filename, (1600, 400)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!(
                    "Chunk {} ({}–{} sec)",
                    chunk_idx,
                    chunk_idx * chunk_seconds,
                    (chunk_idx + 1) * chunk_seconds
                ),
                ("sans-serif", 20),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0f32..chunk_seconds as f32, -1.0f32..1.0f32)
            .unwrap();

        chart
            .configure_mesh()
            .x_desc("Time (s)")
            .y_desc("Amplitude")
            .draw()
            .unwrap();

        let step = DOWNSAMPLE;

        let series: Vec<(f32, f32)> = chunk
            .iter()
            .step_by(step)
            .enumerate()
            .map(|(i, &y)| {
                let t = i as f32 * step as f32 / sample_rate as f32;
                (t, y)
            })
            .collect();

        chart
            .draw_series(std::iter::once(PathElement::new(series, &BLUE)))
            .unwrap();
    }
}

// Read and Normalize
pub fn read_wav(path: &str) -> WaveData {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV");

    let spec = reader.spec();
    println!("WAV spec: {:?}", spec);

    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
        (hound::SampleFormat::Int, 24) => reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / (1 << 23) as f32)
            .collect(),
        (hound::SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| s.unwrap() as f32 / i32::MAX as f32)
            .collect(),
        (hound::SampleFormat::Float, _) => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        _ => panic!("Unsupported WAV format"),
    };

    // Down-mix to mono
    let mono_samples = if channels > 1 {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples.clone()
    };
    //Compute RMS for mono_samples

    let frames_per_chunk = 15 * sample_rate as usize;

    WaveData {
        samples: mono_samples,
        sample_rate,
    }
}

// Plot waveform as PNG
//Whole plot

pub fn plot_waveform(wave: &WaveData, out: &str, max_samples: usize, start: usize) {
    let end = (start + max_samples).min(wave.samples.len());
    let samples = &wave.samples[start..end];

    let root = BitMapBackend::new(out, (1200, 400)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("Waveform", ("sans-serif", 20))
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..samples.len() as f32, -1.1f32..1.1f32)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    chart
        .draw_series(LineSeries::new(
            samples.iter().enumerate().map(|(i, s)| ((i) as f32, *s)),
            &BLUE,
        ))
        .unwrap();

    root.present().unwrap();
}
