//Common interface for all audio transformation layers.

pub trait Layer: Send + Sync {
    //for logging / debugging.
    fn name(&self) -> &'static str;

    // Apply the transformation in-place.
    // `samples` - mono f32 PCM, range roughly [-1.0, 1.0]
    // `sample_rate` – roughly 44100 or 48000
    fn apply(&self, samples: &mut [f32], sample_rate: u32);
}
