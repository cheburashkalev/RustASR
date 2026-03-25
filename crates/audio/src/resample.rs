//! Audio resampling.

use asr_core::{AsrError, AsrResult, AudioBuffer};
use rubato::{FftFixedInOut, Resampler as RubatoResampler};

/// Audio resampler for converting sample rates.
pub struct Resampler {
    target_sample_rate: usize,
}

impl Resampler {
    /// Create a new resampler with target sample rate.
    pub fn new(target_sample_rate: usize) -> Self {
        Self { target_sample_rate }
    }

    /// Resample audio buffer to target sample rate.
    pub fn resample(&self, buffer: &AudioBuffer) -> AsrResult<AudioBuffer> {
        if buffer.sample_rate == self.target_sample_rate {
            return Ok(buffer.clone());
        }

        // Ensure mono audio
        if buffer.channels != 1 {
            return Err(AsrError::Audio(
                "Resampling requires mono audio. Use to_mono() first.".to_string(),
            ));
        }

        let ratio = self.target_sample_rate as f64 / buffer.sample_rate as f64;
        let chunk_size = 1024;

        let mut resampler = FftFixedInOut::<f32>::new(
            buffer.sample_rate,
            self.target_sample_rate,
            chunk_size,
            1, // mono
        )
        .map_err(|e| AsrError::Audio(format!("Failed to create resampler: {}", e)))?;

        let mut output = Vec::with_capacity((buffer.samples.len() as f64 * ratio).ceil() as usize);

        // Обрабатываем только полные блоки, которые требуются на текущем шаге.
        let mut pos = 0;
        while pos < buffer.samples.len() {
            let frames_in = resampler.input_frames_next();
            if pos + frames_in > buffer.samples.len() {
                break;
            }

            let input_chunk = vec![buffer.samples[pos..pos + frames_in].to_vec()];
            let output_chunk = resampler
                .process(&input_chunk, None)
                .map_err(|e| AsrError::Audio(format!("Resampling failed: {}", e)))?;
            output.extend_from_slice(&output_chunk[0]);
            pos += frames_in;
        }

        // Хвост отправляем через process_partial, чтобы rubato сам сделал корректный zero-padding.
        if pos < buffer.samples.len() {
            let input_chunk = vec![buffer.samples[pos..].to_vec()];
            let output_chunk = resampler
                .process_partial(Some(&input_chunk), None)
                .map_err(|e| AsrError::Audio(format!("Resampling failed: {}", e)))?;
            output.extend_from_slice(&output_chunk[0]);
        }

        Ok(AudioBuffer::new(output, self.target_sample_rate, 1))
    }
}

impl Default for Resampler {
    fn default() -> Self {
        Self::new(16000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resampler_no_change() {
        let buffer = AudioBuffer::new(vec![0.0; 1024], 16000, 1);
        let resampler = Resampler::new(16000);
        let result = resampler.resample(&buffer).unwrap();

        assert_eq!(result.sample_rate, 16000);
        assert_eq!(result.samples.len(), buffer.samples.len());
    }

    #[test]
    fn test_resampler_48k_to_16k_irregular_len() {
        let input_len = 48_000 + 137;
        let buffer = AudioBuffer::new(vec![0.0; input_len], 48_000, 1);
        let resampler = Resampler::new(16_000);
        let result = resampler.resample(&buffer).unwrap();

        assert_eq!(result.sample_rate, 16_000);
        assert!(!result.samples.is_empty());
    }
}
