use std::fs::File;
use std::path::Path;
use symphonia::core::audio::{AudioBuffer, AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AudioError {
    #[error("Dosya açılamadı: {0}")]
    FileOpen(#[from] std::io::Error),

    #[error("Format desteklenmiyor")]
    UnsupportedFormat,

    #[error("Decode hatası: {0}")]
    DecodeError(String),

    #[error("Ses verisi bulunamadı")]
    NoAudioData,
}

pub struct AudioData {
    pub samples: Vec<f32>, // Mono, normalized -1.0 to 1.0
    pub sample_rate: u32,
    pub duration_secs: f64,
    pub channels: u16,
}

#[derive(Debug, Clone, Copy)]
pub enum ResamplePolicy {
    /// Always resample when `from_rate != target_rate`.
    Always,
    /// Only resample when `from_rate > target_rate` (downsample); skip upsampling.
    DownsampleOnly,
    /// Never resample.
    Never,
}

pub fn load_audio_file<P: AsRef<Path>>(path: P) -> Result<AudioData, AudioError> {
    let file = File::open(&path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    // Format hint from extension
    let mut hint = Hint::new();
    if let Some(ext) = path.as_ref().extension() {
        hint.with_extension(ext.to_str().unwrap_or(""));
    }

    // Probe format
    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|_| AudioError::UnsupportedFormat)?;

    let mut format = probed.format;
    let track = format.default_track().ok_or(AudioError::NoAudioData)?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| AudioError::DecodeError(e.to_string()))?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(2) as u16;

    let mut all_samples = Vec::new();

    // Decode all packets
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };

        let decoded = decoder
            .decode(&packet)
            .map_err(|e| AudioError::DecodeError(e.to_string()))?;

        // Convert to f32 and mix to mono
        let samples = convert_to_mono_f32(&decoded, channels);
        all_samples.extend(samples);
    }

    let duration_secs = all_samples.len() as f64 / sample_rate as f64;

    Ok(AudioData {
        samples: all_samples,
        sample_rate,
        duration_secs,
        channels,
    })
}

fn convert_to_mono_f32(audio_buf: &AudioBufferRef, channels: u16) -> Vec<f32> {
    match audio_buf {
        AudioBufferRef::F32(buf) => convert_buffer_to_mono(buf, channels),
        AudioBufferRef::S32(buf) => convert_buffer_to_mono(buf, channels),
        AudioBufferRef::S16(buf) => convert_buffer_to_mono(buf, channels),
        AudioBufferRef::U8(buf) => convert_buffer_to_mono(buf, channels),
        _ => Vec::new(),
    }
}

fn convert_buffer_to_mono<S>(buf: &AudioBuffer<S>, channels: u16) -> Vec<f32>
where
    S: symphonia::core::sample::Sample,
    f32: symphonia::core::conv::FromSample<S>,
{
    use symphonia::core::conv::FromSample;

    let num_samples = buf.frames();
    let mut mono = Vec::with_capacity(num_samples);

    if channels == 1 {
        // Already mono
        for frame in 0..num_samples {
            mono.push(f32::from_sample(buf.chan(0)[frame]));
        }
    } else {
        // Mix to mono
        for frame in 0..num_samples {
            let mut sum = 0.0;
            for ch in 0..channels as usize {
                sum += f32::from_sample(buf.chan(ch)[frame]);
            }
            mono.push(sum / channels as f32);
        }
    }

    mono
}

// Resampling for matching sample rates
pub fn resample_audio(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
) -> Result<Vec<f32>, AudioError> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    // Process in chunks to avoid allocating/copying the full file into rubato buffers.
    // `SincFixedIn` requires fixed-size input chunks.
    let chunk_size = 16_384usize.min(samples.len().max(1));

    let mut resampler = SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        2.0,
        params,
        chunk_size,
        1,
    )
    .map_err(|e| AudioError::DecodeError(format!("Resampling error: {}", e)))?;

    let mut wave_in = vec![vec![0.0f32; resampler.input_frames_next()]];
    let mut wave_out = vec![vec![0.0f32; resampler.output_frames_max()]];

    let mut out = Vec::with_capacity(
        ((samples.len() as u64 * to_rate as u64) / from_rate as u64 + 1024) as usize,
    );

    let mut offset = 0usize;
    while offset + wave_in[0].len() <= samples.len() {
        let in_len = wave_in[0].len();
        wave_in[0].copy_from_slice(&samples[offset..offset + in_len]);
        let (_, out_len) = resampler
            .process_into_buffer(&wave_in, &mut wave_out, None)
            .map_err(|e| AudioError::DecodeError(format!("Resampling error: {}", e)))?;
        out.extend_from_slice(&wave_out[0][..out_len]);
        offset += in_len;
    }

    // Handle the last partial chunk and flush delayed frames.
    let remaining = &samples[offset..];
    if !remaining.is_empty() {
        let (_, out_len) = resampler
            .process_partial_into_buffer(Some(&[remaining]), &mut wave_out, None)
            .map_err(|e| AudioError::DecodeError(format!("Resampling error: {}", e)))?;
        out.extend_from_slice(&wave_out[0][..out_len]);
    }
    let (_, out_len) = resampler
        .process_partial_into_buffer::<Vec<f32>, Vec<f32>>(None, &mut wave_out, None)
        .map_err(|e| AudioError::DecodeError(format!("Resampling error: {}", e)))?;
    out.extend_from_slice(&wave_out[0][..out_len]);

    Ok(out)
}

pub fn prepare_audio_for_analysis(
    samples: Vec<f32>,
    from_rate: u32,
    target_rate: u32,
    policy: ResamplePolicy,
) -> Result<(Vec<f32>, u32), AudioError> {
    match policy {
        ResamplePolicy::Never => Ok((samples, from_rate)),
        ResamplePolicy::DownsampleOnly => {
            if from_rate > target_rate {
                Ok((
                    resample_audio(&samples, from_rate, target_rate)?,
                    target_rate,
                ))
            } else {
                Ok((samples, from_rate))
            }
        }
        ResamplePolicy::Always => {
            if from_rate != target_rate {
                Ok((
                    resample_audio(&samples, from_rate, target_rate)?,
                    target_rate,
                ))
            } else {
                Ok((samples, from_rate))
            }
        }
    }
}
