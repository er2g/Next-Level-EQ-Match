use super::analyzer::{AnalysisConfig, FrequencySpectrum};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::statistics::Statistics;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EQProfile {
    pub bands: Vec<FrequencyBand>,
    pub overall_loudness: f32,  // LUFS or dB
    pub dynamic_range: f32,     // dB
    pub spectral_centroid: f32, // Hz
    pub spectral_rolloff: f32,  // Hz
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyBand {
    pub frequency: f32,
    pub gain_db: f32,
    pub bandwidth: f32,  // For Q calculation
    pub confidence: f32, // 0.0 - 1.0
}

pub fn extract_eq_profile(spectrum: &FrequencySpectrum, config: &AnalysisConfig) -> EQProfile {
    let bands = config
        .frequency_bands
        .par_iter()
        .map(|&center_freq| extract_band_info(spectrum, center_freq))
        .collect();

    let overall_loudness = calculate_overall_loudness(&spectrum.magnitudes);
    let dynamic_range = calculate_dynamic_range(&spectrum.magnitudes);
    let spectral_centroid = calculate_spectral_centroid(spectrum);
    let spectral_rolloff = calculate_spectral_rolloff(spectrum, 0.85);

    EQProfile {
        bands,
        overall_loudness,
        dynamic_range,
        spectral_centroid,
        spectral_rolloff,
    }
}

fn extract_band_info(spectrum: &FrequencySpectrum, center_freq: f32) -> FrequencyBand {
    // 1/3 octave bandwidth
    let bandwidth = center_freq * 0.23;
    let lower = center_freq / 2.0f32.powf(1.0 / 6.0);
    let upper = center_freq * 2.0f32.powf(1.0 / 6.0);

    // Find bins in this range
    let bin_indices: Vec<usize> = spectrum
        .frequencies
        .iter()
        .enumerate()
        .filter_map(|(i, &freq)| {
            if freq >= lower && freq <= upper {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    if bin_indices.is_empty() {
        return FrequencyBand {
            frequency: center_freq,
            gain_db: -80.0,
            bandwidth,
            confidence: 0.0,
        };
    }

    // Calculate RMS energy in band
    let band_magnitudes: Vec<f64> = bin_indices
        .iter()
        .map(|&i| spectrum.magnitudes[i] as f64)
        .collect();

    let gain_db = band_magnitudes.clone().mean() as f32;

    // Confidence based on consistency
    let std_dev = band_magnitudes.clone().std_dev() as f32;
    let confidence = (1.0 / (1.0 + std_dev / 10.0)).clamp(0.0, 1.0);

    FrequencyBand {
        frequency: center_freq,
        gain_db,
        bandwidth,
        confidence,
    }
}

fn calculate_overall_loudness(magnitudes: &[f32]) -> f32 {
    // A-weighting approximation
    let rms: f32 = magnitudes
        .iter()
        .map(|&m| {
            let linear = 10.0f32.powf(m / 20.0);
            linear * linear
        })
        .sum::<f32>()
        / magnitudes.len() as f32;

    20.0 * rms.sqrt().log10()
}

fn calculate_dynamic_range(magnitudes: &[f32]) -> f32 {
    let mut sorted = magnitudes.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let percentile_95 = sorted[(sorted.len() as f32 * 0.95) as usize];
    let percentile_5 = sorted[(sorted.len() as f32 * 0.05) as usize];

    percentile_95 - percentile_5
}

fn calculate_spectral_centroid(spectrum: &FrequencySpectrum) -> f32 {
    let mut weighted_sum = 0.0;
    let mut total_magnitude = 0.0;

    for (freq, mag) in spectrum.frequencies.iter().zip(&spectrum.magnitudes) {
        let linear_mag = 10.0f32.powf(mag / 20.0);
        weighted_sum += freq * linear_mag;
        total_magnitude += linear_mag;
    }

    if total_magnitude == 0.0 {
        return 0.0;
    }

    weighted_sum / total_magnitude
}

fn calculate_spectral_rolloff(spectrum: &FrequencySpectrum, threshold: f32) -> f32 {
    let total_energy: f32 = spectrum
        .magnitudes
        .iter()
        .map(|&m| 10.0f32.powf(m / 10.0))
        .sum();

    let threshold_energy = total_energy * threshold;
    let mut cumulative = 0.0;

    for (freq, mag) in spectrum.frequencies.iter().zip(&spectrum.magnitudes) {
        cumulative += 10.0f32.powf(mag / 10.0);
        if cumulative >= threshold_energy {
            return *freq;
        }
    }

    spectrum.frequencies.last().copied().unwrap_or(20000.0)
}
