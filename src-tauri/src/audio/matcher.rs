use super::profile::{EQProfile, FrequencyBand};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchConfig {
    pub intensity: f32,        // 0.0 - 1.0
    pub max_correction: f32,   // Max ±dB per band
    pub smoothing_factor: f32, // 0.0 - 1.0
    pub use_psychoacoustic: bool,
    pub preserve_dynamics: bool, // Don't compress dynamic range
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            intensity: 0.7,
            max_correction: 6.0,
            smoothing_factor: 0.5,
            use_psychoacoustic: true,
            preserve_dynamics: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub correction_profile: EQProfile,
    pub reference_normalized: Vec<f32>,
    pub input_normalized: Vec<f32>,
    pub quality_score: f32, // 0.0 - 1.0
    pub warnings: Vec<String>,
}

pub fn match_profiles(
    reference: &EQProfile,
    input: &EQProfile,
    config: &MatchConfig,
) -> MatchResult {
    let mut warnings = Vec::new();

    // 1. Normalize both profiles to their mean
    let ref_normalized = normalize_profile(reference);
    let inp_normalized = normalize_profile(input);

    // 2. Calculate raw differences
    let mut corrections: Vec<FrequencyBand> = reference
        .bands
        .iter()
        .zip(&input.bands)
        .zip(&ref_normalized)
        .zip(&inp_normalized)
        .map(|(((ref_band, inp_band), &ref_norm), &inp_norm)| {
            let raw_diff = ref_norm - inp_norm;

            FrequencyBand {
                frequency: ref_band.frequency,
                gain_db: raw_diff,
                bandwidth: ref_band.bandwidth,
                confidence: (ref_band.confidence + inp_band.confidence) / 2.0,
            }
        })
        .collect();

    // 3. Apply psychoacoustic weighting
    if config.use_psychoacoustic {
        apply_psychoacoustic_weighting(&mut corrections);
    }

    // 4. Confidence-based attenuation
    apply_confidence_weighting(&mut corrections);

    // 5. Smoothing across frequency bands
    if config.smoothing_factor > 0.0 {
        smooth_corrections(&mut corrections, config.smoothing_factor);
    }

    // 6. Apply intensity scaling
    for band in &mut corrections {
        band.gain_db *= config.intensity;
    }

    // 7. Limiting
    for band in &mut corrections {
        let original = band.gain_db;
        band.gain_db = band
            .gain_db
            .clamp(-config.max_correction, config.max_correction);

        if (original - band.gain_db).abs() > 0.1 {
            warnings.push(format!(
                "{} Hz: Correction limited from {:.1} dB to {:.1} dB",
                band.frequency, original, band.gain_db
            ));
        }
    }

    // 8. Check for extreme corrections
    check_for_extreme_corrections(&corrections, &mut warnings);

    // 9. Dynamic range preservation
    let correction_profile = if config.preserve_dynamics {
        preserve_dynamic_range(reference, input, corrections)
    } else {
        EQProfile {
            bands: corrections,
            overall_loudness: 0.0, // Will be neutral
            dynamic_range: reference.dynamic_range,
            spectral_centroid: reference.spectral_centroid,
            spectral_rolloff: reference.spectral_rolloff,
        }
    };

    // 10. Calculate quality score
    let quality_score = calculate_match_quality(&correction_profile);

    MatchResult {
        correction_profile,
        reference_normalized: ref_normalized,
        input_normalized: inp_normalized,
        quality_score,
        warnings,
    }
}

fn normalize_profile(profile: &EQProfile) -> Vec<f32> {
    let gains: Vec<f32> = profile.bands.iter().map(|b| b.gain_db).collect();
    let mean = gains.iter().sum::<f32>() / gains.len() as f32;

    gains.iter().map(|&g| g - mean).collect()
}

// Fletcher-Munson inspired weighting
fn apply_psychoacoustic_weighting(bands: &mut [FrequencyBand]) {
    let weights = calculate_psychoacoustic_weights();

    for (band, weight) in bands.iter_mut().zip(weights.iter()) {
        // More weight to frequencies we're sensitive to
        band.gain_db *= weight;
    }
}

fn calculate_psychoacoustic_weights() -> Vec<f32> {
    // Based on ISO 226:2003 equal-loudness contours
    // Frequencies: 31, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k
    vec![
        0.6,  // 31 Hz - less sensitive
        0.7,  // 63 Hz
        0.85, // 125 Hz
        0.95, // 250 Hz
        1.1,  // 500 Hz - more sensitive
        1.3,  // 1 kHz - most sensitive
        1.35, // 2 kHz - most sensitive (presence)
        1.25, // 4 kHz - sibilance range
        1.0,  // 8 kHz
        0.7,  // 16 kHz - less sensitive
    ]
}

fn apply_confidence_weighting(bands: &mut [FrequencyBand]) {
    for band in bands.iter_mut() {
        // Low confidence = less correction
        let confidence_factor = band.confidence.powf(0.5); // Square root for gentler curve
        band.gain_db *= confidence_factor;
    }
}

fn smooth_corrections(bands: &mut [FrequencyBand], factor: f32) {
    if bands.len() < 3 {
        return;
    }

    let original: Vec<f32> = bands.iter().map(|b| b.gain_db).collect();

    // Multi-pass Gaussian-like smoothing
    for pass in 0..3 {
        let weight = factor * (0.7f32).powi(pass); // Decreasing weight each pass

        for i in 1..bands.len() - 1 {
            let prev = original[i - 1];
            let curr = original[i];
            let next = original[i + 1];

            // 3-point weighted average
            let smoothed = prev * 0.25 + curr * 0.5 + next * 0.25;
            bands[i].gain_db = curr * (1.0 - weight) + smoothed * weight;
        }
    }
}

fn check_for_extreme_corrections(bands: &[FrequencyBand], warnings: &mut Vec<String>) {
    // Check for steep slopes
    for window in bands.windows(2) {
        let slope = (window[1].gain_db - window[0].gain_db).abs();
        let freq_ratio = window[1].frequency / window[0].frequency;
        let slope_per_octave = slope / freq_ratio.log2();

        if slope_per_octave > 6.0 {
            warnings.push(format!(
                "Steep slope between {} Hz and {} Hz ({:.1} dB/octave)",
                window[0].frequency, window[1].frequency, slope_per_octave
            ));
        }
    }

    // Check for excessive total correction
    let total_correction: f32 = bands.iter().map(|b| b.gain_db.abs()).sum();
    if total_correction > 30.0 {
        warnings.push(format!(
            "High total correction: {:.1} dB. Consider lower intensity.",
            total_correction
        ));
    }
}

fn preserve_dynamic_range(
    reference: &EQProfile,
    input: &EQProfile,
    mut bands: Vec<FrequencyBand>,
) -> EQProfile {
    // Don't let corrections compress the dynamic range too much
    let ref_dr = reference.dynamic_range;
    let inp_dr = input.dynamic_range;

    if ref_dr < inp_dr {
        // Reference is more compressed, don't over-compress input
        let dr_ratio = ref_dr / inp_dr;
        let preservation_factor = 1.0 - (1.0 - dr_ratio).min(0.3); // Max 30% compression

        for band in &mut bands {
            band.gain_db *= preservation_factor;
        }
    }

    EQProfile {
        bands,
        overall_loudness: 0.0,
        dynamic_range: reference.dynamic_range,
        spectral_centroid: reference.spectral_centroid,
        spectral_rolloff: reference.spectral_rolloff,
    }
}

fn calculate_match_quality(profile: &EQProfile) -> f32 {
    let mut score = 1.0;

    // Penalty for large corrections
    let avg_correction: f32 =
        profile.bands.iter().map(|b| b.gain_db.abs()).sum::<f32>() / profile.bands.len() as f32;

    score -= (avg_correction / 10.0).min(0.4); // Max -0.4 penalty

    // Penalty for steep slopes
    for window in profile.bands.windows(2) {
        let slope = (window[1].gain_db - window[0].gain_db).abs();
        if slope > 6.0 {
            score -= 0.05;
        }
    }

    // Bonus for high confidence
    let avg_confidence: f32 =
        profile.bands.iter().map(|b| b.confidence).sum::<f32>() / profile.bands.len() as f32;

    score *= 0.7 + 0.3 * avg_confidence;

    score.clamp(0.0, 1.0)
}
