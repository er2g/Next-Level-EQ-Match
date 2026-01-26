#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use eq_matcher::audio::analyzer::{analyze_spectrum, AnalysisConfig};
use eq_matcher::audio::loader::{load_audio_file, prepare_audio_for_analysis, ResamplePolicy};
use eq_matcher::audio::matcher::{match_profiles, MatchConfig, MatchResult};
use eq_matcher::audio::profile::{extract_eq_profile, EQProfile};
use std::sync::Mutex;

struct AppState {
    reference_profile: Mutex<Option<EQProfile>>,
    input_profile: Mutex<Option<EQProfile>>,
    match_result: Mutex<Option<MatchResult>>,
}

#[tauri::command]
async fn load_reference_audio(path: String, preset: Option<String>) -> Result<EQProfile, String> {
    // Load audio
    let audio = load_audio_file(&path).map_err(|e| format!("Load error: {}", e))?;

    // Resample policy:
    // - Default: avoid unnecessary upsampling (big speed win for 44.1kHz sources).
    // - "legacy": keep previous behavior (always resample to 48kHz).
    // - "no-resample": never resample.
    let standard_rate = 48000;
    let policy = match preset.as_deref() {
        Some("legacy") => ResamplePolicy::Always,
        Some("no-resample") => ResamplePolicy::Never,
        _ => ResamplePolicy::DownsampleOnly,
    };
    let (samples, analyzed_rate) =
        prepare_audio_for_analysis(audio.samples, audio.sample_rate, standard_rate, policy)
            .map_err(|e| format!("Resample error: {}", e))?;

    // Analyze
    let config = AnalysisConfig::default();
    let spectrum = analyze_spectrum(&samples, analyzed_rate, &config);
    let profile = extract_eq_profile(&spectrum, &config);

    Ok(profile)
}

#[tauri::command]
async fn load_input_audio(path: String, preset: Option<String>) -> Result<EQProfile, String> {
    load_reference_audio(path, preset).await // Same process
}

#[tauri::command]
async fn calculate_eq_match(
    reference: EQProfile,
    input: EQProfile,
    config: MatchConfig,
) -> Result<MatchResult, String> {
    Ok(match_profiles(&reference, &input, &config))
}

#[tauri::command]
async fn export_eq_settings(
    result: MatchResult,
    format: String, // "reaper", "json", "txt"
) -> Result<String, String> {
    match format.as_str() {
        "reaper" => export_as_reaper_preset(&result.correction_profile),
        "json" => {
            serde_json::to_string_pretty(&result.correction_profile).map_err(|e| e.to_string())
        }
        "txt" => export_as_text(&result.correction_profile),
        _ => Err("Unknown format".to_string()),
    }
}

fn export_as_reaper_preset(profile: &EQProfile) -> Result<String, String> {
    let mut output = String::from("<FXCHAIN\n");
    output.push_str("WNDRECT 0 0 0 0\n");
    output.push_str("SHOW 0\n");
    output.push_str("LASTSEL 0\n");
    output.push_str("DOCKED 0\n");
    output.push_str("<VST \"VST: ReaEQ (Cockos)\" ReaEQ.vst.dylib 0 \"\" 1919247729\n");

    // ReaEQ bands
    for (i, band) in profile.bands.iter().enumerate().take(10) {
        let base_param = i * 5;

        // Enable band
        output.push_str(&format!("  {} 1.0\n", base_param));

        // Frequency (normalized 0-1)
        let freq_norm =
            (band.frequency.log2() - 20.0f32.log2()) / (20000.0f32.log2() - 20.0f32.log2());
        output.push_str(&format!("  {} {}\n", base_param + 1, freq_norm));

        // Gain (normalized, ±18dB range)
        let gain_norm = (band.gain_db + 18.0) / 36.0;
        output.push_str(&format!("  {} {}\n", base_param + 2, gain_norm));

        // Q
        output.push_str(&format!("  {} 0.5\n", base_param + 3));

        // Type (Bell)
        output.push_str(&format!("  {} 0.4\n", base_param + 4));
    }

    output.push_str(">\n");
    output.push_str("FLOATPOS 0 0 0 0\n");
    output.push_str("FXID {GUID}\n");
    output.push_str("WAK 0 0\n");
    output.push_str(">\n");

    Ok(output)
}

fn export_as_text(profile: &EQProfile) -> Result<String, String> {
    let mut output = String::from("EQ Settings:\n\n");

    for band in &profile.bands {
        output.push_str(&format!(
            "{:>6} Hz: {:>+6.2} dB (Q: {:.2})\n",
            band.frequency as i32,
            band.gain_db,
            calculate_q_from_bandwidth(band.frequency, band.bandwidth)
        ));
    }

    Ok(output)
}

fn calculate_q_from_bandwidth(center_freq: f32, bandwidth: f32) -> f32 {
    center_freq / bandwidth
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            reference_profile: Mutex::new(None),
            input_profile: Mutex::new(None),
            match_result: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            load_reference_audio,
            load_input_audio,
            calculate_eq_match,
            export_eq_settings,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
