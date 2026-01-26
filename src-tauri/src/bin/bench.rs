use eq_matcher::audio::{
    analyzer::{analyze_spectrum, AnalysisConfig},
    loader::{load_audio_file, prepare_audio_for_analysis, ResamplePolicy},
    profile::{extract_eq_profile, EQProfile},
};
use std::env;
use std::time::Instant;

fn parse_preset(args: &[String]) -> ResamplePolicy {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--preset" {
            return match it.next().map(|s| s.as_str()) {
                Some("legacy") => ResamplePolicy::Always,
                Some("no-resample") => ResamplePolicy::Never,
                _ => ResamplePolicy::DownsampleOnly,
            };
        }
    }
    ResamplePolicy::DownsampleOnly
}

fn run_once(path: &str, policy: ResamplePolicy) -> Result<(EQProfile, Metrics), String> {
    let total_start = Instant::now();

    let t0 = Instant::now();
    let audio = load_audio_file(path).map_err(|e| e.to_string())?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let standard_rate = 48_000u32;
    let t1 = Instant::now();
    let (samples, analyzed_rate) =
        prepare_audio_for_analysis(audio.samples, audio.sample_rate, standard_rate, policy)
            .map_err(|e| e.to_string())?;
    let resample_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let config = AnalysisConfig::default();

    let t2 = Instant::now();
    let spectrum = analyze_spectrum(&samples, analyzed_rate, &config);
    let analyze_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = Instant::now();
    let profile = extract_eq_profile(&spectrum, &config);
    let profile_ms = t3.elapsed().as_secs_f64() * 1000.0;

    let total_s = total_start.elapsed().as_secs_f64();
    let rtf = audio.duration_secs / total_s;

    Ok((
        profile,
        Metrics {
            duration_s: audio.duration_secs,
            sample_rate_in: audio.sample_rate,
            sample_rate_analyzed: analyzed_rate,
            load_ms,
            resample_ms,
            analyze_ms,
            profile_ms,
            total_s,
            realtime_factor: rtf,
        },
    ))
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    duration_s: f64,
    sample_rate_in: u32,
    sample_rate_analyzed: u32,
    load_ms: f64,
    resample_ms: f64,
    analyze_ms: f64,
    profile_ms: f64,
    total_s: f64,
    realtime_factor: f64,
}

fn main() -> Result<(), String> {
    let raw_args: Vec<String> = env::args().skip(1).collect();
    let path = raw_args
        .iter()
        .find(|a| !a.starts_with("--"))
        .ok_or_else(|| {
            "Usage: cargo run --release --bin bench -- <audio_file> [--preset legacy|smart|no-resample] [--compare-legacy]".to_string()
        })?
        .to_string();

    let policy = parse_preset(&raw_args);
    let compare_legacy = raw_args.iter().any(|a| a == "--compare-legacy");

    let (profile, metrics) = run_once(&path, policy)?;

    println!("file: {path}");
    println!("duration_s: {:.3}", metrics.duration_s);
    println!("sample_rate_in: {}", metrics.sample_rate_in);
    println!("sample_rate_analyzed: {}", metrics.sample_rate_analyzed);
    println!("load_ms: {:.2}", metrics.load_ms);
    println!("resample_ms: {:.2}", metrics.resample_ms);
    println!("analyze_ms: {:.2}", metrics.analyze_ms);
    println!("profile_ms: {:.2}", metrics.profile_ms);
    println!("total_s: {:.3}", metrics.total_s);
    println!("realtime_factor: {:.2}x", metrics.realtime_factor);

    if compare_legacy {
        let (legacy_profile, _) = run_once(&path, ResamplePolicy::Always)?;
        let max_band_diff = legacy_profile
            .bands
            .iter()
            .zip(&profile.bands)
            .map(|(a, b)| (a.gain_db - b.gain_db).abs())
            .fold(0.0f32, f32::max);
        println!("max_band_diff_vs_legacy_db: {:.3}", max_band_diff);
    }

    Ok(())
}
