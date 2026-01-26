use crate::audio::profile::FrequencyBand;
use biquad::*;

pub struct ParametricEQ {
    filters: Vec<DirectForm2Transposed<f32>>,
}

impl ParametricEQ {
    pub fn new(sample_rate: f32, bands: &[FrequencyBand]) -> Self {
        let filters = bands
            .iter()
            .map(|band| {
                let coeffs = Coefficients::<f32>::from_params(
                    Type::PeakingEQ(band.gain_db),
                    sample_rate.hz(),
                    band.frequency.hz(),
                    Q_BUTTERWORTH_F32, // Q = 0.707
                )
                .unwrap();

                DirectForm2Transposed::<f32>::new(coeffs)
            })
            .collect();

        Self { filters }
    }

    pub fn process(&mut self, sample: f32) -> f32 {
        let mut output = sample;
        for filter in &mut self.filters {
            output = filter.run(output);
        }
        output
    }

    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process(*sample);
        }
    }
}

// Utility for audio preview with EQ applied
pub fn apply_eq_preview(samples: &[f32], sample_rate: u32, bands: &[FrequencyBand]) -> Vec<f32> {
    let mut eq = ParametricEQ::new(sample_rate as f32, bands);
    let mut output = samples.to_vec();
    eq.process_buffer(&mut output);
    output
}
