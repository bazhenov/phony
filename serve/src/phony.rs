use std::error::Error;

use crate::sample::Record;
use encoding::all::WINDOWS_1251;
use encoding::{EncoderTrap, Encoding};

use ndarray::{s, Array1, Array2};

pub type CharacterSpan = (usize, usize);
pub type PhonySample = Record<String, Vec<CharacterSpan>>;

pub struct PhonyProblem<'a> {
    pub chars: Vec<u8>,
    pub left_padding: usize,
    pub right_padding: usize,
    pub sample: &'a PhonySample,
}

impl<'a> PhonyProblem<'a> {
    pub const WINDOW: usize = 64;

    pub fn new(sample: &'a PhonySample) -> Result<Self, Box<dyn Error>> {
        if let Some((left_padding, padded_string, right_padding)) =
            Self::pad_string(&sample.sample, Self::WINDOW)
        {
            Ok(PhonyProblem {
                chars: WINDOWS_1251.encode(&padded_string, EncoderTrap::Strict)?,
                left_padding,
                right_padding,
                sample,
            })
        } else {
            Ok(PhonyProblem {
                chars: WINDOWS_1251.encode(&sample.sample, EncoderTrap::Strict)?,
                left_padding: 0,
                right_padding: 0,
                sample,
            })
        }
    }

    pub fn ground_truth(&self) -> Array2<f32> {
        let mut mask1d = Array1::<f32>::zeros(self.chars.len());

        if let Some(spans) = &self.sample.label {
            for span in spans {
                let from = self.left_padding + span.0;
                let to = self.left_padding + span.1;
                mask1d.slice_mut(s![from..to]).fill(1.);
            }
        }

        let ngrams = self.chars.len() - Self::WINDOW + 1;
        let mut mask2d = Array2::<f32>::zeros((ngrams, Self::WINDOW));

        for (i, mut row) in mask2d.genrows_mut().into_iter().enumerate() {
            let from = i;
            let to = i + Self::WINDOW;
            row.assign(&mask1d.slice(s![from..to]));
        }

        mask2d
    }

    pub fn pad_string(string: &str, desired_length: usize) -> Option<(usize, String, usize)> {
        let char_length = string.chars().count();
        if char_length >= desired_length {
            return None;
        }

        let bytes_length = string.len();
        let left_padding = (desired_length - char_length) / 2;
        let right_padding = desired_length - char_length - left_padding;
        let mut padded_string = String::with_capacity(bytes_length + left_padding + right_padding);

        for _ in 0..left_padding {
            padded_string.push(' ');
        }

        padded_string.push_str(string);

        for _ in 0..right_padding {
            padded_string.push(' ');
        }

        Some((left_padding, padded_string, right_padding))
    }
}

pub fn mask(record: &PhonySample, mask: &str) -> PhonySample {
    let mut result = String::new();
    let mut prev_end = 0;
    let mask_length = mask.chars().count();

    if let Some(spans) = &record.label {
        let mut spans_reconstructed = vec![];
        for span in spans {
            let span_start = byte_offset(&record.sample, span.0);
            let span_end = span.1;
            if span_start > prev_end {
                result.push_str(&record.sample[prev_end..span_start]);
            }
            let current_length = result.chars().count();
            result.push_str(mask);
            spans_reconstructed.push((current_length, current_length + mask_length));
            prev_end = byte_offset(&record.sample, span_end);
        }

        if prev_end < record.sample.len() - 1 {
            result.push_str(&record.sample[prev_end..]);
        }
        PhonySample {
            sample: result,
            label: Some(spans_reconstructed),
            prediction: record.prediction.clone(),
        }
    } else {
        PhonySample {
            sample: record.sample.clone(),
            label: record.label.clone(),
            prediction: record.prediction.clone(),
        }
    }
}

fn byte_offset(text: &str, char_no: usize) -> usize {
    // Looking for the nth character in a text
    if let Some((offset, _)) = text.char_indices().nth(char_no) {
        offset

    // if char_no equals to the length of the string return the byte after the last one
    } else if text.chars().count() == char_no {
        text.len()
    } else {
        panic!(
            "unable to find offset for char no {no} in string '{text}'",
            no = char_no,
            text = text
        );
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn should_be_able_to_mask() {
        let record = PhonySample {
            sample: "Текст с маскируемым текстом".into(),
            label: Some(vec![(8, 19), (20, 27)]),
            ..Default::default()
        };

        let masked_record = mask(&record, "<MASK>");
        assert_eq!(masked_record.sample, "Текст с <MASK> <MASK>");
        assert_eq!(masked_record.label, Some(vec![(8, 14), (15, 21)]));
    }
}
