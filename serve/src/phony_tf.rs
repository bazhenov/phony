use crate::phony::{CharacterSpan, PhonySample};
use crate::tf_problem::{EvaluationMetric, TensorflowProblem};
use encoding::all::WINDOWS_1251;
use encoding::{EncoderTrap, Encoding};
use ndarray::{s, Array1, Array2};

use std::error::Error;
use std::fmt;
use std::ops::Range;

/// Implements modified metric for evaluating NER systems proposed on Message Understanding Conference.
///
/// MUC metric is designed for evaluating two set of annotations (character spans in text): ground truth and
/// prediction and categorizes all the matches in a following way:
///
/// * `STRICT` – prediction matches ground truth label exactly;
/// * `PARTIAL` – there is some intersection between ground truth label and prediction label, but
/// span bounds are not the same;
/// * `MISSING` – ground truth label has no pair in prediction;
/// * `SPURIOUS` – prediction label has no pair in ground truth label.
///
/// `MISSING` and `SPURIOUS` categories are frequently occuring together. Suppose following example:
///
/// * ground truth – `[15, 28]`;
/// * prediction – `[12, 14]`.
///
/// This set of labels contains 1 `MISSING` annotation (`[15, 28]`) and 1 `SPURIOUS` (`[12, 14]`).
///
/// See: [MUC-5 EVALUATION METRICS](https://www.aclweb.org/anthology/M93-1007),
/// [Named-Entity evaluation metrics based on entity-level](http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/)
#[derive(Default, PartialEq, Debug)]
pub struct MucMetric {
    examples: usize,
    strict: usize,
    partial: usize,
    missing: usize,
    pub spurious: usize,
}

impl MucMetric {
    pub fn new() -> Self {
        MucMetric {
            ..Default::default()
        }
    }
}

impl EvaluationMetric<[CharacterSpan]> for MucMetric {
    fn update(&mut self, truth: &[CharacterSpan], prediction: &[CharacterSpan]) {
        self.examples += 1;

        for span in prediction {
            match truth.iter().find(|b| does_spans_intersects(*span, **b)) {
                None => self.spurious += 1,
                Some(pair) if pair == span => self.strict += 1,
                Some(_) => self.partial += 1,
            }
        }

        for span in truth {
            let matches = prediction
                .iter()
                .find(|b| does_spans_intersects(*span, **b));
            if matches.is_none() {
                self.missing += 1;
            }
        }
    }
}

impl fmt::Display for MucMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total = (self.strict + self.partial + self.missing + self.spurious) as f64;
        writeln!(
            f,
            "{:10} : {:10} ({:6.2}%)",
            "Strict",
            self.strict,
            self.strict as f64 / total * 100.0
        )?;
        writeln!(
            f,
            "{:10} : {:10} ({:6.2}%)",
            "Partial",
            self.partial,
            self.partial as f64 / total * 100.0
        )?;
        writeln!(
            f,
            "{:10} : {:10} ({:6.2}%)",
            "Missing",
            self.missing,
            self.missing as f64 / total * 100.0
        )?;
        writeln!(
            f,
            "{:10} : {:10} ({:6.2}%)",
            "Spurious",
            self.spurious,
            self.spurious as f64 / total * 100.0
        )?;
        write!(f, "{:10} : {:10}", "Examples", self.examples)
    }
}

pub fn does_spans_intersects(a: CharacterSpan, b: CharacterSpan) -> bool {
    (b.0 >= a.0 && b.0 < a.1) || (a.0 >= b.0 && a.0 < b.1)
}

#[derive(Debug)]
pub struct PhonyProblem<'a> {
    chars: Vec<u8>,
    left_padding: usize,
    right_padding: usize,
    sample: &'a PhonySample,
    window: usize,
}

impl<'a> PhonyProblem<'a> {
    const WINDOW: usize = 64;

    pub fn new(sample: &'a PhonySample) -> Result<Self, Box<dyn Error>> {
        Self::new_with_window(sample, Self::WINDOW)
    }

    fn new_with_window(sample: &'a PhonySample, window: usize) -> Result<Self, Box<dyn Error>> {
        assert!(window > 0, "window should be positive");

        let length = sample.sample.chars().count();
        let padding = (window - (length % window)) % window;
        let desired_length = length + padding;

        // At this point string desired length should be multiple of window size
        assert!(desired_length % window == 0, "padding algorithm failed");

        if let Some((left_padding, padded_string, right_padding)) =
            pad_string(&sample.sample, desired_length)
        {
            Ok(PhonyProblem {
                chars: WINDOWS_1251.encode(&padded_string, EncoderTrap::Strict)?,
                left_padding,
                right_padding,
                sample,
                window,
            })
        } else {
            Ok(PhonyProblem {
                chars: WINDOWS_1251.encode(&sample.sample, EncoderTrap::Strict)?,
                left_padding: 0,
                right_padding: 0,
                sample,
                window,
            })
        }
    }
}

impl<'a> TensorflowProblem for PhonyProblem<'a> {
    type TensorInputType = f32;
    type TensorOutputType = f32;
    type InputDim = ndarray::Ix2;
    type OutputDim = ndarray::Ix2;
    type Input = PhonySample;
    type Output = Vec<CharacterSpan>;
    const GRAPH_INPUT_NAME: &'static str = "serving_default_input";
    const GRAPH_OUTPUT_NAME: &'static str = "StatefulPartitionedCall";

    fn features(&self) -> Array2<Self::TensorInputType> {
        let partitions = self.chars.chunks_exact(self.window).collect::<Vec<_>>();

        let mut result = Array2::zeros((partitions.len(), self.window));

        for (i, part) in partitions.iter().enumerate() {
            for (j, c) in part.iter().enumerate() {
                result[[i, j]] = f32::from(*c);
            }
        }

        result
    }

    fn output(&self, tensors: Array2<Self::TensorOutputType>) -> Self::Output {
        let mut mask = vec![0.; self.chars.len()];
        let character_length = self.chars.len() - self.left_padding - self.right_padding;

        for i in 0..tensors.rows() {
            for j in 0..tensors.cols() {
                mask[(i * self.window) + j] = tensors[[i, j]];
            }
        }
        mask.iter()
            // отрезаем от маски "хвостики" порожденные padding'ом строки
            .skip(self.left_padding)
            .take(character_length)
            // получаем непрерывные span'ы с высоким значением вероятности
            .spans(|a| *a > 0.5)
            .map(|r| (r.start, r.end))
            .collect()
    }

    fn ground_truth(&self) -> Array2<f32> {
        let length = self.chars.len();
        assert!(
            length % self.window == 0,
            "String length should be miltiple of window"
        );
        let mut mask1d = Array1::<f32>::zeros(length);

        if let Some(spans) = &self.sample.label {
            for span in spans {
                let from = self.left_padding + span.0;
                let to = self.left_padding + span.1;
                mask1d.slice_mut(s![from..to]).fill(1.);
            }
        }

        let partitions = length / self.window;
        let mut mask2d = Array2::<f32>::zeros((partitions, self.window));

        for (i, mut row) in mask2d.genrows_mut().into_iter().enumerate() {
            let from = i * self.window;
            let to = from + self.window;
            row.assign(&mask1d.slice(s![from..to]));
        }

        mask2d
    }
}

struct Spans<'a, I, F> {
    iterator: &'a mut I,
    position: usize,
    predicate: F,
}

trait SpanExtension: Iterator + Sized {
    fn spans<F>(&mut self, f: F) -> Spans<'_, Self, F>
    where
        F: Fn(Self::Item) -> bool,
    {
        Spans {
            iterator: self,
            position: 0,
            predicate: f,
        }
    }
}

fn pad_string(string: &str, desired_length: usize) -> Option<(usize, String, usize)> {
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

impl<T: Iterator> SpanExtension for T {}

impl<I, F> Iterator for Spans<'_, I, F>
where
    I: Iterator,
    F: Fn(I::Item) -> bool,
{
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.position += 1;
            match self.iterator.next().map(&self.predicate) {
                Some(true) => break,
                None => return None,
                Some(false) => {}
            }
        }
        let from = self.position - 1;
        loop {
            self.position += 1;
            match self.iterator.next().map(&self.predicate) {
                Some(false) => return Some(from..self.position - 1),
                None => return Some(from..self.position - 1),
                Some(true) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::phony::PhonySample;
    use ndarray::arr2;

    #[test]
    fn check_pad_string() {
        assert_eq!(
            pad_string("123", 5),
            Some((1usize, String::from(" 123 "), 1usize))
        );
    }

    #[test]
    fn groups() {
        let v = vec![0, 0, 1, 0, 0, 1, 1, 1, 0];

        let spans = v.iter().spans(|i| *i > 0).collect::<Vec<_>>();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], 2..3);
        assert_eq!(spans[1], 5..8);

        let spans = v.iter().spans(|i| *i == 0).collect::<Vec<_>>();
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0], 0..2);
        assert_eq!(spans[1], 3..5);
        assert_eq!(spans[2], 8..9);
    }

    #[test]
    fn should_be_able_to_reconstruct_ground_truth_labels() {
        let example =
            PhonySample::from_sample_and_label(String::from("text"), vec![(0, 1), (2, 4)]);
        let p = PhonyProblem::new(&example).unwrap();

        let truth = p.ground_truth();
        assert_eq!(p.left_padding, 30);
        assert_eq!(p.right_padding, 30);
        // Индексы в маске смещены из за padding'а. Это необходимо учитывать при изменений ширины окна
        let mut expected = Array2::zeros((1, 64));
        expected[[0, p.left_padding]] = 1.;
        expected[[0, p.left_padding + 2]] = 1.;
        expected[[0, p.left_padding + 3]] = 1.;
        assert_eq!(truth, expected);
        //assert_eq!(p.output(truth), vec![(0, 1), (2, 4)]);
    }

    #[test]
    fn muc_strict_match() {
        let mut muc = MucMetric::new();
        muc.update(&[(0, 1)], &[(0, 1)]);

        let expected = MucMetric {
            strict: 1,
            examples: 1,
            ..Default::default()
        };
        assert_eq!(muc, expected);
    }

    #[test]
    fn muc_partial_match() {
        let mut muc = MucMetric::new();
        muc.update(&[(0, 10)], &[(5, 8)]);

        let expected = MucMetric {
            partial: 1,
            examples: 1,
            ..Default::default()
        };
        assert_eq!(muc, expected);
    }

    #[test]
    fn muc_spurious_and_missing_match() {
        let mut muc = MucMetric::new();
        muc.update(&[(0, 10)], &[(10, 20)]);

        let expected = MucMetric {
            spurious: 1,
            missing: 1,
            examples: 1,
            ..Default::default()
        };
        assert_eq!(muc, expected);
    }

    #[test]
    fn muc_complex_match() {
        let mut muc = MucMetric::new();
        muc.update(&[(0, 1), (5, 10), (12, 15)], &[(5, 10), (12, 16)]);

        let expected = MucMetric {
            strict: 1,
            partial: 1,
            spurious: 0,
            missing: 1,
            examples: 1,
        };
        assert_eq!(muc, expected);
    }

    #[test]
    fn spans_can_be_tested_for_intersection() {
        assert_eq!(does_spans_intersects((0, 1), (1, 2)), false);
        assert_eq!(does_spans_intersects((1, 2), (0, 1)), false);

        assert_eq!(does_spans_intersects((0, 2), (1, 2)), true);
        assert_eq!(does_spans_intersects((1, 2), (0, 2)), true);

        assert_eq!(does_spans_intersects((1, 10), (2, 3)), true);
        assert_eq!(does_spans_intersects((2, 3), (1, 10)), true);
    }

    #[test]
    fn should_be_able_to_read_sample_without_lables_and_prediction() {
        let json = r#"{"sample":"Текст"}"#;
        let record = serde_json::from_str::<PhonySample>(json).expect("Unable to read sample");

        assert_eq!(record.sample, "Текст");
        assert_eq!(record.label, None);
        assert_eq!(record.prediction, None);
    }

    #[test]
    fn should_be_able_to_read_sample_with_labels() {
        let json = r#"{
            "sample": "Первый: 1, второй: 2",
            "label": [[8, 9], [19, 20]]}"#;
        let record = serde_json::from_str::<PhonySample>(json).expect("Unable to read sample");

        assert_eq!(record.sample, "Первый: 1, второй: 2");
        assert_eq!(record.label, Some(vec![(8, 9), (19, 20)]));
        assert_eq!(record.prediction, None);
    }

    #[test]
    fn should_be_able_to_read_sample_with_labels_and_prediction() {
        let json = r#"{
            "sample":"Первый: 1, второй: 2",
            "label":[[8, 9], [19, 20]],
            "prediction": [[1,2], [3, 5]]}"#;
        let record = serde_json::from_str::<PhonySample>(json).expect("Unable to read sample");

        assert_eq!(record.sample, "Первый: 1, второй: 2");
        assert_eq!(record.label, Some(vec![(8, 9), (19, 20)]));
        assert_eq!(record.prediction, Some(vec![(1, 2), (3, 5)]));
    }

    #[test]
    fn check_small_string_are_padded_correctly() {
        // Check paddings with smaller values to make test easier
        let sample = PhonySample::from_sample("foo".to_owned());
        let problem = PhonyProblem::new_with_window(&sample, 4).unwrap();

        // All paddings (left and right) should be 1 = 4 (window) - 3 (string length)
        let all_paddings = problem.right_padding + problem.left_padding;
        assert_eq!(all_paddings, 1);
    }

    #[test]
    fn check_long_string_are_padded_correctly() {
        // Check paddings with smaller values to make test easier
        let sample = PhonySample::from_sample("some long string".to_owned());
        let problem = PhonyProblem::new_with_window(&sample, 5).unwrap();

        // All paddings (left and right) should be 4 = window - (string length % window)
        // 4 = 5 - (16 % 5)
        let all_paddings = problem.right_padding + problem.left_padding;
        assert_eq!(all_paddings, 4);

        let features = problem.features();
        // 4 parts 5 characters each
        assert_eq!(features.dim(), (4, 5));
    }

    #[test]
    fn check_features_and_ground_labels_are_the_same_in_size() {
        let sample =
            PhonySample::from_sample_and_label("some long string".to_owned(), vec![(0, 4)]);
        let problem = PhonyProblem::new_with_window(&sample, 5).unwrap();

        let features = problem.features();
        let ground_truth = problem.ground_truth();
        assert_eq!(features.dim().0, ground_truth.dim().0);
    }

    #[test]
    fn output_should_return_correctly_aligned_labels() {
        // special string to ensure zero padding
        let sample = PhonySample::from_sample("1234567890".to_owned());
        let problem = PhonyProblem::new_with_window(&sample, 5).unwrap();

        let out = problem.output(arr2(&[[0., 1., 1., 0., 1.], [1., 1., 0., 0., 0.]]));
        assert_eq!(out, vec![(1, 3), (4, 7)]);
    }
}
