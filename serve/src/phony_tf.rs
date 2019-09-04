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
    spurious: usize,
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

impl<'a> TensorflowProblem for PhonyProblem<'a> {
    type TensorInputType = f32;
    type TensorOutputType = f32;
    type Input = PhonySample;
    type Output = Vec<CharacterSpan>;
    const GRAPH_INPUT_NAME: &'static str = "input";
    const GRAPH_OUTPUT_NAME: &'static str = "flatten/Reshape";

    fn features(&self) -> Array2<Self::TensorInputType> {
        let ngrams = self.chars.windows(Self::WINDOW).collect::<Vec<_>>();

        let mut result = Array2::zeros((ngrams.len(), Self::WINDOW));

        for (i, ngram) in ngrams.iter().enumerate() {
            for (j, c) in ngram.iter().enumerate() {
                result[[i, j]] = f32::from(*c);
            }
        }

        result
    }

    fn output(&self, tensors: Array2<Self::TensorOutputType>) -> Self::Output {
        let mut mask = vec![Accumulator(0, 0); self.chars.len()];
        let character_length = self.chars.len() - self.left_padding - self.right_padding;

        for i in 0..tensors.rows() {
            for j in 0..tensors.cols() {
                mask[i + j].register(tensors[[i, j]] > 0.5);
            }
        }
        mask.iter()
            .map(|a| a.ratio() > 0.5)
            // отрезаем от маски "хвостики" порожденные padding'ом строки
            .skip(self.left_padding)
            .take(character_length)
            .spans(|c| c)
            .map(|r| (r.start, r.end))
            .collect()
    }

    fn ground_truth(&self) -> Array2<f32> {
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

/// Простой счетчик – регистририует количество ложных/положительных срабатываный. Метод [`register`](#method.register)
#[derive(Copy, Clone)]
struct Accumulator(u16, u16);

impl Accumulator {
    /// Регистрирует срабатывание: ложное или положительное в зависимости от значения аргумента `hit`.
    fn register(&mut self, hit: bool) {
        if hit {
            self.0 += 1;
        }
        self.1 += 1;
    }

    /// доля положительных вызовов по отношению к общему количеству
    fn ratio(self) -> f32 {
        f32::from(self.0) / f32::from(self.1)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::phony::PhonySample;

    #[test]
    fn pad_string() {
        assert_eq!(
            PhonyProblem::pad_string("123", 5),
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
        let example = PhonySample {
            sample: String::from("text"),
            label: Some(vec![(0, 1), (2, 4)]),
            prediction: None,
        };
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
    fn shoud_be_able_to_stack() {
        use ndarray::arr2;

        let a = arr2(&[[1, 2], [3, 4]]);
        let b = arr2(&[[5, 6], [7, 8]]);

        let r = ndarray::stack(ndarray::Axis(0), &[a.view(), b.view()]);
        println!("{:?}", r);
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
    fn should_be_able_to_read_sample_only() {
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
}
