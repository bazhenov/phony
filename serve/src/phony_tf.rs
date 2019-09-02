use crate::phony::CharacterSpan;
use crate::tf_problem::EvaluationMetric;
use std::fmt;

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

#[cfg(test)]
mod tests {

    use super::*;
    use crate::phony::PhonySample;

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
