use crate::sample::{does_spans_intersects, CharacterSpan};
use std::fmt;

#[derive(Default, PartialEq, Debug)]
pub struct MucMetric {
    /// Total number of given examples
    examples: usize,

    /// correctly annotated
    strict: usize,

    /// bounds of label and predictions are intersecting but not the same
    partial: usize,

    /// number of labels missing from the prediction
    missing: usize,

    /// spurious labels not present in the learning set
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
    fn feed(&mut self, truth: &[CharacterSpan], prediction: &[CharacterSpan]) {
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

pub trait EvaluationMetric<T: ?Sized> {
    fn feed(&mut self, truth: &T, prediction: &T);
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn muc_strict_match() {
        let mut muc = MucMetric::new();
        muc.feed(&[(0, 1)], &[(0, 1)]);

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
        muc.feed(&[(0, 10)], &[(5, 8)]);

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
        muc.feed(&[(0, 10)], &[(10, 20)]);

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
        muc.feed(&[(0, 1), (5, 10), (12, 15)], &[(5, 10), (12, 16)]);

        let expected = MucMetric {
            strict: 1,
            partial: 1,
            spurious: 0,
            missing: 1,
            examples: 1,
        };
        assert_eq!(muc, expected);
    }
}
