use crate::sample::Record;

pub type CharacterSpan = (usize, usize);
pub type PhonySample = Record<String, Vec<CharacterSpan>>;

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
