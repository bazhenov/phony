use serde::{Deserialize, Serialize};

type CharacterSpan = (usize, usize);

#[derive(Serialize, Deserialize)]
pub struct PhonySample {
    pub text: String,
    pub spans: Vec<CharacterSpan>,
}

impl PhonySample {
    pub fn create<T: AsRef<str>>(
        message: &str,
        values: &mut impl Iterator<Item = T>,
    ) -> Option<PhonySample> {
        let mut text = String::with_capacity(message.len());
        let mut offset_chars = 0;
        let mut span = message;
        let pattern = "<PHONE>";
        let mut spans = vec![];

        while let Some(match_offset) = span.find(pattern) {
            offset_chars += span[..match_offset].chars().count();
            text.push_str(&span[..match_offset]);
            if let Some(n) = values.next() {
                let replacement = n.as_ref();
                let replacement_chars_length = replacement.chars().count();
                spans.push((offset_chars, offset_chars + replacement_chars_length));
                text.push_str(replacement);

                offset_chars += replacement_chars_length;
            } else {
                return None;
            }

            span = &span[match_offset + pattern.len()..];
        }

        if !span.is_empty() {
            text.push_str(span);
        }

        Some(PhonySample { text, spans })
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_create_augmented_sample() {
        let text = "Первый: <PHONE>, второй: <PHONE>";
        let list = vec!["1", "2"];
        let sample = PhonySample::create(text, &mut list.iter()).unwrap();

        assert_eq!(sample.text, "Первый: 1, второй: 2");
        assert_eq!(sample.spans, vec![(8, 9), (19, 20)]);

        let json = serde_json::to_string(&sample).unwrap();
        let expected = r#"{"text":"Первый: 1, второй: 2","spans":[[8,9],[19,20]]}"#;
        assert_eq!(json, expected)
    }

    #[test]
    fn test_read_augmented_sample() {
        let record = r#"{"text":"Первый: 1, второй: 2","spans":[[8,9],[19,20]]}"#;
        let sample = serde_json::from_str::<PhonySample>(record).expect("Unable to read sample");

        assert_eq!(sample.text, "Первый: 1, второй: 2");
        assert_eq!(sample.spans, vec![(8, 9), (19, 20)]);
    }
}
