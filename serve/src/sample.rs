use serde::{Deserialize, Serialize};

type CharacterSpan = (usize, usize);

#[derive(Serialize, Deserialize)]
pub struct PhonySample {
    pub text: String,
    pub spans: Vec<CharacterSpan>,
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_read_augmented_sample() {
        let record = r#"{"text":"Первый: 1, второй: 2","spans":[[8,9],[19,20]]}"#;
        let sample = serde_json::from_str::<PhonySample>(record).expect("Unable to read sample");

        assert_eq!(sample.text, "Первый: 1, второй: 2");
        assert_eq!(sample.spans, vec![(8, 9), (19, 20)]);
    }
}
