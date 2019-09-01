use serde::{Deserialize, Serialize};

type CharacterSpan = (usize, usize);

/// Запись (кортеж) используемая для хранения информации о примере, а также (опционально) пометку примера
/// и ответ системы.
/// X – тип примера
/// Y – тип ответа системы
#[derive(Serialize, Deserialize)]
pub struct Record<X, Y> {
    pub sample: X,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<Y>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Y>,
}

pub type PhonySample = Record<String, Vec<CharacterSpan>>;

#[cfg(test)]
mod tests {

    use super::*;

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
