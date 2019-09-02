use serde::{Deserialize, Serialize};

/// Record is the one of the key types in a system storing following information:
///
/// * the sample (or example in other terms) of type `X`;
/// * the label system expecting to produce (grount truth in other terms) of type `Y`;
/// * prediction of the system of the same type `Y`.
///
/// Label and prediction are both optional.
#[derive(Serialize, Deserialize)]
pub struct Record<X, Y> {
    pub sample: X,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<Y>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<Y>,
}
