use crate::sample::Record;

pub type CharacterSpan = (usize, usize);
pub type PhonySample = Record<String, Vec<CharacterSpan>>;
