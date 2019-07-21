mod digits;
mod numerate;
mod odds;

use digits::Digits;
use numerate::numerate;
use std::collections::HashMap;

fn main() {}

/// Базовый тип для описания телефона.
///
/// Состоит из трех чисел:
/// * код страны;
/// * код региона;
/// * номер клиента.
pub type Phone = (u16, u16, u32);

fn phone_format(phone: Phone, format: &str) -> String {
    let mut result = String::with_capacity(format.len());
    let phone_string = format!("{}{}{}", phone.0, phone.1, phone.2);
    let mut i = 0usize;
    for c in format.chars() {
        if c == '#' {
            result.push(phone_string.chars().nth(i).unwrap());
            i += 1;
        } else {
            result.push(c);
        }
    }
    result
}

trait PhoneFormatRule {
    fn format_as(&self, number: u32) -> Option<String>;
}

struct AsDigitPhoneFormat;

impl PhoneFormatRule for AsDigitPhoneFormat {
    fn format_as(&self, number: u32) -> Option<String> {
        Some(format!("{}", number))
    }
}

struct AsTextPhoneFormat;

impl PhoneFormatRule for AsTextPhoneFormat {
    fn format_as(&self, number: u32) -> Option<String> {
        if number <= 999 {
            Some(numerate(number))
        } else {
            None
        }
    }
}

struct PhoneGenerator {
    country_formats: Vec<String>,
    region_formats: Vec<String>,
    number_formats: Vec<String>,
    rules: Vec<Box<PhoneFormatRule>>,
}

impl PhoneGenerator {
    fn new(
        country_formats: Vec<String>,
        region_formats: Vec<String>,
        number_formats: Vec<String>,
    ) -> Self {
        PhoneGenerator {
            country_formats,
            region_formats,
            number_formats,
            rules: vec![],
        }
    }

    fn register_rule(&mut self, rule: Box<PhoneFormatRule>) {
        self.rules.push(rule);
    }

    fn format(&self, phone: Phone) -> Option<String> {
        let mut result = String::new();
        result.push_str(
            &self
                .format_part(phone.0 as u32, &self.country_formats)
                .unwrap(),
        );
        result.push_str(
            &self
                .format_part(phone.1 as u32, &self.region_formats)
                .unwrap(),
        );
        result.push_str(
            &self
                .format_part(phone.2 as u32, &self.number_formats)
                .unwrap(),
        );
        Some(result)
    }

    fn format_part(&self, number: u32, formats: &Vec<String>) -> Option<String> {
        let chosen_format = &formats[0];
        let mut result = String::with_capacity(chosen_format.len());
        let rule = &self.rules[0];

        let mut seen_number = 0u32;
        let mut digits = number.digits();
        for c in chosen_format.chars() {
            // Прерываем сканирование на третьей цифре
            if seen_number >= 1000 {
                result.push_str(&rule.format_as(seen_number / 10).unwrap());
                seen_number = 0;
            }
            match c {
                '#' => {
                    if let Some(digit) = digits.next() {
                        seen_number += digit as u32;
                        seen_number *= 10;
                    } else {
                        return None;
                    }
                }
                _ => {
                    if seen_number > 0 {
                        result.push_str(&rule.format_as(seen_number / 10).unwrap());
                        seen_number = 0;
                    }
                    result.push(c);
                }
            }
        }
        if seen_number > 0 {
            result.push_str(&rule.format_as(seen_number / 10).unwrap());
            seen_number = 0;
        }
        Some(result)
    }
}

trait PostProcessingRule {
    fn transform(&self, phone: &str) -> Option<String>;
}

struct PlusSevenPostProcessingRule;

impl PostProcessingRule for PlusSevenPostProcessingRule {
    fn transform(&self, phone: &str) -> Option<String> {
        if phone.starts_with("+7 ") {
            Some(phone[3..].to_string())
        } else if phone.starts_with("+7") {
            Some(phone[2..].to_string())
        } else {
            None
        }
    }
}

struct GlyphPostProcessingRule {
    transformations: HashMap<char, char>,
}

impl GlyphPostProcessingRule {
    fn new() -> Self {
        let mut transformations: HashMap<char, char> = HashMap::new();

        transformations.insert('0', 'O');
        transformations.insert('1', 'l');
        transformations.insert('4', 'Ч');

        GlyphPostProcessingRule { transformations }
    }
}

impl PostProcessingRule for GlyphPostProcessingRule {
    fn transform(&self, phone: &str) -> Option<String> {
        Some(
            phone
                .chars()
                .map(|c| *self.transformations.get(&c).unwrap_or(&c))
                .collect(),
        )
    }
}

mod tests {

    use super::*;

    #[test]
    fn test_number_format() {
        assert_eq!(
            phone_format((7, 999, 3053315), "+# (###) ###-##-##"),
            "+7 (999) 305-33-15"
        );
    }

    #[test]
    fn generator_base_scenario() {
        let mut g = PhoneGenerator::new(
            vec!["+#".to_owned()],
            vec![" (###) ".to_owned()],
            vec!["###-##-##".to_owned()],
        );
        g.register_rule(Box::new(AsDigitPhoneFormat));

        assert_eq!(
            g.format((7, 999, 3056617)),
            Some("+7 (999) 305-66-17".to_owned())
        );
    }

    #[test]
    fn generator_text_scenario() {
        let mut g = PhoneGenerator::new(
            vec!["+#".to_owned()],
            vec![" (###) ".to_owned()],
            vec!["###-##-##".to_owned()],
        );
        g.register_rule(Box::new(AsTextPhoneFormat));

        assert_eq!(
            g.format((7, 999, 3056617)),
            Some(
                "+семь (девятьсот девяносто девять) триста пять-шестьдесят шесть-семнадцать"
                    .to_owned()
            )
        );
    }

    #[test]
    fn test_as_digit_phone_format() {
        let rule = AsDigitPhoneFormat;

        assert_eq!(rule.format_as(31), Some("31".to_string()));
        assert_eq!(rule.format_as(4), Some("4".to_string()));
    }

    #[test]
    fn test_as_text_phone_format() {
        let rule = AsTextPhoneFormat;

        assert_eq!(
            rule.format_as(91),
            Some("девяносто один".to_string())
        );
        assert_eq!(rule.format_as(6), Some("шесть".to_string()));
        assert_eq!(rule.format_as(1345), None);
    }

    #[test]
    fn test_plus_seven_post_processing_rule() {
        let rule = PlusSevenPostProcessingRule;

        assert_eq!(rule.transform("+7914"), Some("914".to_owned()));
        assert_eq!(rule.transform("+7 914"), Some("914".to_owned()));
    }

    #[test]
    fn test_glyph_post_processing_rule() {
        let rule = GlyphPostProcessingRule::new();

        assert_eq!(rule.transform("+79041"), Some("+79OЧl".to_owned()));
    }
}
