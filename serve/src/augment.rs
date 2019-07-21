mod digits;
mod numerate;
mod odds;

use digits::Digits;
use numerate::numerate;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
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
    fn format_phone(&self, number: u32) -> Option<String>;
}

struct AsDigitPhoneFormat;

impl PhoneFormatRule for AsDigitPhoneFormat {
    fn format_phone(&self, number: u32) -> Option<String> {
        Some(format!("{}", number))
    }
}

struct AsTextPhoneFormat;

impl PhoneFormatRule for AsTextPhoneFormat {
    fn format_phone(&self, number: u32) -> Option<String> {
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
    postprocessors: Vec<Box<PostProcessingRule>>,
}

impl PhoneGenerator {
    fn new(country_format: &str, region_format: &str, number_format: &str,
    ) -> Self {
        let mut s = PhoneGenerator {
            country_formats: vec![],
            region_formats: vec![],
            number_formats: vec![],
            rules: vec![],
            postprocessors: vec![]
        };
        s.register_country_formats(vec![country_format]);
        s.register_region_formats(vec![region_format]);
        s.register_number_formats(vec![number_format]);
        s
    }

    fn generate_random(&self) -> Option<String> {
        let mut rng = thread_rng();
        let country = 7;
        let region = rng.gen_range(900, 999);
        let number = rng.gen_range(1000000, 9999999);
        self.format((country, region, number))
            .map(|result| self.postprocess(result))
    }

    fn postprocess(&self, phone: String) -> String {
        if self.postprocessors.is_empty() {
            return phone;
        }
        let mut rng = thread_rng();
        let chosen = self.postprocessors.choose(&mut rng).unwrap();
        chosen.transform(&phone).unwrap_or(phone)
    }

    fn register_rule(&mut self, rule: Box<PhoneFormatRule>) {
        self.rules.push(rule);
    }

    fn register_postprocessor(&mut self, postprocessor: Box<PostProcessingRule>) {
        self.postprocessors.push(postprocessor);
    }

    fn register_country_formats<'a>(&mut self, formats: impl IntoIterator<Item=&'a str>) {
        for format in formats {
            self.country_formats.push(format.to_owned());
        }
    }

    fn register_region_formats<'a>(&mut self, formats: impl IntoIterator<Item=&'a str>) {
        for format in formats {
            self.region_formats.push(format.to_owned());
        }
    }

    fn register_number_formats<'a>(&mut self, formats: impl IntoIterator<Item=&'a str>) {
        for format in formats {
            self.number_formats.push(format.to_owned());
        }
    }

    fn format(&self, phone: Phone) -> Option<String> {
        let mut result = String::new();
        let phone_parts = [
            (phone.0 as u32, &self.country_formats),
            (phone.1 as u32, &self.region_formats),
            (phone.2, &self.number_formats),
        ];
        for (part, formats) in &phone_parts {
            let formatted = &self.format_part(*part, formats).unwrap();
            result.push_str(formatted);
        }
        Some(result)
    }

    fn format_part(&self, number: u32, formats: &Vec<String>) -> Option<String> {
        let mut rng = thread_rng();
        let chosen_format = formats.choose(&mut rng).unwrap();
        let chosen_rule = self.rules.choose(&mut rng).unwrap();
        let mut result = String::with_capacity(chosen_format.len());

        let mut seen_number = 0u32;
        let mut digits = number.digits();
        for c in chosen_format.chars() {
            // Прерываем сканирование на третьей цифре
            if seen_number >= 1000 {
                result.push_str(&chosen_rule.format_phone(seen_number / 10).unwrap());
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
                        result.push_str(&chosen_rule.format_phone(seen_number / 10).unwrap());
                        seen_number = 0;
                    }
                    result.push(c);
                }
            }
        }
        if seen_number > 0 {
            result.push_str(&chosen_rule.format_phone(seen_number / 10).unwrap());
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
        let mut g = PhoneGenerator::new("+#", " (###) ", "###-##-##");
        g.register_rule(Box::new(AsDigitPhoneFormat));

        assert_eq!(
            g.format((7, 999, 3056617)),
            Some("+7 (999) 305-66-17".to_owned())
        );
    }

    #[test]
    fn generator_text_scenario() {
        let mut g = PhoneGenerator::new("+#", " (###) ", "###-##-##");
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

        assert_eq!(rule.format_phone(31), Some("31".to_string()));
        assert_eq!(rule.format_phone(4), Some("4".to_string()));
    }

    #[test]
    fn test_as_text_phone_format() {
        let rule = AsTextPhoneFormat;

        assert_eq!(
            rule.format_phone(91),
            Some("девяносто один".to_string())
        );
        assert_eq!(rule.format_phone(6), Some("шесть".to_string()));
        assert_eq!(rule.format_phone(1345), None);
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

    #[test]
    fn test_generate_phone() {
        let mut g = PhoneGenerator::new("+#", " (###) ", "###-##-##");
        g.register_country_formats(vec!["#"]);
        g.register_region_formats(vec!["###", " ### ", "-###-"]);
        g.register_number_formats(vec!["#######"]);

        g.register_rule(Box::new(AsTextPhoneFormat));
        g.register_rule(Box::new(AsDigitPhoneFormat));

        g.register_postprocessor(Box::new(GlyphPostProcessingRule::new()));
        g.register_postprocessor(Box::new(PlusSevenPostProcessingRule));

        println!("{}", g.generate_random().unwrap());
    }
}
