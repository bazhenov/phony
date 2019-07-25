#[macro_use]
extern crate clap;

mod digits;
mod numerate;

use clap::App;
use digits::Digits;
use numerate::numerate;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use serde::Serialize;
use std::collections::HashMap;
use std::io::{stdin, stdout, BufRead};

fn main() {
    let app = App::new("phone-generate")
        .author("Denis Bazhenov <dotsid@gmail.com>")
        .version("1.0.0")
        .about("CLI utility for synthetic augmentation of phone numbers")
        .arg_from_usage(
            "[random] -r, --random 'Generate random phone numbers'",
        )
        .arg_from_usage("[stream] -s, --stream 'Generating numbers for a stream of input examples'")
        .arg_from_usage("[count] -c, --count=[COUNT] 'Generate given number of examples'");

    let matches = app.get_matches();
    let stream_mode = matches.is_present("stream");
    let random_mode = matches.is_present("random");

    let count = value_t!(matches, "count", u32).unwrap_or(1);

    let mut generator = prepare_generator();

    if random_mode {
        for _ in 0..count {
            println!("{}", generator.generate_random());
        }
    } else if stream_mode {
        for line in stdin().lock().lines() {
            let line = line.expect("Unable to read");
            for _ in 0..count {
                let sample = AugmentedSample::create(&line, &mut generator);
                serde_json::to_writer(stdout(), &sample).expect("Unable to write json");
                println!();
            }
        }
    }
}

fn prepare_generator() -> PhoneGenerator {
    let mut generator = PhoneGenerator::new("+#", "###", "#######");
    generator.register_country_formats(vec!["#"]);
    generator.register_region_formats(vec![" (###) ", "-###-"]);
    generator.register_number_formats(vec!["###-##-##", "### ## ##"]);

    generator.register_rule(Box::new(AsDigitPhoneFormat::new()));
    generator.register_rule(Box::new(AsTextPhoneFormat::new()));

    generator.register_postprocessor(Box::new(GlyphPostProcessingRule::new()));
    generator.register_postprocessor(Box::new(PlusSevenPostProcessingRule::new()));

    generator
}

/// Базовый тип для описания телефона.
///
/// Состоит из трех чисел:
/// * код страны;
/// * код региона;
/// * номер клиента.
pub type Phone = (u16, u16, u32);

trait PhoneFormatRule {
    fn format_phone(&self, number: u32) -> Option<String>;
}

struct AsDigitPhoneFormat;

impl AsDigitPhoneFormat {
    fn new() -> Self {
        AsDigitPhoneFormat
    }
}

impl PhoneFormatRule for AsDigitPhoneFormat {
    fn format_phone(&self, number: u32) -> Option<String> {
        Some(format!("{}", number))
    }
}

struct AsTextPhoneFormat;

impl AsTextPhoneFormat {
    fn new() -> Self {
        AsTextPhoneFormat
    }
}

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

impl Iterator for PhoneGenerator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.generate_random())
    }
}

impl PhoneGenerator {
    fn new(country_format: &str, region_format: &str, number_format: &str) -> Self {
        let mut s = PhoneGenerator {
            country_formats: vec![],
            region_formats: vec![],
            number_formats: vec![],
            rules: vec![],
            postprocessors: vec![],
        };
        s.register_country_formats(vec![country_format]);
        s.register_region_formats(vec![region_format]);
        s.register_number_formats(vec![number_format]);
        s
    }

    fn generate_random(&self) -> String {
        let mut rng = thread_rng();
        let country = 7;
        let region = rng.gen_range(900, 999);
        let number = rng.gen_range(1_000_000, 9_999_999);
        self.format((country, region, number))
            .map(|result| self.postprocess(result))
            .unwrap()
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

    fn register_country_formats<'a>(&mut self, formats: impl IntoIterator<Item = &'a str>) {
        for format in formats {
            self.country_formats.push(format.to_owned());
        }
    }

    fn register_region_formats<'a>(&mut self, formats: impl IntoIterator<Item = &'a str>) {
        for format in formats {
            self.region_formats.push(format.to_owned());
        }
    }

    fn register_number_formats<'a>(&mut self, formats: impl IntoIterator<Item = &'a str>) {
        for format in formats {
            self.number_formats.push(format.to_owned());
        }
    }

    fn format(&self, phone: Phone) -> Option<String> {
        let mut result = String::new();
        let phone_parts = [
            (u32::from(phone.0), &self.country_formats),
            (u32::from(phone.1), &self.region_formats),
            (phone.2, &self.number_formats),
        ];
        for (part, formats) in &phone_parts {
            if let Some(formatted) = &self.format_part(*part, formats, &self.rules) {
                result.push_str(formatted);
            } else {
                eprintln!("Unable to print phone: {:?}", phone);
                return None;
            }
        }
        Some(result)
    }

    fn format_part(
        &self,
        number: u32,
        formats: &[String],
        rules: &[Box<dyn PhoneFormatRule>],
    ) -> Option<String> {
        let mut rng = thread_rng();
        let chosen_format = formats.choose(&mut rng).unwrap();
        let chosen_rule = rules.choose(&mut rng).unwrap();
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
                        seen_number += u32::from(digit);
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

impl PlusSevenPostProcessingRule {
    fn new() -> Self {
        PlusSevenPostProcessingRule
    }
}

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

#[derive(Serialize)]
struct AugmentedSample {
    message: String,
    phone_indexes: Vec<(usize, usize)>,
}

impl AugmentedSample {
    fn create<T: AsRef<str>>(
        text: &str,
        values: &mut impl Iterator<Item = T>,
    ) -> Option<AugmentedSample> {
        let mut message = String::with_capacity(text.len());
        let mut offset_chars = 0;
        let mut span = text;
        let pattern = "<PHONE>";
        let mut phone_indexes = vec![];

        while let Some(match_offset) = span.find(pattern) {
            offset_chars += span[..match_offset].chars().count();
            message.push_str(&span[..match_offset]);
            if let Some(n) = values.next() {
                let replacement = n.as_ref();
                let replacement_chars_length = replacement.chars().count();
                phone_indexes.push((offset_chars, offset_chars + replacement_chars_length));
                message.push_str(replacement);

                offset_chars += replacement_chars_length;
            } else {
                return None;
            }

            span = &span[match_offset + pattern.len()..];
        }

        if !span.is_empty() {
            message.push_str(span);
        }

        Some(AugmentedSample {
            message,
            phone_indexes,
        })
    }
}

mod tests {

    #![allow(unused)]
    use super::*;

    #[test]
    fn generator_base_scenario() {
        let mut g = PhoneGenerator::new("+#", " (###) ", "###-##-##");
        g.register_rule(Box::new(AsDigitPhoneFormat::new()));

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
        let rule = AsDigitPhoneFormat::new();

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
        let rule = PlusSevenPostProcessingRule::new();

        assert_eq!(rule.transform("+7914"), Some("914".to_owned()));
        assert_eq!(rule.transform("+7 914"), Some("914".to_owned()));
    }

    #[test]
    fn test_glyph_post_processing_rule() {
        let rule = GlyphPostProcessingRule::new();

        assert_eq!(rule.transform("+79041"), Some("+79OЧl".to_owned()));
    }

    #[test]
    fn test_regression() {
        let g = PhoneGenerator::new("+#", " (###) ", "###-##-##");
        let rules: Vec<Box<dyn PhoneFormatRule>> = vec![Box::new(AsTextPhoneFormat)];
        let r = g.format_part(
            8563222,
            &vec!["#######".to_owned()],
            &rules,
        );
        assert_eq!(r.is_some(), true);
    }

    #[test]
    fn test_generate_phone() {
        let mut g = PhoneGenerator::new("+#", " (###) ", "###-##-##");
        g.register_country_formats(vec!["#"]);
        g.register_region_formats(vec!["###", " ### ", "-###-"]);
        g.register_number_formats(vec!["#######"]);

        g.register_rule(Box::new(AsTextPhoneFormat::new()));
        g.register_rule(Box::new(AsDigitPhoneFormat::new()));

        g.register_postprocessor(Box::new(GlyphPostProcessingRule::new()));
        g.register_postprocessor(Box::new(PlusSevenPostProcessingRule::new()));

        println!("{}", g.generate_random());
    }

    #[test]
    fn test_create_augmented_sample() {
        let text = "Первый: <PHONE>, второй: <PHONE>";
        let list = vec!["1", "2"];
        let sample = AugmentedSample::create(text, &mut list.iter()).unwrap();

        assert_eq!(sample.message, "Первый: 1, второй: 2");
        assert_eq!(sample.phone_indexes, vec![(8, 9), (19, 20)]);

        let json = serde_json::to_string(&sample).unwrap();
        assert_eq!(
            json,
            r#"{"message":"Первый: 1, второй: 2","phone_indexes":[[8,9],[19,20]]}"#
        )
    }
}
