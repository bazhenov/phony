mod digits;
mod numerate;
mod odds;

fn main() {}

type Phone = (u16, u16, u32);

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

trait PhoneFormat {
    fn format_as_phone(&self, format: &str) -> Option<String>;
}

impl<T> PhoneFormat for T
where
    f32: From<T>,
    T: Copy,
{
    fn format_as_phone(&self, format: &str) -> Option<String> {
        let magnitude = f32::from(*self).log10().ceil() as u8;
        Some(String::from(format!("{}", magnitude)))
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
    fn format_phone_part() {
        assert_eq!(314_u16.format_as_phone("(###)"), Some("(314)".to_string()));
    }
}
