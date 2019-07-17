mod odds;

const NUMBERS: [&str; 10] = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"];
const NUMBERS_TENTH: [&str; 10] = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
  "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"];
const NUMBERS_10: [&str; 10] = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят",
  "восемдесят", "девяносто"];
const NUMBERS_100: [&str; 10] = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот",
  "восемьсот", "девятьсот"];

fn main() {
  
}
  
fn numerate(n: u32) -> String {
  let mut result = String::new();
  numerate_part(&mut result, n);
  return result;
}

fn numerate_part(base: &mut String, n: u32) {
  match n {
    x if x < 10 => base.push_str(NUMBERS[x as usize]),
    x if x < 20 => base.push_str(NUMBERS_TENTH[(x - 10) as usize]),
    x if x < 100 => {
      base.push_str(NUMBERS_10[(x % 100 / 10) as usize]);
      base.push(' ');
      numerate_part(base, n % 10);
    },
    x if x <= 1000 => {
      base.push_str(NUMBERS_100[(x % 1000 / 100) as usize]);
      base.push(' ');
      numerate_part(base, n % 100);
    }
    _ => panic!("Not implmented for numbers greater than 999")
  }
}

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

#[cfg(test)]
mod tests {

  use super::*;

  #[test]
  fn test_numerate() {
    assert_eq!(numerate(0), "ноль");
    assert_eq!(numerate(1), "один");
    assert_eq!(numerate(10), "десять");
    assert_eq!(numerate(13), "тринадцать");
    assert_eq!(numerate(37), "тридцать семь");
    assert_eq!(numerate(101), "сто один");
    assert_eq!(numerate(105), "сто пять");
    assert_eq!(numerate(833), "восемьсот тридцать три");
    assert_eq!(numerate(916), "девятьсот шестнадцать");
  }

	#[test]
	fn test_number_format() {
		assert_eq!(phone_format((7, 999, 3053315), "+# (###) ###-##-##"), "+7 (999) 305-33-15");
	}
}
