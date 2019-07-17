
mod odds {
	use rand::{thread_rng, Rng};

	struct Odds<T>(u8, T);

	trait RandomChoice<T> {

		fn choose_random(&self) -> T;
		fn sum(&self) -> u32;
	}

	impl<T> RandomChoice<T> for Vec<Odds<T>>
		where T: Copy {

		fn choose_random(&self) -> T {
			let mut sum = self.sum();
			let mut n = thread_rng().gen_range(0, sum);
			let mut iterator = self.iter();
			while let Some(Odds(odds, inner)) = iterator.next() {
				if n <= *odds as u32 {
					return *inner;
				}
				n -= *odds as u32;
			}
			self[0].1
		}

		fn sum(&self) -> u32 {
			self.iter().map(|i| i.0 as u32).sum()
		}
	}

	mod tests {

		use super::*;

		#[test]
		fn choose_random() {
			let choices = vec![Odds(1, "always"), Odds(0, "never")];
			assert_eq!("always", choices.choose_random());
		}
	}
}
