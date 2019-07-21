mod digits {

    struct DigitIterator {
        number: u32,
        magnitude: u32,
    }

    impl Iterator for DigitIterator {
        type Item = u8;

        fn next(&mut self) -> Option<Self::Item> {
            if self.magnitude <= 0 {
                None
            } else {
                let result = self.number / self.magnitude;
                self.number = self.number % self.magnitude;
                self.magnitude /= 10;
                Some(result as u8)
            }
        }
    }

    trait Digits {
        /// Returns iterator over decimal digits for a given type
        fn digits(&self) -> DigitIterator;
    }

    impl<T> Digits for T
    where
        T: Copy,
        f64: From<T>,
        u32: From<T>,
    {
        fn digits(&self) -> DigitIterator {
            let magnitude = f64::from(*self).log10().ceil() as u32;
            DigitIterator {
                number: u32::from(*self),
                magnitude: 10u32.pow(magnitude - 1),
            }
        }
    }

    mod tests {

        use super::*;

        #[test]
        fn iterating_over_u8() {
            let mut iter = 32u8.digits();

            assert_eq!(iter.next(), Some(3));
            assert_eq!(iter.next(), Some(2));
            assert_eq!(iter.next(), None);
        }

        #[test]
        fn iterating_over_u32() {
            let mut iter = 123456789u32.digits();

            for i in 1..10 {
                assert_eq!(iter.next(), Some(i));
            }
            assert_eq!(iter.next(), None);
        }
    }
}
