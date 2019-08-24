extern crate encoding;
extern crate tensorflow;
#[macro_use(s)]
extern crate ndarray;

pub mod sample;
pub mod tf_problem;

use clap::{App, ArgMatches, SubCommand};
use std::env;

use ndarray::{arr2, Array1, Array2};
use std::error::Error;
use std::io::{stdin, BufRead};
use std::ops::Range;
use std::process::exit;
use tf_problem::{TensorflowProblem, TensorflowRunner};

use sample::PhonySample;

use encoding::all::WINDOWS_1251;
use encoding::{EncoderTrap, Encoding};

fn main() {
    let matches = App::new("phony-serve")
        .author("Denis Bazhenov <dotsid@gmail.com>")
        .version("1.0.0")
        .about("CLI utility for phony classification problem")
        .subcommand(
            SubCommand::with_name("inference")
                .about("run inference over examples from stdin")
                .arg_from_usage("<model> -m, --model=[DIRECTORY] 'Sets model directory'")
                .arg_from_usage(
                    "[only_mode] -o, --only 'Print only matched characters from phone'",
                ),
        )
        .subcommand(
            SubCommand::with_name("export")
                .about("export features learning data in HDF5 format")
                .arg_from_usage("<file> -o, --output=[FILE] 'Output file'"),
        )
        .get_matches();

    match matches.subcommand() {
        ("inference", Some(matches)) => {
            inference(&matches);
        }
        ("export", Some(matches)) => {
            export_features(&matches);
        }
        _ => {
            eprintln!("{}", matches.usage());
            exit(1);
        }
    }
}

// Срабатывает deref_addrof на макрос s![.., ..]. Не смог разобраться как исправить,
// поэтому отключил. Рекомендация использовать s![.., ..] есть в официальной документации
// ndarray.
// см. https://rust-lang.github.io/rust-clippy/master/index.html#deref_addrof
#[allow(clippy::deref_addrof)]
fn export_features(matches: &ArgMatches) {
    use hdf5::File;

    let file = matches.value_of("file").unwrap();
    let input_group = File::open(file, "w")
        .and_then(|f| f.create_group("input"))
        .expect("Unable to open group");

    for (i, line) in stdin().lock().lines().enumerate() {
        let line = line.expect("Unable to read line");
        let record =
            serde_json::from_str::<PhonySample>(line.trim()).expect("Unable to read sample");
        let problem = PhonyProblem::new(&record).expect("Unable to create context");
        let features = problem.features();

        input_group
            .new_dataset::<f32>()
            .create(&format!("{}", i), features.dim())
            .and_then(|dataset| dataset.write(features.slice(s![.., ..])))
            .expect("Unable to create dataset");
    }
}

fn inference(matches: &ArgMatches) {
    env::set_var("TF_CPP_MIN_LOG_LEVEL", "1");
    let model_path = matches.value_of("model").unwrap();
    let only_mode = matches.is_present("only_mode");

    let runner = TensorflowRunner::create_session(model_path).expect("Unable to create session");
    for line in stdin().lock().lines() {
        let line = line.expect("Unable to read line");
        let record = serde_json::from_str::<PhonySample>(line.trim())
            .expect("Unable to read record from stdin");
        let problem = PhonyProblem::new(&record).expect("Unable to build problem");
        match runner.run_problem(&problem) {
            Ok(mask) => {
                if only_mode {
                    for span in mask.iter().spans(|c| *c) {
                        let phone = line
                            .chars()
                            .skip(span.start)
                            .take(span.end - span.start)
                            .collect::<String>();
                        println!("{}", phone);
                    }
                } else {
                    let mask_text = mask
                        .iter()
                        .map(|c| if *c { '^' } else { ' ' })
                        .collect::<String>();
                    println!("{}", line);
                    println!("{}", mask_text);
                }
            }
            Err(e) => {
                eprintln!("{}", e);
                exit(1);
            }
        }
    }
}

struct PhonyProblem<'a> {
    chars: Vec<u8>,
    left_padding: usize,
    right_padding: usize,
    sample: &'a PhonySample,
}

impl<'a> PhonyProblem<'a> {
    const WINDOW: usize = 16;

    fn new(sample: &'a PhonySample) -> Result<Self, Box<dyn Error>> {
        if let Some((left_padding, padded_string, right_padding)) =
            Self::pad_string(&sample.text, Self::WINDOW)
        {
            Ok(PhonyProblem {
                chars: WINDOWS_1251.encode(&padded_string, EncoderTrap::Strict)?,
                left_padding,
                right_padding,
                sample,
            })
        } else {
            Ok(PhonyProblem {
                chars: WINDOWS_1251.encode(&sample.text, EncoderTrap::Strict)?,
                left_padding: 0,
                right_padding: 0,
                sample,
            })
        }
    }

    fn ground_truth(&self) -> Array2<f32> {
        let mut mask1d = Array1::<f32>::zeros(self.chars.len());

        for span in &self.sample.spans {
            let from = self.left_padding + span.0;
            let to = self.left_padding + span.1;
            mask1d.slice_mut(s![from..to]).fill(1.);
        }

        let ngrams = self.chars.len() - Self::WINDOW + 1;
        let mut mask2d = Array2::<f32>::zeros((ngrams, Self::WINDOW));

        for (i, mut row) in mask2d.genrows_mut().into_iter().enumerate() {
            row.assign(&mask1d.slice(s![i..i + Self::WINDOW]));
        }

        mask2d
    }

    fn pad_string(string: &str, desired_length: usize) -> Option<(usize, String, usize)> {
        let char_length = string.chars().count();
        if char_length >= desired_length {
            return None;
        }

        let bytes_length = string.len();
        let left_padding = (desired_length - char_length) / 2;
        let right_padding = desired_length - char_length - left_padding;
        let mut padded_string = String::with_capacity(bytes_length + left_padding + right_padding);

        for _ in 0..left_padding {
            padded_string.push(' ');
        }

        padded_string.push_str(string);

        for _ in 0..right_padding {
            padded_string.push(' ');
        }

        Some((left_padding, padded_string, right_padding))
    }
}

impl<'a> TensorflowProblem for PhonyProblem<'a> {
    type TensorInputType = f32;
    type TensorOutputType = f32;
    type Input = PhonySample;
    type Output = Vec<bool>;
    const GRAPH_INPUT_NAME: &'static str = "input";
    const GRAPH_OUTPUT_NAME: &'static str = "output/Reshape";

    fn features(&self) -> Array2<Self::TensorInputType> {
        let ngrams = self.chars.windows(Self::WINDOW).collect::<Vec<_>>();

        let mut result = Array2::zeros((ngrams.len(), Self::WINDOW));

        for (i, ngram) in ngrams.iter().enumerate() {
            for (j, c) in ngram.iter().enumerate() {
                result[[i, j]] = f32::from(*c);
            }
        }

        result
    }

    fn output(&self, tensors: Array2<Self::TensorOutputType>) -> Vec<bool> {
        let mut mask = vec![Accumulator(0, 0); self.chars.len()];
        let character_length = self.chars.len() - self.left_padding - self.right_padding;

        for i in 0..tensors.rows() {
            for j in 0..tensors.cols() {
                mask[i + j].register(tensors[[i, j]] > 0.5);
            }
        }
        mask.iter()
            .map(|a| a.ratio() > 0.5)
            // отрезаем от маски "хвостики" порожденные padding'ом строки
            .skip(self.left_padding)
            .take(character_length)
            .collect()
    }
}

/// Простой счетчик – регистририует количество ложных/положительных срабатываный. Метод [`register`](#method.register)
#[derive(Copy, Clone)]
struct Accumulator(u16, u16);

impl Accumulator {
    /// Регистрирует срабатывание: ложное или положительное в зависимости от значения аргумента `hit`.
    fn register(&mut self, hit: bool) {
        if hit {
            self.0 += 1;
        }
        self.1 += 1;
    }

    /// доля положительных вызовов по отношению к общему количеству
    fn ratio(self) -> f32 {
        f32::from(self.0) / f32::from(self.1)
    }
}

pub struct CharNgrams<'a> {
    text: &'a str,
    position: (usize, usize),
}

fn advance_character(text: &str, pos: usize, n: usize) -> usize {
    let mut pos = pos;
    for _ in 0..n {
        pos += 1;
        while !text.is_char_boundary(pos) && pos <= text.len() {
            pos += 1;
        }
    }
    pos
}

impl<'a> Iterator for CharNgrams<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position.1 > self.text.len() {
            return None;
        }
        let span = self.position.0..self.position.1;
        let result = Some(&self.text[span]);

        self.position.0 = advance_character(self.text, self.position.0, 1);
        self.position.1 = advance_character(self.text, self.position.1, 1);

        result
    }
}

pub fn character_ngrams(text: &str, n: usize) -> CharNgrams<'_> {
    CharNgrams {
        text,
        position: (0, advance_character(text, 0, n)),
    }
}

struct Spans<'a, I, F> {
    iterator: &'a mut I,
    position: usize,
    predicate: F,
}

trait SpanExtension: Iterator + Sized {
    fn spans<F>(&mut self, f: F) -> Spans<'_, Self, F>
    where
        F: Fn(Self::Item) -> bool,
    {
        Spans {
            iterator: self,
            position: 0,
            predicate: f,
        }
    }
}

impl<T: Iterator> SpanExtension for T {}

impl<I, F> Iterator for Spans<'_, I, F>
where
    I: Iterator,
    F: Fn(I::Item) -> bool,
{
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            self.position += 1;
            match self.iterator.next().map(&self.predicate) {
                Some(true) => break,
                None => return None,
                Some(false) => {}
            }
        }
        let from = self.position - 1;
        loop {
            self.position += 1;
            match self.iterator.next().map(&self.predicate) {
                Some(false) => return Some(from..self.position - 1),
                None => return Some(from..self.position - 1),
                Some(true) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn text_segmentate() {
        let mut l = character_ngrams("12345", 3);
        assert_eq!(l.next(), Some("123"));
        assert_eq!(l.next(), Some("234"));
        assert_eq!(l.next(), Some("345"));
        assert_eq!(l.next(), None);
    }

    #[test]
    fn ngrams_from_utf8_text() {
        let mut l = character_ngrams("абвгд", 3);
        assert_eq!(l.next(), Some("абв"));
        assert_eq!(l.next(), Some("бвг"));
        assert_eq!(l.next(), Some("вгд"));
        assert_eq!(l.next(), None);
    }

    #[test]
    fn ngrams_from_short_text() {
        let mut l = character_ngrams("абвгд", 6);
        assert_eq!(l.next(), None);
    }

    #[test]
    fn ngrams_from_one_char() {
        let mut l = character_ngrams("1", 1);
        assert_eq!(l.next(), Some("1"));
        assert_eq!(l.next(), None);
    }

    #[test]
    fn pad_string() {
        assert_eq!(
            PhonyProblem::pad_string("123", 5),
            Some((1usize, String::from(" 123 "), 1usize))
        );
    }

    #[test]
    fn groups() {
        let v = vec![0, 0, 1, 0, 0, 1, 1, 1, 0];

        let spans = v.iter().spans(|i| *i > 0).collect::<Vec<_>>();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], 2..3);
        assert_eq!(spans[1], 5..8);

        let spans = v.iter().spans(|i| *i == 0).collect::<Vec<_>>();
        assert_eq!(spans.len(), 3);
        assert_eq!(spans[0], 0..2);
        assert_eq!(spans[1], 3..5);
        assert_eq!(spans[2], 8..9);
    }

    #[test]
    fn should_be_able_to_reconstruct_ground_truth_labels() {
        let example = PhonySample {
            text: String::from("text"),
            spans: vec![(0, 1), (2, 4)],
        };
        let p = PhonyProblem::new(&example).unwrap();

        let truth = p.ground_truth();
        assert_eq!(p.left_padding, 6);
        assert_eq!(p.right_padding, 6);
        // Индексы в маске смещены из за padding'а. Это необходимо учитывать при изменений ширины окна
        assert_eq!(
            truth,
            arr2(&[[0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0.,]])
        );
        //assert_eq!(p.output(truth), vec![(0, 1), (2, 4)]);
    }
}
