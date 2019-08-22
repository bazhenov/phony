extern crate encoding;
extern crate tensorflow;

use clap::App;
use std::env;

use ndarray::{Array, Array2, ArrayBase, ShapeBuilder};
use std::error::Error;
use std::io::{stdin, BufRead};
use std::ops::Range;
use std::path::Path;
use std::process::exit;
use tensorflow::{
    FetchToken, Graph, Operation, Session, SessionOptions, SessionRunArgs, Status, Tensor,
    TensorType,
};

use encoding::all::WINDOWS_1251;
use encoding::{EncoderTrap, Encoding};

fn main() {
    let matches = App::new("phony-serve")
        .author("Denis Bazhenov <dotsid@gmail.com>")
        .version("1.0.0")
        .about("CLI utility for phony classification problem")
        .arg_from_usage("<model> -m, --model=[DIRECTORY] 'Sets model directory'")
        .arg_from_usage("[only_mode] -o, --only 'Print only matched characters from phone'")
        .get_matches();

    env::set_var("TF_CPP_MIN_LOG_LEVEL", "1");
    let model_path = matches.value_of("model").unwrap();
    let only_mode = matches.is_present("only_mode");

    if let Ok(runner) = TensorflowRunner::create_session(model_path) {
        for line in stdin().lock().lines() {
            let line = line.expect("Unable to read line");
            let line = line.trim();
            match runner.run_problem::<PhonyProblem>(line) {
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
}

struct TensorflowRunner {
    session: Session,
    graph: Graph,
}

impl TensorflowRunner {
    fn create_session<M: AsRef<Path>>(model_path: M) -> Result<Self, Status> {
        let mut graph = Graph::new();
        let tags: Vec<&str> = vec!["serve"];
        let session_options = SessionOptions::new();
        let session = Session::from_saved_model(&session_options, tags, &mut graph, model_path)?;
        Ok(TensorflowRunner { session, graph })
    }

    fn run_problem<P: TensorflowProblem>(
        &self,
        example: &P::Input,
    ) -> Result<P::Output, Box<dyn Error>> {
        let problem = P::new_context(&example)?;
        let (input_op, output_op) = problem.retrieve_input_output_operation(&self.graph)?;

        let inputs = problem.tensors_from_example(&example)?;
        if !inputs.is_standard_layout() {
            panic!("ndarray should be in standard layout");
        }

        let tensor = tensor_from_ndarray(&inputs);
        let output = problem.feed(&self.session, &input_op, &output_op, &tensor)?;
        Ok(problem.output_from_tensors(&example, ndarray_from_tensor(&output)))
    }
}

fn tensor_from_ndarray<D, T, S>(input: &ArrayBase<D, S>) -> Tensor<T>
where
    T: TensorType,
    D: ndarray::Data<Elem = T>,
    S: ndarray::Dimension,
{
    let shape = input.shape().iter().map(|i| *i as u64).collect::<Vec<_>>();
    if let Some(slice) = input.as_slice() {
        Tensor::new(&shape[..])
            .with_values(slice)
            .expect("Can't build tensor")
    } else {
        panic!("Can't get slice from ndarray");
    }
}

fn ndarray_from_tensor<T: TensorType + Copy>(input: &Tensor<T>) -> Array2<T> {
    let vector = Array::from_iter(input.iter().map(|i| *i));
    let [rows, columns] = match *input.dims() {
        [a, b] => [a, b],
        _ => panic!("Should be 2-dimensional "),
    };
    let dims = (rows as usize, columns as usize);
    vector.into_shape(dims).expect("Unable to reshape")
}

/// Микрофреймворк для решения задач при помощи библиотеки Tensorflow.
///
/// Подразумевается, что используя framework программист определяет следующие аспекты поведения:
/// * как из модели достать точки входа и выхода. Вход – это placeholder, который определяется при помощи
/// входного тензора. Выход – это ответ системы, который содержит пометку класса или любую другую информацию, которая
/// является целью вычислений.
/// * как из примера получить тензор
pub trait TensorflowProblem {
    /// Тип тензора-входа (`u32`/`f32` и т.д.)
    type TensorInputType: TensorType + Copy;

    /// Тип тензора-выхода (`u32`/`f32` и т.д.)
    type TensorOutputType: TensorType + Copy;

    const INPUT_OPERATION: &'static str;
    const OUTPUT_OPERATION: &'static str;

    type Input: ?Sized;
    type Output;

    fn new_context(example: &Self::Input) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized;

    fn tensors_from_example(
        &self,
        example: &Self::Input,
    ) -> Result<Array2<Self::TensorInputType>, Box<dyn Error>>;

    fn retrieve_input_output_operation(
        &self,
        graph: &Graph,
    ) -> Result<(Operation, Operation), Status> {
        let input = graph.operation_by_name_required(Self::INPUT_OPERATION)?;
        let output = graph.operation_by_name_required(Self::OUTPUT_OPERATION)?;

        Ok((input, output))
    }

    fn output_from_tensors(
        &self,
        example: &Self::Input,
        tensors: Array2<Self::TensorOutputType>,
    ) -> Self::Output;

    fn fetch_tensor(
        &self,
        args: &mut SessionRunArgs,
        token: FetchToken,
    ) -> Result<Tensor<Self::TensorOutputType>, Status> {
        args.fetch::<Self::TensorOutputType>(token)
    }

    fn feed(
        &self,
        session: &Session,
        input_op: &Operation,
        output_op: &Operation,
        input: &Tensor<Self::TensorInputType>,
    ) -> Result<Tensor<Self::TensorOutputType>, Box<dyn Error>> {
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&input_op, 0, input);
        let output_token = run_args.request_fetch(&output_op, 0);

        session.run(&mut run_args)?;

        Ok(run_args.fetch(output_token)?)
    }
}

struct PhonyProblem {
    chars: Vec<u8>,
    left_padding: usize,
    right_padding: usize,
}

impl PhonyProblem {
    const WINDOW: usize = 16;

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

impl TensorflowProblem for PhonyProblem {
    type TensorInputType = f32;
    type TensorOutputType = f32;
    type Input = str;
    type Output = Vec<bool>;
    const INPUT_OPERATION: &'static str = "input";
    const OUTPUT_OPERATION: &'static str = "output/Reshape";

    fn new_context(example: &Self::Input) -> Result<Self, Box<dyn Error>> {
        if let Some((left_padding, padded_string, right_padding)) =
            Self::pad_string(example, Self::WINDOW)
        {
            Ok(PhonyProblem {
                chars: WINDOWS_1251.encode(&padded_string, EncoderTrap::Strict)?,
                left_padding,
                right_padding,
            })
        } else {
            Ok(PhonyProblem {
                chars: WINDOWS_1251.encode(example, EncoderTrap::Strict)?,
                left_padding: 0,
                right_padding: 0,
            })
        }
    }

    fn tensors_from_example(
        &self,
        _e: &Self::Input,
    ) -> Result<Array2<Self::TensorInputType>, Box<dyn Error>> {
        let ngrams = self.chars.windows(Self::WINDOW).collect::<Vec<_>>();

        let mut result = Array2::zeros((ngrams.len(), Self::WINDOW));

        for (i, ngram) in ngrams.iter().enumerate() {
            for (j, c) in ngram.iter().enumerate() {
                result[[i, j]] = f32::from(*c);
            }
        }

        Ok(result)
    }

    fn output_from_tensors(
        &self,
        _example: &Self::Input,
        tensors: Array2<Self::TensorOutputType>,
    ) -> Vec<bool> {
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
    use ndarray::arr2;

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
    fn create_ndarray_from_tensor() {
        let tensor = Tensor::new(&[2, 2]).with_values(&[1, 2, 3, 4]);
        let ndarray = tensor.as_ref().map(ndarray_from_tensor).unwrap();
        assert_eq!(ndarray.dim(), (2usize, 2usize));
        // Array values are expected in row-major format
        assert_eq!(ndarray.as_slice(), Some(&[1, 2, 3, 4][..]));
    }

    #[test]
    fn create_tensor_from_ndarray() {
        let ndarray = arr2(&[[1, 2], [3, 4]]);
        let tensor = tensor_from_ndarray(&ndarray);
        assert_eq!(tensor.dims(), &[2, 2]);
        let content = tensor.iter().cloned().collect::<Vec<_>>();
        assert_eq!(content, vec![1, 2, 3, 4]);
    }
}
