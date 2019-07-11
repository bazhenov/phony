extern crate tensorflow;
extern crate encoding;

use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor, Operation, Status, FetchToken, TensorType};
use std::error::Error;
use std::process::exit;
use std::env;
use std::io::{stdin, Read};
use clap::App;
use std::path::Path;

use encoding::{Encoding, EncoderTrap};
use encoding::all::WINDOWS_1251;

fn main() {
	let app = App::new("phony-serve")
		.author("Denis Bazhenov <dotsid@gmail.com>")
		.version("1.0.0")
		.about("CLI utility for phony classification problem")
		.arg_from_usage("<model> -m, --model=[DIRECTORY] 'Sets model directory'");

	env::set_var("TF_CPP_MIN_LOG_LEVEL", "1");
	let matches = app.get_matches();
	let model_path = matches.value_of("model").unwrap();

	let mut line = String::new();

	if let Ok(runner) = TensorflowRunner::create_session(model_path) {
		loop {
			match stdin().read_to_string(&mut line) {
				Ok(0) => break,
				
				Ok(_) => {
					let line = line.trim();
					match runner.run_problem(PhonyProblem{}, &line) {
						Ok(mask) => {
							let mask_text  = mask.iter()
								.map(|c| if *c { '^' } else { ' ' })
								.collect::<String>();
							println!("{}", line);
							println!("{}", mask_text);
						},
						Err(e) => {
							eprintln!("{}", e);
							exit(1);
						}
					}
				},
				
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
	graph: Graph
}

impl TensorflowRunner {

	fn create_session<P: AsRef<Path>>(model_path: P) -> Result<Self, Status> {
		let mut graph = Graph::new();
		let tags: Vec<&str> = vec!["serve"];
		let session_options = SessionOptions::new();
		let session = Session::from_saved_model(&session_options, tags, &mut graph, model_path)?;
		Ok(TensorflowRunner { session, graph })
	}

	fn run_problem<I, O>(&self, problem: impl TensorflowProblem<I, O>, example: I) -> Result<O, Box<dyn Error>> {
		let inputs = problem.tensors_from_example(&example)?;

		let (input_op, output_op) = problem.retrieve_input_output_operation(&self.graph)?;

		let outputs = inputs.iter()
			.map(|t| problem.feed(&self.session, &input_op, &output_op, t).unwrap())
			.collect::<Vec<_>>();

		Ok(problem.output_from_tensors(&example, outputs))
	}
}

/// Микрофреймворк для решения задач при помощи библиотеки Tensorflow.
/// 
/// Подразумевается, что используя framework программист определяет следующие аспекты поведения:
/// * как из модели достать точки входа и выхода. Вход – это placeholder, который определяется при помощи
/// входного тензора. Выход – это ответ системы, который содержит пометку класса или любую другую информацию, которая
/// является целью вычислений.
/// * как из примера получить тензор
pub trait TensorflowProblem<I: ?Sized, O> {

	/// Тип тензора-входа (`u32`/`f32` и т.д.)
	type TensorInputType: TensorType;

	/// Тип тензора-выхода (`u32`/`f32` и т.д.)
	type TensorOutputType: TensorType;

	fn tensors_from_example(&self, example: &I) -> Result<Vec<Tensor<Self::TensorInputType>>, Box<dyn Error>>;

	fn retrieve_input_output_operation(&self, graph: &Graph) -> Result<(Operation, Operation), Status>;

	fn output_from_tensors(&self, example: &I, tensors: Vec<Tensor<Self::TensorOutputType>>) -> O;

	fn fetch_tensor(&self, args: &mut SessionRunArgs, token: FetchToken) -> Result<Tensor<Self::TensorOutputType>, Status> {
		args.fetch::<Self::TensorOutputType>(token)
	}

	fn feed(&self, session: &Session, input_op: &Operation, output_op: &Operation, input: &Tensor<Self::TensorInputType>)
			-> Result<Tensor<Self::TensorOutputType>, Box<dyn Error>> {
		
		let mut run_args = SessionRunArgs::new();
		run_args.add_feed(&input_op, 0, input);
		let output_token = run_args.request_fetch(&output_op, 0);

		session.run(&mut run_args)?;

		Ok(run_args.fetch(output_token)?)
	}
}

struct PhonyProblem {}

impl PhonyProblem {
	
	fn str_to_tensor<T>(string: &str) -> Result<Tensor<T>, Box<dyn Error>>
			where T: From<u8> + TensorType {
		
		let len = string.chars().count();

		let mut tensor: Tensor<T> = Tensor::new(&[1, len as u64]);
		for (i, c) in string.chars().enumerate() {
			let r = WINDOWS_1251.encode(&c.to_string(), EncoderTrap::Strict)?;
			tensor[i] = r[0].into();
		}
		Ok(tensor)
	}
}

impl TensorflowProblem<&str, Vec<bool>> for PhonyProblem {

	type TensorInputType = f32;
	type TensorOutputType = f32;

	fn tensors_from_example(&self, e: &&str) -> Result<Vec<Tensor<Self::TensorInputType>>, Box<dyn Error>> {
		character_ngrams(e, 16)
			.map(PhonyProblem::str_to_tensor)
			.collect()
	}

	fn retrieve_input_output_operation(&self, graph: &Graph) -> Result<(Operation, Operation), Status> {
		let input = graph.operation_by_name_required("input")?;
		let output = graph.operation_by_name_required("output/Reshape")?;

		Ok((input, output))
	}

	fn output_from_tensors(&self, _example: &&str, tensors: Vec<Tensor<Self::TensorOutputType>>) -> Vec<bool> {
		let mut mask = vec![0u8; _example.len()];
		let tensors_length = tensors.len() as u8;
		let length = tensors[0].dims()[1] as usize;
		for (offset, tensor) in tensors.iter().enumerate() {
			for i in 0..length {
				if tensor[i] > 0.5 {
					mask[i + offset] += 1;
				}
			}
		}
		mask.iter().map(|i| *i > tensors_length / 2).collect()
	}
}

pub struct CharNgrams<'a> {
	text: &'a str,
	position: (usize, usize)
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

pub fn character_ngrams<'a>(text: &'a str, n: usize) -> CharNgrams<'a> {
	CharNgrams { text, position: (0, advance_character(text, 0, n)) }
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
}