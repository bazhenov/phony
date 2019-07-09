extern crate tensorflow;
extern crate encoding;

use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor, Operation, Status};
use std::error::Error;
use std::process::exit;
use std::env;
use std::io::{stdin, Read};
use clap::App;
use std::path::Path;
use std::str::CharIndices;
use std::iter::Skip;

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

	if let Ok((session, graph)) = create_session(model_path) {
		loop {
			match stdin().read_to_string(&mut line) {
				Ok(0) => break,
				
				Ok(_) => {
					if let Err(e) = run(line.trim(), &session, &graph) {
						eprintln!("{}", e);
						exit(1);
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

pub trait TensorflowProblem<E> {

	fn tensor_from_example(example: E) -> Result<Tensor<f32>, Box<dyn Error>>;

	fn retrieve_input_output_operation(graph: &Graph) -> Result<(Operation, Operation), Status>;
}

struct PhonyProblem {}

impl TensorflowProblem<&str> for PhonyProblem {

	fn tensor_from_example(e: &str) -> Result<Tensor<f32>, Box<dyn Error>> {
		let len = e.chars().count();

		let mut tensor = Tensor::new(&[1, len as u64]);
		for (i, c) in e.chars().enumerate() {
			let r = WINDOWS_1251.encode(&c.to_string(), EncoderTrap::Strict)?;
			tensor[i] = r[0] as f32;
		}
		Ok(tensor)
	}

	fn retrieve_input_output_operation(graph: &Graph) -> Result<(Operation, Operation), Status> {
		let input = graph.operation_by_name_required("input")?;
		let output = graph.operation_by_name_required("output/Reshape")?;

		Ok((input, output))
	}
}

fn tensor_from_str(string: &str) -> Result<Tensor<f32>, Box<dyn Error>> {
	let len = string.chars().count();

	let mut tensor = Tensor::new(&[1, len as u64]);
	for (i, c) in string.chars().enumerate() {
		let r = WINDOWS_1251.encode(&c.to_string(), EncoderTrap::Strict)?;
		tensor[i] = r[0] as f32;
	}
	Ok(tensor)
}

fn session_run(session: &Session, input: &Operation, input_tensor: &Tensor<f32>,
							 output: &Operation) -> Result<Tensor<f32>, Box<dyn Error>> {
	let mut run_args = SessionRunArgs::new();

	run_args.add_feed(&input, 0, &input_tensor);
	let output_token = run_args.request_fetch(&output, 0);

	session.run(&mut run_args)?;

	Ok(run_args.fetch::<f32>(output_token)?)
}

fn create_session<P: AsRef<Path>>(model_path: P) -> Result<(Session, Graph), Status> {
	let mut graph = Graph::new();
	let tags: Vec<&str> = vec!["serve"];
	let session_options = SessionOptions::new();
	let session = Session::from_saved_model(&session_options, tags, &mut graph, model_path)?;
	Ok((session, graph))
}

fn run(text: &str, session: &Session, graph: &Graph) -> Result<(), Box<dyn Error>> {
	let input = graph.operation_by_name_required("input")?;
	let output = graph.operation_by_name_required("output/Reshape")?;

	let indices = text.char_indices().collect::<Vec<_>>();

	let chunk_size = 32;

	for chunk in indices.chunks(chunk_size) {
		let chunk_str = chunk.iter()
			.map(|i| i.1)
			.collect::<String>();
		let input_tensor = tensor_from_str(&chunk_str)?;
		let output_tensor = session_run(&session, &input, &input_tensor, &output)?;
		let mut vector = vec![0f32; chunk_size];
		for i in 0..chunk.len() {
			vector[i] = output_tensor[i];
		}
		let mask_str = vector.iter()
			.map(|i| if *i >= 0.5 { '^' } else { ' ' })
			.collect::<String>();
		println!("{}", chunk_str);
		println!("{}", mask_str);
	}

	Ok(())
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
		let result = Some(&self.text[self.position.0..self.position.1]);

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