extern crate tensorflow;
extern crate encoding;

use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor, Operation, Status};
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

	if let Ok((session, graph)) = create_session(model_path) {
		match stdin().read_to_string(&mut line) {
			Ok(n) if n <= 0 => exit(0),
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