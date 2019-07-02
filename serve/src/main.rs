extern crate tensorflow;
extern crate encoding;

use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor, Operation};
use std::error::Error;
use std::process::exit;

use encoding::{Encoding, EncoderTrap};
use encoding::all::WINDOWS_1251;

fn main() {
	exit(match run() {
		Ok(_) => 0,
		Err(e) => {
			eprintln!("{}", e);
			1
		}
	})
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

fn run() -> Result<(), Box<dyn Error>> {
	let export_dir = "../private/model/1561865486";

	let mut graph = Graph::new();
	let tags: Vec<&str> = vec!["serve"];
	let session = Session::from_saved_model(
		&SessionOptions::new(),
		tags,
		&mut graph,
		export_dir,
	)?;

	let input = graph.operation_by_name_required("input")?;
	let output = graph.operation_by_name_required("output/Reshape")?;

	let text = String::from("Диски в наличии есть?! Может быть есть ещё варианты? Скиньте на вотсап 8 П 924 О 695 Ж 95 А77 Луйста");

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
