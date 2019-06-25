extern crate tensorflow;
extern crate encoding;

use tensorflow::{Graph, Session, SessionOptions, SessionRunArgs, Tensor};
use std::fs::File;
use std::io::Read;
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

fn run() -> Result<(), Box<dyn Error>> {
	let mut g = Graph::new();

	let mut proto = Vec::new();
	File::open("../private/model.pb")?.read_to_end(&mut proto)?;
	let export_dir = "../private/model/1561464809";

	let mut graph = Graph::new();
	let tags: Vec<&str> = vec!["serve"];
	let session = Session::from_saved_model(
		&SessionOptions::new(),
		tags,
		&mut graph,
		export_dir,
	)?;

	if let Ok(devices) = session.device_list() {
		for ref d in devices {
			println!("{}", d.device_type);
			println!("{}", d.name);
		}
	}

	let input = graph.operation_by_name_required("embedding_input")?;
	let output = graph.operation_by_name_required("flatten/Reshape")?;

	let mut input_tensor = Tensor::new(&[1, 64]).with_values(&[0f32; 64]).unwrap();

	let text = "Привет, перезвоните мне на +7(907-8O57113 скорее";
	for (i, c) in text.chars().enumerate().take(64) {
		let r = WINDOWS_1251.encode(&c.to_string(), EncoderTrap::Strict)?;
		input_tensor[i] = r[0] as f32;
	}

	let mut run_args = SessionRunArgs::new();

	run_args.add_feed(&input, 0, &input_tensor);
	let output_token = run_args.request_fetch(&output, 0);

	session.run(&mut run_args)?;

	let tensor = run_args.fetch::<f32>(output_token)?;
	
	for i in 0..64 {
		if tensor[i] > 0.5 {
			print!("{}", text.chars().nth(i).unwrap());
		}
	}
	println!("");
	Ok(())
}
