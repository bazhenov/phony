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

	let text = String::from("Привет, перезвоните мне на +7(907-8O57113 скорее");
	for (i, c) in text.chars().enumerate().take(64) {
		let r = WINDOWS_1251.encode(&c.to_string(), EncoderTrap::Strict)?;
		input_tensor[i] = r[0] as f32;
	}

	let mut run_args = SessionRunArgs::new();

	run_args.add_feed(&input, 0, &input_tensor);
	let output_token = run_args.request_fetch(&output, 0);

	session.run(&mut run_args)?;

	let size = 64;
	let tensor = run_args.fetch::<f32>(output_token)?;
	
	let mut vector = vec![0f32; size];
	for i in 0..size {
		vector[i] = tensor[i];
	}

	for (start, stop) in get_indicies(vector, 0.5) {
		let fragment = text.chars().skip(start).take(stop - start).collect::<String>();
		println!(" - {}", fragment);
	}

	println!("");
	Ok(())
}

#[derive(PartialEq)]
enum State {
	SCANNING, CAPTURING
}

/// Возвращает пары (начало, конец) подпоследовательностей.
fn get_indicies<T: PartialOrd + Copy>(list: Vec<T>, t: T) -> Vec<(usize, usize)> {
	use State::*;
	let mut result = vec![];
	let mut start_index = 0;
	let mut state = SCANNING;
	for (i, item) in list.iter().enumerate() {
		state = match (state, *item > t) {
			(SCANNING, false) => SCANNING,
			(CAPTURING, true) => CAPTURING,
			(SCANNING, true) => {
				start_index = i;
				CAPTURING
			},
			(CAPTURING, false) => {
				result.push((start_index, i));
				SCANNING
			}
		}
	}
	if state == CAPTURING {
		result.push((start_index, list.len()));
	}
	result
}

mod tests {

	use super::*;

	#[test]
	fn test_get_indicies() {
		assert_eq!(get_indicies(vec![], 0f32), vec![]);

		assert_eq!(get_indicies(vec![0, 1, 1, 0, 1], 0), vec![(1, 3), (4, 5)]);
		assert_eq!(get_indicies(vec![0, 1, 1, 0], 0), vec![(1, 3)]);

		assert_eq!(get_indicies(vec![1, 1, 1, 0], 0), vec![(0, 3)]);
	}
}
