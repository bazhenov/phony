extern crate encoding;
extern crate ndarray;
extern crate tensorflow;

pub mod phony;
pub mod phony_tf;
pub mod sample;
pub mod tf_problem;

use phony_tf::MucMetric;

use clap::{App, ArgMatches, SubCommand};
use std::env;

use ndarray::{stack, Array, ArrayBase, Axis, RemoveAxis};
use std::error::Error;
use std::fs::File;
use std::io::{stdin, BufRead, BufReader, Write};
use std::process::exit;
use tf_problem::{EvaluationMetric, TensorflowProblem, TensorflowRunner};

use phony::PhonySample;
use phony_tf::PhonyProblem;

fn main() -> Result<(), Box<dyn Error>> {
    let matches = App::new("phony")
        .author("Denis Bazhenov <dotsid@gmail.com>")
        .version("0.1.0")
        .about("CLI utility for phony classification problem")
        .subcommand(
            SubCommand::with_name("inference")
                .about("run inference over examples from stdin")
                .arg_from_usage("<model> -m, --model=[DIRECTORY] 'Sets model directory'"),
        )
        .subcommand(
            SubCommand::with_name("inference-file")
                .about("run inference over examples from file and update file in place")
                .arg_from_usage("<model> -m, --model=[DIRECTORY] 'Sets model directory'")
                .arg_from_usage("<input_file> -i, --input=[FILE] 'File with examples'")
                .arg_from_usage(
                    "<output_file> -o, --output=[FILE] 'Output file with processed examples'",
                ),
        )
        .subcommand(
            SubCommand::with_name("export")
                .about("export features learning data in HDF5 format")
                .arg_from_usage("<file> -o, --output=[FILE] 'Output file'"),
        )
        .subcommand(
            SubCommand::with_name("eval")
                .about("evaluate result using file with inference results")
                .arg_from_usage("<file> [FILE] 'Inference file'"),
        )
        .get_matches();

    match matches.subcommand() {
        ("inference", Some(matches)) => inference(&matches),
        ("inference-file", Some(matches)) => inference_and_update_file(&matches),
        ("eval", Some(matches)) => evaluate_results(&matches),
        ("export", Some(matches)) => export_features(&matches),
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
fn export_features(matches: &ArgMatches) -> Result<(), Box<dyn Error>> {
    use hdf5::File;

    let file = matches.value_of("file").unwrap();
    let file = File::open(file, "w")?;
    let input_group = file.create_group("input")?;
    let output_group = file.create_group("output")?;
    let mut segment_index = 0usize..;
    let mut segment_input = vec![];
    let mut segment_output = vec![];

    const MAX_SEGMENT_SIZE: usize = 10000;

    let mut current_segment_size = 0usize;

    for (line_no, json) in stdin().lock().lines().enumerate() {
        let json = json?;
        let record = serde_json::from_str::<PhonySample>(json.trim())
            .unwrap_or_else(|_| panic!("Unable to prase JSON on line {}", line_no));
        let problem = PhonyProblem::new(&record)?;

        let features = problem.features();
        let ground_truth = problem.ground_truth();

        current_segment_size += features.dim().0;
        if current_segment_size > MAX_SEGMENT_SIZE {
            // Flushing segment to the HDF5
            let segment_index = segment_index.next().unwrap();

            let s = stack_segment(&segment_input[..]);
            input_group
                .new_dataset::<f32>()
                .create(&format!("{}", segment_index), s.dim())
                .and_then(|dataset| dataset.write(s.view()))?;

            let s = stack_segment(&segment_output[..]);
            output_group
                .new_dataset::<f32>()
                .create(&format!("{}", segment_index), s.dim())
                .and_then(|dataset| dataset.write(s.view()))?;

            current_segment_size = features.dim().0;
            segment_input.clear();
            segment_output.clear();
        }

        segment_input.push(features);
        segment_output.push(ground_truth);
    }

    Ok(())
}

fn stack_segment<T: Copy, D: RemoveAxis>(input: &[Array<T, D>]) -> Array<T, D> {
    let views = input.iter().map(ArrayBase::view).collect::<Vec<_>>();
    stack(Axis(0), &views).expect("Invalid array shape")
}

fn inference(matches: &ArgMatches) -> Result<(), Box<dyn Error>> {
    env::set_var("TF_CPP_MIN_LOG_LEVEL", "1");
    let model_path = matches.value_of("model").unwrap();

    let runner = TensorflowRunner::create_session(model_path)?;
    for line in stdin().lock().lines() {
        let line = line?;
        let record = serde_json::from_str::<PhonySample>(line.trim())?;
        let problem = PhonyProblem::new(&record)?;
        let spans = runner.run_problem(&problem)?;
        for span in spans {
            let phone = record
                .sample
                .chars()
                .skip(span.0)
                .take(span.1 - span.0)
                .collect::<String>();
            println!("{}", phone);
        }
    }
    Ok(())
}

fn inference_and_update_file(matches: &ArgMatches) -> Result<(), Box<dyn Error>> {
    env::set_var("TF_CPP_MIN_LOG_LEVEL", "1");
    let model_path = matches.value_of("model").unwrap();
    let input_file = matches.value_of("input_file").unwrap();
    let output_file = matches.value_of("output_file").unwrap();
    let input = BufReader::new(File::open(input_file)?);
    let mut output = File::create(output_file)?;

    let runner = TensorflowRunner::create_session(model_path)?;
    for line in input.lines() {
        let line = line?;
        let mut record = serde_json::from_str::<PhonySample>(line.trim())?;
        let problem = PhonyProblem::new(&record)?;
        let spans = runner.run_problem(&problem)?;
        record.prediction = Some(spans);
        let bytes = serde_json::to_vec(&record)?;
        output.write_all(&bytes)?;
        writeln!(output)?;
    }
    Ok(())
}

fn evaluate_results(matches: &ArgMatches) -> Result<(), Box<dyn Error>> {
    let input_file = matches.value_of("file").unwrap();
    let input = BufReader::new(File::open(input_file)?);

    let mut metric = MucMetric::new();
    for line in input.lines() {
        let line = line?;
        let record = serde_json::from_str::<PhonySample>(line.trim())?;
        if let Some(label) = record.label {
            if let Some(prediction) = record.prediction {
                metric.update(&label, &prediction);
            }
        }
    }

    println!("{}", metric);
    Ok(())
}
