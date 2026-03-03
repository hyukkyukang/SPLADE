use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

mod basic;
mod encode;
mod wordpiece;

fn load_vocab(vocab_file: &str) -> PyResult<HashMap<String, usize>> {
    let file = File::open(vocab_file)
        .map_err(|error| PyErr::new::<PyIOError, _>(format!("Failed to open vocab file: {error}")))?;
    let reader = BufReader::new(file);
    let mut vocab: HashMap<String, usize> = HashMap::new();
    for (index, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|error| {
            PyErr::new::<PyIOError, _>(format!("Failed to read vocab file line: {error}"))
        })?;
        let token = line.trim().to_string();
        if token.is_empty() {
            continue;
        }
        vocab.insert(token, index);
    }
    Ok(vocab)
}

#[pyclass]
pub struct AnnaFastBackend {
    vocab: HashMap<String, usize>,
    do_lower_case: bool,
    max_input_chars_per_word: usize,
    unk_token: String,
}

#[pymethods]
impl AnnaFastBackend {
    #[new]
    #[pyo3(signature=(vocab_file, do_lower_case=true, max_input_chars_per_word=200, unk_token="[UNK]".to_string()))]
    fn new(
        vocab_file: String,
        do_lower_case: bool,
        max_input_chars_per_word: usize,
        unk_token: String,
    ) -> PyResult<Self> {
        let vocab = load_vocab(&vocab_file)?;
        Ok(Self {
            vocab,
            do_lower_case,
            max_input_chars_per_word,
            unk_token,
        })
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        wordpiece::tokenize_text(
            text,
            &self.vocab,
            self.do_lower_case,
            &self.unk_token,
            self.max_input_chars_per_word,
        )
    }

    fn tokenize_batch(&self, texts: Vec<String>) -> Vec<Vec<String>> {
        encode::encode_batch(
            &texts,
            &self.vocab,
            self.do_lower_case,
            &self.unk_token,
            self.max_input_chars_per_word,
        )
    }

    fn normalize_for_anna(&self, text: &str) -> String {
        let _ = self;
        basic::normalize_for_anna(text)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn backend_version(&self) -> &'static str {
        let _ = self;
        "0.2.0"
    }
}

#[pyfunction]
fn backend_version() -> &'static str {
    "0.2.0"
}

#[pymodule]
fn anna_fast_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AnnaFastBackend>()?;
    m.add_function(wrap_pyfunction!(backend_version, m)?)?;
    Ok(())
}
