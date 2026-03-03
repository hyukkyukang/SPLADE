use std::collections::HashMap;

use crate::wordpiece::tokenize_text;

pub fn encode_batch(
    texts: &[String],
    vocab: &HashMap<String, usize>,
    do_lower_case: bool,
    unk_token: &str,
    max_input_chars_per_word: usize,
) -> Vec<Vec<String>> {
    texts
        .iter()
        .map(|text| {
            tokenize_text(
                text,
                vocab,
                do_lower_case,
                unk_token,
                max_input_chars_per_word,
            )
        })
        .collect()
}
