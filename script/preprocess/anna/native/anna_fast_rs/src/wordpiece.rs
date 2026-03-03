use std::collections::HashMap;

use crate::basic::{
    clean_text, split_on_punc, strip_accents, tokenize_chinese_chars, whitespace_tokenize,
};

fn find_punc_vocab(split_punc: &[String], vocab: &HashMap<String, usize>) -> Vec<String> {
    let mut re_output: Vec<String> = Vec::new();
    let mut i = 0usize;
    let n = split_punc.len();
    while i < n {
        let mut matched = false;
        for ii in 0..n {
            let end = n - ii;
            let chunk = if i < end {
                split_punc[i..end].concat()
            } else {
                String::new()
            };
            if vocab.contains_key(&chunk) || chunk == split_punc[i] {
                re_output.push(chunk);
                i = end;
                matched = true;
                break;
            }
        }
        if !matched {
            i += 1;
        }
    }
    re_output
}

fn basic_tokenize(
    text: &str,
    vocab: &HashMap<String, usize>,
    do_lower_case: bool,
) -> Vec<String> {
    let cleaned = clean_text(text);
    let with_cjk = tokenize_chinese_chars(&cleaned);
    let orig_tokens = whitespace_tokenize(&with_cjk);
    let mut split_tokens: Vec<String> = Vec::new();

    for token in orig_tokens {
        let normalized = if do_lower_case {
            strip_accents(&token.to_lowercase())
        } else {
            token
        };
        if vocab.contains_key(&normalized) {
            split_tokens.push(normalized);
        } else {
            let split_punc = split_on_punc(&normalized);
            split_tokens.extend(find_punc_vocab(&split_punc, vocab));
        }
    }

    whitespace_tokenize(&split_tokens.join(" "))
}

fn tokenize_wordpiece_single(
    token: &str,
    vocab: &HashMap<String, usize>,
    unk_token: &str,
    max_input_chars_per_word: usize,
) -> Vec<String> {
    let chars: Vec<char> = token.chars().collect();
    if chars.len() > max_input_chars_per_word {
        return vec![unk_token.to_string()];
    }
    let mut is_bad = false;
    let mut start = 0usize;
    let mut sub_tokens: Vec<String> = Vec::new();
    while start < chars.len() {
        let mut end = chars.len();
        let mut cur_substr: Option<String> = None;
        while start < end {
            let mut substr: String = chars[start..end].iter().collect();
            if start > 0 {
                substr = format!("##{}", substr);
            }
            if vocab.contains_key(&substr) {
                cur_substr = Some(substr);
                break;
            }
            end -= 1;
        }
        if cur_substr.is_none() {
            is_bad = true;
            break;
        }
        if let Some(sub) = cur_substr {
            sub_tokens.push(sub);
        }
        start = end;
    }
    if is_bad {
        vec![unk_token.to_string()]
    } else {
        sub_tokens
    }
}

pub fn tokenize_text(
    text: &str,
    vocab: &HashMap<String, usize>,
    do_lower_case: bool,
    unk_token: &str,
    max_input_chars_per_word: usize,
) -> Vec<String> {
    let basic_tokens = basic_tokenize(text, vocab, do_lower_case);
    let mut output_tokens: Vec<String> = Vec::new();
    for token in basic_tokens {
        output_tokens.extend(tokenize_wordpiece_single(
            &token,
            vocab,
            unk_token,
            max_input_chars_per_word,
        ));
    }
    output_tokens
}
