use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::{char::canonical_combining_class, UnicodeNormalization};

pub fn is_whitespace(ch: char) -> bool {
    if matches!(ch, ' ' | '\t' | '\n' | '\r') {
        return true;
    }
    get_general_category(ch) == GeneralCategory::SpaceSeparator
}

pub fn is_control(ch: char) -> bool {
    if matches!(ch, '\t' | '\n' | '\r') {
        return false;
    }
    matches!(
        get_general_category(ch),
        GeneralCategory::Control | GeneralCategory::Format
    )
}

pub fn is_punctuation(ch: char) -> bool {
    let cp = ch as u32;
    if (33..=47).contains(&cp)
        || (58..=64).contains(&cp)
        || (91..=96).contains(&cp)
        || (123..=126).contains(&cp)
    {
        return true;
    }
    matches!(
        get_general_category(ch),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::OtherPunctuation
            | GeneralCategory::OpenPunctuation
    )
}

pub fn is_chinese_char(cp: u32) -> bool {
    if (0x4E00..=0x9FFF).contains(&cp) || (0x3400..=0x4DBF).contains(&cp) {
        return true;
    }
    if (0x20000..=0x2A6DF).contains(&cp) || (0x2A700..=0x2B73F).contains(&cp) {
        return true;
    }
    if (0x2B740..=0x2B81F).contains(&cp) || (0x2B820..=0x2CEAF).contains(&cp) {
        return true;
    }
    if (0xF900..=0xFAFF).contains(&cp) || (0x2F800..=0x2FA1F).contains(&cp) {
        return true;
    }
    false
}

pub fn clean_text(text: &str) -> String {
    let mut output = String::new();
    for ch in text.chars() {
        let cp = ch as u32;
        if cp == 0 || cp == 0xFFFD || is_control(ch) {
            continue;
        }
        if is_whitespace(ch) {
            output.push(' ');
        } else {
            output.push(ch);
        }
    }
    output
}

pub fn tokenize_chinese_chars(text: &str) -> String {
    let mut output = String::new();
    for ch in text.chars() {
        let cp = ch as u32;
        if is_chinese_char(cp) {
            output.push(' ');
            output.push(ch);
            output.push(' ');
        } else {
            output.push(ch);
        }
    }
    output
}

pub fn strip_accents(text: &str) -> String {
    text.nfd()
        .filter(|ch| canonical_combining_class(*ch) == 0)
        .collect()
}

pub fn split_on_punc(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    let mut start_new_word = true;
    let mut output: Vec<Vec<char>> = Vec::new();
    while i < chars.len() {
        let ch = chars[i];
        if is_punctuation(ch) {
            output.push(vec![ch]);
            start_new_word = true;
        } else {
            if start_new_word {
                output.push(Vec::new());
            }
            start_new_word = false;
            if let Some(last) = output.last_mut() {
                last.push(ch);
            }
        }
        i += 1;
    }
    output
        .into_iter()
        .map(|token_chars| token_chars.into_iter().collect())
        .collect()
}

pub fn whitespace_tokenize(text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed
        .split_whitespace()
        .map(|token| token.to_string())
        .collect()
}

pub fn normalize_for_anna(text: &str) -> String {
    let cleaned = clean_text(text);
    tokenize_chinese_chars(&cleaned)
}
