#[derive(Debug, PartialEq, Eq)]
pub enum Block {
    Heading { level: u8, text: String },
    Paragraph(String),
    List { ordered: bool, start: usize, items: Vec<String> },
    Table { headers: Vec<String>, rows: Vec<Vec<String>> },
    Code { language: String, text: String },
}

fn heading(line: &str) -> Option<(u8, &str)> {
    let line = line.trim_start();
    let count = line.bytes().take_while(|b| *b == b'#').count();
    if (1..=6).contains(&count) && line.as_bytes().get(count).is_some_and(|b| b.is_ascii_whitespace()) {
        Some((count as u8, line[count..].trim()))
    } else {
        None
    }
}

/// (упорядоченный?, номер в тексте, содержимое)
fn list_item(line: &str) -> Option<(bool, usize, &str)> {
    let line = line.trim_start();
    if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")).or_else(|| line.strip_prefix("+ ")) {
        return Some((false, 0, rest));
    }
    let digits = line.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits > 0 {
        let rest = &line[digits..];
        if let Some(rest) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
            return Some((true, line[..digits].parse().unwrap_or(1), rest));
        }
    }
    None
}

/// Ширина отступа строки (таб = 4).
fn indent(line: &str) -> usize {
    line.chars().take_while(|c| c.is_whitespace()).map(|c| if c == '\t' { 4 } else { 1 }).sum()
}

fn cells(line: &str) -> Vec<String> {
    let line = line.trim().strip_prefix('|').unwrap_or(line.trim());
    let line = line.strip_suffix('|').unwrap_or(line);
    line.split('|').map(|cell| cell.trim().to_owned()).collect()
}

fn table_separator(line: &str) -> bool {
    line.contains('|') && cells(line).iter().all(|cell| {
        let cell = cell.trim_matches(':');
        cell.len() >= 3 && cell.bytes().all(|b| b == b'-')
    })
}

fn fence(line: &str) -> Option<(u8, usize, &str)> {
    let line = line.trim_start();
    let marker = *line.as_bytes().first()?;
    if marker != b'`' && marker != b'~' { return None; }
    let count = line.bytes().take_while(|b| *b == marker).count();
    (count >= 3).then_some((marker, count, &line[count..]))
}

pub fn parse_markdown(input: &str) -> Vec<Block> {
    let lines: Vec<&str> = input.split_inclusive('\n').collect();
    fn content(line: &str) -> &str { line.trim_end_matches(['\r', '\n']) }
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = content(lines[i]);
        if line.trim().is_empty() { i += 1; continue; }
        if let Some((marker, count, info)) = fence(line) {
            let language = info.trim().to_owned();
            let mut text = String::new();
            i += 1;
            while i < lines.len() {
                if let Some((end_marker, end_count, rest)) = fence(content(lines[i])) {
                    if marker == end_marker && end_count >= count && rest.trim().is_empty() {
                        i += 1;
                        break;
                    }
                }
                text.push_str(lines[i]);
                i += 1;
            }
            blocks.push(Block::Code { language, text });
        } else if let Some((level, text)) = heading(line) {
            blocks.push(Block::Heading { level, text: text.to_owned() });
            i += 1;
        } else if i + 1 < lines.len() && line.contains('|') && table_separator(content(lines[i + 1])) && cells(line).len() == cells(content(lines[i + 1])).len() {
            let headers = cells(line);
            let mut rows = Vec::new();
            i += 2;
            while i < lines.len() && !content(lines[i]).trim().is_empty() && content(lines[i]).contains('|') {
                rows.push(cells(content(lines[i])));
                i += 1;
            }
            blocks.push(Block::Table { headers, rows });
        } else if let Some((ordered, start, _)) = list_item(line) {
            let base = indent(line);
            let mut items: Vec<String> = Vec::new();
            while i < lines.len() {
                // пустые строки между пунктами одного списка его не разрывают
                let mut j = i;
                while j < lines.len() && content(lines[j]).trim().is_empty() { j += 1; }
                let Some(next) = lines.get(j).map(|l| content(l)) else { break };
                let ind = indent(next);
                match list_item(next) {
                    // вложенный пункт любого вида (отступ = неразрывные пробелы) или пункт того же вида
                    Some((kind, _, item)) if ind > base || kind == ordered => {
                        let pad = "\u{a0}".repeat(ind.saturating_sub(base));
                        items.push(format!("{pad}{item}"));
                    }
                    // продолжение пункта: непустая строка с отступом без маркера
                    None if j == i && ind > base => {
                        if let Some(last) = items.last_mut() {
                            last.push(' ');
                            last.push_str(next.trim());
                        }
                    }
                    _ => break,
                }
                i = j + 1;
            }
            blocks.push(Block::List { ordered, start, items });
        } else {
            let mut paragraph = vec![line.to_owned()];
            i += 1;
            while i < lines.len() {
                let next = content(lines[i]);
                if next.trim().is_empty() || heading(next).is_some() || fence(next).is_some() || list_item(next).is_some() || (i + 1 < lines.len() && next.contains('|') && table_separator(content(lines[i + 1]))) { break; }
                paragraph.push(next.to_owned());
                i += 1;
            }
            blocks.push(Block::Paragraph(paragraph.join("\n")));
        }
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_unclosed_fences_ordered_lists_and_crlf() {
        assert_eq!(parse_markdown("1. первый\r\n2. второй\r\n\r\n~~~txt\r\n# code\r\n"), vec![
            Block::List { ordered: true, start: 1, items: vec!["первый".into(), "второй".into()] },
            Block::Code { language: "txt".into(), text: "# code\r\n".into() },
        ]);
    }

    #[test]
    fn nested_items_keep_indent_and_continuations_join() {
        assert_eq!(parse_markdown("- a\n  - b\n  продолжение\n- c\n"), vec![
            Block::List { ordered: false, start: 0, items: vec!["a".into(), "\u{a0}\u{a0}b продолжение".into(), "c".into()] },
        ]);
    }

    #[test]
    fn list_then_paragraph_stays_separate() {
        assert_eq!(parse_markdown("- a\n\nтекст\n"), vec![
            Block::List { ordered: false, start: 0, items: vec!["a".into()] },
            Block::Paragraph("текст".into()),
        ]);
    }

    #[test]
    fn keeps_start_number_and_loose_list_items() {
        assert_eq!(parse_markdown("3. a

4. b
"), vec![
            Block::List { ordered: true, start: 3, items: vec!["a".into(), "b".into()] },
        ]);
    }

    #[test]
    fn parses_blocks_and_keeps_code_literal() {
        assert_eq!(parse_markdown("# Заголовок\n\nТекст\nстрока\n\n- один\n- два\n\n| A | B |\n| --- | :---: |\n| x | y |\n\n```rust\n# не заголовок\n```"), vec![
            Block::Heading { level: 1, text: "Заголовок".into() },
            Block::Paragraph("Текст\nстрока".into()),
            Block::List { ordered: false, start: 0, items: vec!["один".into(), "два".into()] },
            Block::Table { headers: vec!["A".into(), "B".into()], rows: vec![vec!["x".into(), "y".into()]] },
            Block::Code { language: "rust".into(), text: "# не заголовок\n".into() },
        ]);
    }
}
