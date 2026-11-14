//! Экспорт Markdown-отчёта в Nevo через его MCP-мост (`nevo.exe --mcp-server`, построчный JSON-RPC по stdio).
//! Файлы хранилища Nevo напрямую не трогаем: там свой формат и индексы.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

/// nevo.exe: переменная NEVO_EXE, затем типовые места установки, иначе поиск в PATH.
fn nevo_exe() -> PathBuf {
    if let Some(p) = std::env::var_os("NEVO_EXE") {
        return p.into();
    }
    let mut candidates = vec![PathBuf::from(r"D:\Programs\Nevo\nevo.exe")];
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        candidates.insert(0, PathBuf::from(local).join(r"Programs\Nevo\nevo.exe"));
    }
    candidates.into_iter().find(|p| p.exists()).unwrap_or_else(|| "nevo".into())
}

/// Текст ответа инструмента или ошибка с сообщением от Nevo.
fn tool_text(resp: &Value) -> Result<String, String> {
    if let Some(e) = resp.get("error") {
        return Err(e.get("message").and_then(Value::as_str).unwrap_or("ошибка MCP").to_string());
    }
    let result = &resp["result"];
    let text = result["content"]
        .as_array()
        .and_then(|c| c.iter().find_map(|b| b["text"].as_str()))
        .unwrap_or("")
        .to_string();
    if result["isError"].as_bool() == Some(true) {
        return Err(if text.is_empty() { "Nevo вернул ошибку".into() } else { text });
    }
    Ok(text)
}

/// id новой заметки из ответа nevo_create_note: JSON с полем id/noteId либо голый текст.
fn note_id(text: &str) -> Option<String> {
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        let v = v.get("note").unwrap_or(&v);
        return ["noteId", "id"].iter().find_map(|k| v[k].as_str()).map(str::to_string);
    }
    let t = text.trim();
    (!t.is_empty() && !t.contains(char::is_whitespace)).then(|| t.to_string())
}

/// Создаёт заметку `title` с телом `markdown`. Возвращает id заметки.
pub fn export(title: &str, markdown: &str) -> Result<String, String> {
    let exe = nevo_exe();
    let mut cmd = Command::new(&exe);
    cmd.arg("--mcp-server").env("NEVO_MCP_MANAGED", "1").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null());
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut cmd, 0x0800_0000); // CREATE_NO_WINDOW
    let mut child = cmd.spawn().map_err(|e| format!("Не удалось запустить {}: {e}", exe.display()))?;
    let mut stdin = child.stdin.take().ok_or("нет stdin")?;
    let mut lines = BufReader::new(child.stdout.take().ok_or("нет stdout")?).lines();

    // shortcut: без таймаута — если мост завис, зависает только фоновая задача экспорта.
    let mut call = |id: u64, method: &str, params: Value| -> Result<Value, String> {
        let msg = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        writeln!(stdin, "{msg}").and_then(|_| stdin.flush()).map_err(|e| format!("Мост Nevo закрылся: {e}"))?;
        for line in lines.by_ref() {
            let line = line.map_err(|e| e.to_string())?;
            let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
            if v["id"].as_u64() == Some(id) {
                return Ok(v);
            }
        }
        Err("Мост Nevo завершился без ответа".into())
    };

    let result = (|| {
        tool_text(&call(1, "initialize", json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "mlstudio", "version": env!("CARGO_PKG_VERSION")}
        }))?)?;
        let create = |name: &str, args: Value| json!({"name": name, "arguments": args});
        let text = tool_text(&call(2, "tools/call", create("nevo_create_note", json!({"title": title, "icon": "📊"})))?)?;
        let id = note_id(&text).ok_or_else(|| format!("Nevo не вернул id заметки: {text}"))?;
        tool_text(&call(3, "tools/call", create("nevo_write_markdown", json!({"noteId": id, "mode": "append", "markdown": markdown})))?)?;
        Ok(id)
    })();
    drop(call);
    drop(stdin);
    _ = child.kill();
    _ = child.wait();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_results() {
        let ok = json!({"result": {"content": [{"type": "text", "text": "{\"id\":\"abc\"}"}], "isError": false}});
        assert_eq!(tool_text(&ok).unwrap(), "{\"id\":\"abc\"}");
        let bad = json!({"result": {"content": [{"type": "text", "text": "Nevo is not running"}], "isError": true}});
        assert_eq!(tool_text(&bad).unwrap_err(), "Nevo is not running");
        assert_eq!(tool_text(&json!({"error": {"message": "x"}})).unwrap_err(), "x");
    }

    #[test]
    fn parses_note_id() {
        assert_eq!(note_id("{\"id\":\"abc\"}").as_deref(), Some("abc"));
        assert_eq!(note_id("{\"noteId\":\"n1\",\"title\":\"t\"}").as_deref(), Some("n1"));
        assert_eq!(note_id("{\"note\":{\"id\":\"n2\"}}").as_deref(), Some("n2"));
        assert_eq!(note_id(" 019d-uuid \n").as_deref(), Some("019d-uuid"));
        assert_eq!(note_id("Created note abc"), None);
    }

    /// Реальный экспорт: нужен запущенный Nevo. `cargo test nevo -- --ignored`
    #[test]
    #[ignore]
    fn exports_to_running_nevo() {
        let md = std::fs::read_to_string("../runs/ground_xyz/report.md").unwrap();
        let id = export("Отчёт · ground_xyz", &md).unwrap();
        assert!(!id.is_empty());
    }
}
