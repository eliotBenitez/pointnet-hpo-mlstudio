//! Свои сценарии: последовательность запусков, как run_all.py, но собранная в «Запуске».
//! Файл scenarios/<имя>.json в корне проекта — им можно поделиться с командой.
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};

pub const DIR: &str = "scenarios";

#[derive(Clone)]
pub struct Step {
    pub title: String,
    /// Аргументы Python без самого интерпретатора: интерпретатор берётся из «Запуска» у того, кто запускает.
    pub args: Vec<String>,
    /// Своя конфигурация: пишется в файл --fixed-params перед запуском шага.
    pub fixed_params: Value,
}

pub struct Scenario {
    pub name: String,
    pub steps: Vec<Step>,
}

fn file(root: &Path, name: &str) -> PathBuf {
    root.join(DIR).join(format!("{name}.json"))
}

pub fn scan(root: &Path) -> Vec<Scenario> {
    let mut out: Vec<Scenario> = fs::read_dir(root.join(DIR))
        .map(|r| r.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "json")).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|p| {
            let v: Value = serde_json::from_str(&fs::read_to_string(&p).ok()?).ok()?;
            let steps = v["steps"]
                .as_array()?
                .iter()
                .map(|s| Step {
                    title: s["title"].as_str().unwrap_or_default().to_string(),
                    args: serde_json::from_value(s["args"].clone()).unwrap_or_default(),
                    fixed_params: s["fixed_params"].clone(),
                })
                .collect();
            Some(Scenario { name: p.file_stem()?.to_string_lossy().into_owned(), steps })
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

impl Scenario {
    pub fn save(&self, root: &Path) -> std::io::Result<()> {
        let steps: Vec<Value> = self.steps.iter().map(|s| json!({ "title": s.title, "args": s.args, "fixed_params": s.fixed_params })).collect();
        fs::create_dir_all(root.join(DIR))?;
        fs::write(file(root, &self.name), serde_json::to_vec_pretty(&json!({ "steps": steps })).unwrap())
    }

    /// Переименование = новый файл + удаление старого; имя не должно быть занято.
    pub fn rename(&mut self, root: &Path, to: &str) -> Result<(), String> {
        let to = to.trim();
        if to.is_empty() || to.contains(['/', '\\', ':', '*', '?', '"', '<', '>', '|']) {
            return Err("Имя не может быть пустым или содержать / \\ : * ? \" < > |".into());
        }
        if to == self.name {
            return Ok(());
        }
        if file(root, to).exists() {
            return Err(format!("Сценарий «{to}» уже есть"));
        }
        let old = file(root, &self.name);
        self.name = to.to_string();
        self.save(root).map_err(|e| e.to_string())?;
        _ = fs::remove_file(old);
        Ok(())
    }

    pub fn delete(&self, root: &Path) -> Result<(), String> {
        trash::delete(file(root, &self.name)).map_err(|e| e.to_string())
    }
}

/// Свободное имя «сценарий N».
pub fn new_name(root: &Path) -> String {
    (1..).map(|n| format!("сценарий {n}")).find(|n| !file(root, n).exists()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_scan_rename() {
        let root = std::env::temp_dir().join(format!("mlstudio-scn-{}", std::process::id()));
        _ = fs::remove_dir_all(&root);
        let mut s = Scenario { name: new_name(&root), steps: vec![Step { title: "a".into(), args: vec!["-u".into(), "hpo_pointnet.py".into()], fixed_params: json!({"lr": 0.001}) }] };
        assert_eq!(s.name, "сценарий 1");
        s.save(&root).unwrap();
        assert_eq!(new_name(&root), "сценарий 2");
        s.rename(&root, "ночь").unwrap();
        assert!(s.rename(&root, "a/b").is_err());
        let all = scan(&root);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].name, "ночь");
        assert_eq!(all[0].steps[0].args[1], "hpo_pointnet.py");
        assert_eq!(all[0].steps[0].fixed_params["lr"], 0.001);
        fs::remove_dir_all(&root).unwrap();
    }
}
