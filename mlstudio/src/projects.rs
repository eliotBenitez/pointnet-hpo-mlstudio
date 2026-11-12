//! Недавние и избранные проекты, как в IDE. Хранятся в конфиге пользователя, а не рядом с .exe.
use std::path::{Path, PathBuf};

const MAX_RECENT: usize = 12;

#[derive(Default)]
pub struct Projects {
    pub recent: Vec<PathBuf>,
    pub favorites: Vec<PathBuf>,
}

/// Проект — каталог с hpo_pointnet.py.
pub fn is_project(p: &Path) -> bool {
    p.join("hpo_pointnet.py").is_file()
}

/// %APPDATA%\mlstudio\projects.json или $XDG_CONFIG_HOME (~/.config)/mlstudio/projects.json.
fn file() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA")
        .or_else(|| std::env::var_os("XDG_CONFIG_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("mlstudio").join("projects.json"))
}

impl Projects {
    pub fn load() -> Self {
        let Some(v) = file().and_then(|f| std::fs::read(f).ok()).and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok()) else {
            return Self::default();
        };
        let list = |k: &str| v[k].as_array().map(|a| a.iter().filter_map(|s| s.as_str()).map(PathBuf::from).collect()).unwrap_or_default();
        Self { recent: list("recent"), favorites: list("favorites") }
    }

    /// Ошибка записи не мешает работе — список просто не сохранится.
    pub fn save(&self) {
        let Some(f) = file() else { return };
        let json = serde_json::json!({ "recent": self.recent, "favorites": self.favorites });
        _ = std::fs::create_dir_all(f.parent().unwrap()).and_then(|_| std::fs::write(f, serde_json::to_vec_pretty(&json).unwrap()));
    }

    /// Поднимает проект в начало недавних.
    pub fn touch(&mut self, root: &Path) {
        self.recent.retain(|p| p != root);
        self.recent.insert(0, root.to_path_buf());
        self.recent.truncate(MAX_RECENT);
    }

    pub fn is_favorite(&self, root: &Path) -> bool {
        self.favorites.iter().any(|p| p == root)
    }

    pub fn toggle_favorite(&mut self, root: &Path) {
        if self.is_favorite(root) {
            self.favorites.retain(|p| p != root);
        } else {
            self.favorites.push(root.to_path_buf());
        }
    }

    pub fn forget(&mut self, root: &Path) {
        self.recent.retain(|p| p != root);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_dedupes_and_caps() {
        let mut p = Projects::default();
        for i in 0..20 {
            p.touch(Path::new(&format!("/p{i}")));
        }
        p.touch(Path::new("/p5"));
        assert_eq!(p.recent.len(), MAX_RECENT);
        assert_eq!(p.recent[0], Path::new("/p5"));
        assert_eq!(p.recent.iter().filter(|x| *x == Path::new("/p5")).count(), 1);
        p.toggle_favorite(Path::new("/a"));
        assert!(p.is_favorite(Path::new("/a")));
        p.toggle_favorite(Path::new("/a"));
        assert!(!p.is_favorite(Path::new("/a")));
    }
}
