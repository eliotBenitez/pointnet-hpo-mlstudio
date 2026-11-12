//! Окружение Python, как в run_all.cmd/.ps1: Python 3.11, .venv311, сборка torch под GPU/CPU, зависимости.
//! Всё здесь блокирующее (запуск процессов) — вызывать из фонового потока.
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const VENV: &str = ".venv311";
pub const TORCH: &str = "2.8.0";
/// Модули, которые импортирует hpo_pointnet.py.
const MODULES: [&str; 6] = ["torch", "pyg_lib", "torch_geometric", "torch_pointcloud", "optuna", "numpy"];

pub struct Gpu {
    pub name: String,
    /// Максимальная версия CUDA, которую поддерживает драйвер (строка «CUDA Version» в nvidia-smi).
    pub cuda: Option<(u32, u32)>,
}

pub struct Packages {
    /// Версия по модулю; None — не импортируется.
    pub versions: Vec<(&'static str, Option<String>)>,
    pub cuda_available: bool,
}

impl Packages {
    pub fn missing(&self) -> Vec<&'static str> {
        self.versions.iter().filter(|v| v.1.is_none()).map(|v| v.0).collect()
    }
    pub fn torch(&self) -> Option<&str> {
        self.versions.iter().find(|v| v.0 == "torch")?.1.as_deref()
    }
}

pub struct EnvInfo {
    /// Python 3.11 для создания окружения: (путь, версия).
    pub base: Option<(PathBuf, String)>,
    pub venv: Option<PathBuf>,
    pub gpu: Option<Gpu>,
    /// Проверка импорта в .venv311 (Err — текст ошибки).
    pub packages: Option<Result<Packages, String>>,
    pub uv: bool,
    pub winget: bool,
}

impl EnvInfo {
    /// Сборка torch по драйверу: колёса torch 2.8.0 есть для cu126 и cu128.
    pub fn recommended_build(&self) -> &'static str {
        match self.gpu.as_ref().and_then(|g| g.cuda) {
            Some(v) if v >= (12, 8) => "cu128",
            Some(v) if v >= (12, 6) => "cu126",
            _ => "cpu",
        }
    }
}

pub fn venv_python(root: &Path) -> PathBuf {
    let win = root.join(VENV).join("Scripts").join("python.exe");
    if cfg!(windows) || win.exists() { win } else { root.join(VENV).join("bin").join("python") }
}

/// Запуск без окна консоли; stdout при коде 0.
pub fn run(program: &str, args: &[&str], cwd: &Path) -> Option<String> {
    let mut cmd = Command::new(program);
    cmd.args(args).current_dir(cwd).stdin(Stdio::null()).stderr(Stdio::null()).env("PYTHONIOENCODING", "utf-8");
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut cmd, 0x0800_0000);
    let out = cmd.output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

const WHO: &str = "import sys; print(sys.executable); print('%d.%d.%d' % sys.version_info[:3])";

/// Интерпретатор и его версия, если это 3.11 (pyg-lib не собран под 3.12+).
fn probe(program: &str, pre: &[&str], cwd: &Path) -> Option<(PathBuf, String)> {
    let args: Vec<&str> = pre.iter().copied().chain(["-c", WHO]).collect();
    let out = run(program, &args, cwd)?;
    let mut lines = out.lines();
    let (path, ver) = (lines.next()?.trim(), lines.next()?.trim());
    ver.starts_with("3.11.").then(|| (PathBuf::from(path), ver.to_string()))
}

/// Пути 3.11 из `py -0p`: строки вида ` -V:Astral/CPython3.11.17   D:\…\python.exe` — тег не всегда «3.11», поэтому `py -3.11` их не находит.
fn py_launcher_paths(out: &str) -> Vec<String> {
    out.lines().filter(|l| l.contains("3.11")).filter_map(|l| l.split_whitespace().last().map(String::from)).collect()
}

/// Интерпретатор, из которого создано .venv311 (pyvenv.cfg: executable = …).
fn venv_base(root: &Path) -> Option<String> {
    let cfg = std::fs::read_to_string(root.join(VENV).join("pyvenv.cfg")).ok()?;
    cfg.lines().find_map(|l| Some(l.strip_prefix("executable")?.trim_start().strip_prefix('=')?.trim().to_string()))
}

pub fn find_python311(root: &Path) -> Option<(PathBuf, String)> {
    let uv = run("uv", &["python", "find", "3.11"], root).map(|s| s.trim().to_string());
    let local = std::env::var("LOCALAPPDATA").ok().map(|d| format!("{d}\\Programs\\Python\\Python311\\python.exe"));
    let launcher = run("py", &["-0p"], root).map(|o| py_launcher_paths(&o)).unwrap_or_default();
    launcher
        .iter()
        .find_map(|p| probe(p, &[], root))
        .or_else(|| probe("py", &["-3.11"], root))
        .or_else(|| venv_base(root).and_then(|p| probe(&p, &[], root)))
        .or_else(|| uv.as_deref().and_then(|p| probe(p, &[], root)))
        .or_else(|| local.as_deref().and_then(|p| probe(p, &[], root)))
        .or_else(|| probe("python3.11", &[], root))
        .or_else(|| probe("python", &[], root))
        .or_else(|| probe("python3", &[], root))
}

fn parse_cuda(smi: &str) -> Option<(u32, u32)> {
    let rest = &smi[smi.find("CUDA Version:")? + 13..];
    let (major, minor) = rest.split_whitespace().next()?.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

fn find_gpu(root: &Path) -> Option<Gpu> {
    let name = run("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"], root)?;
    Some(Gpu {
        name: name.lines().next().unwrap_or("NVIDIA").trim().to_string(),
        cuda: run("nvidia-smi", &[], root).as_deref().and_then(parse_cuda),
    })
}

const CHECK: &str = r#"
import importlib, json
out = {}
for m in MODULES:
    try:
        out[m] = str(getattr(importlib.import_module(m), "__version__", "?"))
    except Exception:
        out[m] = None
try:
    import torch; out["cuda"] = bool(torch.cuda.is_available())
except Exception:
    out["cuda"] = False
print(json.dumps(out))
"#;

fn check_packages(py: &Path, root: &Path) -> Result<Packages, String> {
    let code = CHECK.replace("MODULES", &format!("{MODULES:?}"));
    let out = run(&py.to_string_lossy(), &["-c", &code], root).ok_or("python из .venv311 не запускается")?;
    let v: serde_json::Value = serde_json::from_str(out.lines().last().unwrap_or("")).map_err(|e| e.to_string())?;
    Ok(Packages {
        versions: MODULES.iter().map(|m| (*m, v.get(m).and_then(|x| x.as_str()).map(String::from))).collect(),
        cuda_available: v["cuda"].as_bool().unwrap_or(false),
    })
}

pub fn detect(root: &Path) -> EnvInfo {
    let venv = Some(venv_python(root)).filter(|p| p.exists());
    EnvInfo {
        packages: venv.as_ref().map(|p| check_packages(p, root)),
        base: find_python311(root),
        venv,
        gpu: find_gpu(root),
        uv: run("uv", &["--version"], root).is_some(),
        winget: cfg!(windows) && run("winget", &["--version"], root).is_some(),
    }
}

/// Команда установки Python 3.11 (uv — без прав администратора; иначе winget для текущего пользователя).
pub fn install_python_cmd(info: &EnvInfo) -> Option<(String, Vec<String>)> {
    let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect();
    if info.uv {
        Some(("uv".into(), v(&["python", "install", "3.11"])))
    } else if info.winget {
        Some((
            "winget".into(),
            v(&["install", "-e", "--id", "Python.Python.3.11", "--scope", "user", "--silent", "--accept-package-agreements", "--accept-source-agreements"]),
        ))
    } else {
        None
    }
}

/// Шаги установки зависимостей (как в run_all.cmd); torch и pyg-lib — одной сборки.
pub fn setup_steps(root: &Path, base: Option<&Path>, venv_exists: bool, build: &str) -> Vec<Vec<String>> {
    let py = venv_python(root).to_string_lossy().into_owned();
    let s = |a: &[&str]| a.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let mut steps = vec![];
    if !venv_exists {
        if let Some(b) = base {
            steps.push(s(&[&b.to_string_lossy(), "-m", "venv", VENV]));
        }
    }
    steps.push(s(&[&py, "-m", "pip", "install", "--upgrade", "pip"]));
    // Локальная версия (+cpu/+cu128) обязательна: иначе pip сочтёт установленную сборку подходящей и не заменит её.
    steps.push(s(&[&py, "-m", "pip", "install", &format!("torch=={TORCH}+{build}"), "--index-url", &format!("https://download.pytorch.org/whl/{build}")]));
    steps.push(s(&[&py, "-m", "pip", "install", "--force-reinstall", "--no-deps", "pyg-lib", "-f", &format!("https://data.pyg.org/whl/torch-{TORCH}+{build}.html")]));
    steps.push(s(&[&py, "-m", "pip", "install", "torch-geometric", "torch-pointcloud==0.0.16", "optuna", "numpy"]));
    steps
}

/// Python-скрипт, выполняющий шаги по очереди одним процессом (останавливается на первой ошибке).
pub const RUNNER: &str = r#"
import json, subprocess, sys
for step in json.loads(sys.argv[1]):
    print("=== " + " ".join(step), flush=True)
    code = subprocess.call(step)
    if code:
        print("=== шаг завершился с кодом %d" % code, flush=True)
        sys.exit(code)
print("=== окружение готово", flush=True)
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_version_and_build() {
        let smi = "| NVIDIA-SMI 570.86   Driver Version: 570.86   CUDA Version: 12.8     |";
        assert_eq!(parse_cuda(smi), Some((12, 8)));
        let info = |cuda| EnvInfo { base: None, venv: None, gpu: Some(Gpu { name: "x".into(), cuda }), packages: None, uv: false, winget: false };
        assert_eq!(info(Some((12, 9))).recommended_build(), "cu128");
        assert_eq!(info(Some((12, 6))).recommended_build(), "cu126");
        assert_eq!(info(Some((11, 8))).recommended_build(), "cpu");
        assert_eq!(info(None).recommended_build(), "cpu");
    }

    #[test]
    fn py_launcher_lines() {
        let out = " -V:3.14 *        C:\\Python314\\python.exe\n -V:Astral/CPython3.11.17 D:\\a\\uv_python\\cpython-3.11.17\\python.exe\n";
        assert_eq!(py_launcher_paths(out), vec!["D:\\a\\uv_python\\cpython-3.11.17\\python.exe".to_string()]);
    }

    #[test]
    fn steps_create_venv_only_when_missing() {
        let root = Path::new("R");
        let base = PathBuf::from("py311");
        assert_eq!(setup_steps(root, Some(&base), false, "cu128").len(), 5);
        let steps = setup_steps(root, Some(&base), true, "cpu");
        assert_eq!(steps.len(), 4);
        assert!(steps[1].contains(&"torch==2.8.0+cpu".to_string()));
    }
}
