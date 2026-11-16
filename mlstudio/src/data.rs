//! Чтение файлов проекта: датасеты (*.npy), прогоны (runs/<name>/), разбор лога обучения.
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct Scene {
    pub name: String,
    pub bytes: u64,
    pub shape: Vec<u64>,
}

#[derive(Clone)]
pub struct Dataset {
    pub name: String,
    pub path: PathBuf,
    pub scenes: Vec<Scene>,
}

impl Dataset {
    pub fn bytes(&self) -> u64 {
        self.scenes.iter().map(|s| s.bytes).sum()
    }
    pub fn points(&self) -> u64 {
        self.scenes.iter().map(|s| s.shape.first().copied().unwrap_or(0)).sum()
    }
}

#[derive(Default)]
pub struct Trial {
    pub number: String,
    pub state: String,
    pub f1: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub seconds: Option<f64>,
    pub params: String,
    pub error: String,
    pub baseline_f1: Option<f64>,
}

pub struct Run {
    pub name: String,
    pub path: PathBuf,
    pub best: Option<serde_json::Value>,
    /// split.json пишется в начале прогона — источник класса и признаков, пока best.json нет.
    pub split: Option<serde_json::Value>,
    /// studio.json: кто и какой командой запускал (пишет ML Studio), заметка.
    pub studio: Option<serde_json::Value>,
    /// pred/scenes__trial<лучший>.json: метрики лучшей модели по сценам валидации (predict_scene.py --out-dir).
    pub scene_eval: Option<serde_json::Value>,
    pub trials: Vec<Trial>,
}

impl Run {
    pub fn best_f1(&self) -> Option<f64> {
        self.best.as_ref()?.get("f1")?.as_f64()
    }

    fn info(&self, key: &str) -> Option<String> {
        [&self.split, &self.best].into_iter().flatten().find_map(|v| v.get(key)?.as_str().map(String::from))
    }

    /// Версия пространства поиска из best.json (пишет hpo_pointnet.py с версии 7); None у старых прогонов.
    pub fn space_version(&self) -> Option<u32> {
        self.best.as_ref()?.get("search_space")?.as_u64().map(|v| v as u32)
    }

    /// Класс прогона: split.json → best.json; None, если неизвестен.
    pub fn target(&self) -> Option<String> {
        self.info("target")
    }

    pub fn features(&self) -> Option<String> {
        self.info("features")
    }

    /// Функция потерь: "ce" (и у прогонов, записанных до выбора loss) или "focal(gamma=…)".
    pub fn loss(&self) -> String {
        self.info("loss").unwrap_or_else(|| "ce".into())
    }

    /// Baseline F1: best.json, иначе любой trial (колонка trials.csv).
    pub fn baseline(&self) -> Option<f64> {
        self.best.as_ref().and_then(|b| b.get("baseline_f1")?.as_f64()).or_else(|| self.trials.iter().find_map(|t| t.baseline_f1))
    }
}

/// Версия пространства поиска, под которую написан SPACE (SEARCH_SPACE_VERSION в hpo_pointnet.py).
pub const SPACE_VERSION: u32 = 7;

/// SEARCH_SPACE_VERSION из hpo_pointnet.py корня проекта.
pub fn script_space_version(root: &Path) -> Option<u32> {
    let text = fs::read_to_string(root.join("hpo_pointnet.py")).ok()?;
    let line = text.lines().find(|l| l.trim_start().starts_with("SEARCH_SPACE_VERSION"))?;
    line.split('=').nth(1)?.split('#').next()?.trim().parse().ok()
}

// ---------- пространство гиперпараметров (зеркало run_trial в hpo_pointnet.py, SEARCH_SPACE_VERSION 7) ----------

pub type Params = serde_json::Map<String, serde_json::Value>;

pub enum Kind {
    LogFloat(f64, f64),
    Float(f64, f64),
    Int(i64, i64),
    /// Варианты как JSON-литералы: "2", "true".
    Choice(&'static [&'static str]),
}

pub struct Spec {
    pub key: &'static str,
    pub label: &'static str,
    pub hint: &'static str,
    pub kind: Kind,
}

pub const SPACE: [Spec; 9] = [
    Spec { key: "lr", label: "Learning rate", hint: "AdamW, лог. шкала", kind: Kind::LogFloat(1e-4, 1e-2) },
    Spec { key: "weight_decay", label: "Weight decay", hint: "AdamW (decoupled), лог. шкала", kind: Kind::LogFloat(1e-5, 1e-1) },
    Spec { key: "dropout", label: "Dropout", hint: "в голове сегментации", kind: Kind::Float(0.0, 0.5) },
    Spec { key: "batch_size", label: "Патчей в батче", hint: "≈16 тыс. точек на патч", kind: Kind::Choice(&["2", "3", "4"]) },
    Spec { key: "n_sa_layers", label: "SA-слоёв", hint: "глубина PointNet++", kind: Kind::Choice(&["2", "3", "4"]) },
    Spec { key: "n_points_l0", label: "Точек FPS в 1-м слое", hint: "", kind: Kind::Int(1024, 4096) },
    Spec { key: "radius_l0", label: "Радиус 1-го слоя", hint: "в долях радиуса патча", kind: Kind::Float(0.05, 0.5) },
    Spec { key: "nsample", label: "Соседей в шаре", hint: "", kind: Kind::Int(16, 64) },
    Spec { key: "use_MSG", label: "Multi-scale grouping", hint: "MSG вместо SSG", kind: Kind::Choice(&["false", "true"]) },
];

impl Spec {
    /// Текст для поля ввода — полная точность, чтобы конфигурация trial воспроизводилась точно.
    pub fn show(&self, v: &serde_json::Value) -> String {
        v.to_string()
    }

    /// Привести значение к пределам пространства (вариант вне списка → середина).
    fn clamp(&self, v: serde_json::Value) -> serde_json::Value {
        match (&self.kind, v.as_f64()) {
            (Kind::LogFloat(lo, hi) | Kind::Float(lo, hi), Some(x)) => x.clamp(*lo, *hi).into(),
            (Kind::Int(lo, hi), Some(x)) => (x.round() as i64).clamp(*lo, *hi).into(),
            _ => self.parse(&v.to_string()).unwrap_or_else(|_| self.middle()),
        }
    }

    /// Разбор и проверка введённого значения по пределам пространства поиска.
    pub fn parse(&self, s: &str) -> Result<serde_json::Value, String> {
        let s = s.trim();
        match self.kind {
            Kind::LogFloat(lo, hi) | Kind::Float(lo, hi) => match s.replace(',', ".").parse::<f64>() {
                Ok(x) if (lo..=hi).contains(&x) => Ok(x.into()),
                Ok(_) => Err(format!("от {lo} до {hi}")),
                Err(_) => Err("нужно число".into()),
            },
            Kind::Int(lo, hi) => match s.parse::<i64>() {
                Ok(x) if (lo..=hi).contains(&x) => Ok(x.into()),
                Ok(_) => Err(format!("целое от {lo} до {hi}")),
                Err(_) => Err("нужно целое число".into()),
            },
            Kind::Choice(opts) if opts.contains(&s) => Ok(serde_json::from_str(s).unwrap()),
            Kind::Choice(opts) => Err(format!("одно из {}", opts.join(", "))),
        }
    }

    pub fn range(&self) -> String {
        match self.kind {
            Kind::LogFloat(lo, hi) => format!("{lo:.0e} … {hi:.0e}"),
            Kind::Float(lo, hi) => format!("{lo} … {hi}"),
            Kind::Int(lo, hi) => format!("{lo} … {hi}"),
            Kind::Choice(o) => o.join(" / "),
        }
    }

    /// Середина пространства — значение, когда опереться не на что.
    pub fn middle(&self) -> serde_json::Value {
        match self.kind {
            Kind::LogFloat(lo, hi) => (lo * hi).sqrt().into(),
            Kind::Float(lo, hi) => ((lo + hi) / 2.).into(),
            Kind::Int(lo, hi) => ((lo + hi) / 2).into(),
            Kind::Choice(o) => serde_json::from_str(o[o.len() / 2]).unwrap(),
        }
    }
}

pub struct Suggestion {
    pub params: Params,
    /// Сколько trials легло в основу, из скольких завершённых, и из каких прогонов.
    pub used: usize,
    pub of: usize,
    pub runs: Vec<String>,
    /// Пришлось взять trials всех классов: своих нет.
    pub all_classes: bool,
}

/// Автоподбор: лучшая четверть завершённых trials класса (минимум 3) по F1;
/// числа — медиана (lr и weight_decay — в лог. шкале), варианты — самый частый (при равенстве — с лучшим F1).
pub fn suggest(runs: &[Run], target: &str) -> Suggestion {
    let collect = |any: bool| -> Vec<(f64, Params, String)> {
        runs.iter()
            .filter(|r| r.space_version() == Some(SPACE_VERSION)) // trials старых версий несравнимы
            .filter(|r| any || r.target().as_deref() == Some(target))
            .flat_map(|r| r.trials.iter().map(move |t| (r, t)))
            .filter(|(_, t)| t.state == "COMPLETE")
            .filter_map(|(r, t)| Some((t.f1?, serde_json::from_str::<Params>(&t.params).ok()?, r.name.clone())))
            .collect()
    };
    let mut pool = collect(false);
    let all_classes = pool.is_empty();
    if all_classes {
        pool = collect(true);
    }
    pool.sort_by(|a, b| b.0.total_cmp(&a.0));
    let of = pool.len();
    let top = &pool[..of.min((of / 4).max(3))];
    let mut runs_used: Vec<String> = top.iter().map(|t| t.2.clone()).collect();
    runs_used.sort();
    runs_used.dedup();

    let mut params = Params::new();
    for spec in &SPACE {
        let vals: Vec<(f64, &serde_json::Value)> = top.iter().filter_map(|(f, p, _)| Some((*f, p.get(spec.key)?))).collect();
        let nums = || vals.iter().filter_map(|v| v.1.as_f64());
        let v = if vals.is_empty() {
            spec.middle()
        } else {
            match spec.kind {
                Kind::LogFloat(..) => median(nums().filter(|x| *x > 0.).map(f64::ln).collect()).map_or_else(|| spec.middle(), |m| m.exp().into()),
                Kind::Float(..) | Kind::Int(..) => median(nums().collect()).map_or_else(|| spec.middle(), Into::into),
                Kind::Choice(_) => {
                    // (значение, частота, лучший F1)
                    let mut counts: Vec<(&serde_json::Value, usize, f64)> = vec![];
                    for (f, v) in &vals {
                        match counts.iter_mut().find(|c| c.0 == *v) {
                            Some(c) => (c.1, c.2) = (c.1 + 1, c.2.max(*f)),
                            None => counts.push((v, 1, *f)),
                        }
                    }
                    counts.iter().max_by(|a, b| a.1.cmp(&b.1).then(a.2.total_cmp(&b.2))).unwrap().0.clone()
                }
            }
        };
        // ln/exp и старые пространства поиска могут вывести за пределы — прижимаем к границе.
        params.insert(spec.key.into(), spec.clamp(v));
    }
    Suggestion { params, used: top.len(), of, runs: runs_used, all_classes }
}

fn median(mut xs: Vec<f64>) -> Option<f64> {
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(xs[n / 2]),
        _ => Some((xs[n / 2 - 1] + xs[n / 2]) / 2.),
    }
}

/// Читает форму массива из заголовка .npy (v1–v3), не загружая данные.
pub fn npy_shape(path: &Path) -> Option<Vec<u64>> {
    let mut f = fs::File::open(path).ok()?;
    let mut head = [0u8; 12];
    f.read_exact(&mut head).ok()?;
    if &head[..6] != b"\x93NUMPY" {
        return None;
    }
    let (len, start) = if head[6] == 1 {
        (u16::from_le_bytes([head[8], head[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize, 12)
    };
    let mut buf = head[start..].to_vec();
    buf.resize(len.clamp(2, 1 << 16), 0);
    f.read_exact(&mut buf[12 - start..]).ok()?;
    parse_shape(&String::from_utf8_lossy(&buf))
}

pub fn parse_shape(header: &str) -> Option<Vec<u64>> {
    let rest = &header[header.find("'shape':")? + 8..];
    let inner = &rest[rest.find('(')? + 1..rest.find(')')?];
    Some(inner.split(',').filter_map(|s| s.trim().parse().ok()).collect())
}

/// Каталоги корня проекта, где лежат .npy (exports, exports_small, …).
pub fn scan_datasets(root: &Path) -> Vec<Dataset> {
    let mut out: Vec<Dataset> = dirs(root)
        .into_iter()
        .filter_map(|dir| {
            let mut scenes: Vec<Scene> = fs::read_dir(&dir)
                .ok()?
                .flatten()
                .filter(|e| e.path().extension().is_some_and(|x| x == "npy"))
                .map(|e| Scene {
                    name: e.file_name().to_string_lossy().into(),
                    bytes: e.metadata().map(|m| m.len()).unwrap_or(0),
                    shape: npy_shape(&e.path()).unwrap_or_default(),
                })
                .collect();
            if scenes.is_empty() {
                return None;
            }
            scenes.sort_by(|a, b| a.name.cmp(&b.name));
            Some(Dataset { name: file_name(&dir), path: dir, scenes })
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Мини-копия датасета для быстрой проверки конвейера: `n` сцен, равномерно по списку, рядом с исходным
/// каталогом как `<имя>_mini`. Жёсткие ссылки — без копирования гигабайт (иначе копия, если ФС не умеет).
pub fn make_mini(d: &Dataset, n: usize) -> Result<PathBuf, String> {
    let dir = d.path.with_file_name(format!("{}_mini", d.name));
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    for i in spread(d.scenes.len(), n) {
        let name = &d.scenes[i].name;
        let (src, dst) = (d.path.join(name), dir.join(name));
        if dst.exists() {
            continue;
        }
        fs::hard_link(&src, &dst).or_else(|_| fs::copy(&src, &dst).map(|_| ())).map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(dir)
}

/// `n` номеров из `0..total`, равномерно по списку (все, если n >= total).
pub fn spread(total: usize, n: usize) -> Vec<usize> {
    let n = n.min(total);
    (0..n).map(|i| i * total / n).collect()
}

// ---------- тестирование моделей ----------

/// Загруженные во вкладку «Тестирование» модели (.pt) лежат в models/, результаты тестов — в tests/.
pub const MODELS_DIR: &str = "models";
pub const TESTS_DIR: &str = "tests";

pub fn imported_models(root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(root.join(MODELS_DIR))
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "pt")).collect())
        .unwrap_or_default();
    out.sort();
    out
}


/// Путь последнего сохранённого лога прогона из `mlstudio-logs` (без чтения файла).
///
/// Имя прогона сравнивается с частью до `__` без интерпретации пути;
/// выбирается лексикографически последний обычный файл.
pub fn artifact_log_path(root: &Path, run_name: &str) -> Option<PathBuf> {
    if run_name.is_empty() || run_name == "." || run_name == ".." || run_name.contains(['/', '\\']) {
        return None;
    }
    let mut logs: Vec<PathBuf> = fs::read_dir(root.join("mlstudio-logs"))
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.strip_suffix(".log")
                        .and_then(|name| name.split_once("__"))
                        .is_some_and(|(title, _)| title == run_name)
                })
        })
        .collect();
    logs.sort();
    logs.pop()
}

/// Markdown-артефакт, найденный в каталоге проекта.
#[derive(Clone)]
pub struct ArtifactDocument {
    pub title: String,
    pub path: PathBuf,
}

/// Все отчёты и сводки, доступные без запуска внешнего редактора.
pub fn artifact_documents(_root: &Path, runs_dir: &Path) -> Vec<ArtifactDocument> {
    let mut out = vec![ArtifactDocument {
        title: "Сводный отчёт".into(),
        path: runs_dir.join("report.md"),
    }];
    for dir in dirs(runs_dir) {
        for (name, title) in [("report.md", "Отчёт"), ("data_summary.md", "Данные")] {
            let path = dir.join(name);
            if path.exists() {
                out.push(ArtifactDocument {
                    title: format!("{title} · {}", file_name(&dir)),
                    path,
                });
            }
        }
    }
    out
}

pub fn artifact_selected(docs: &[ArtifactDocument], path: Option<&Path>) -> usize {
    path.and_then(|path| docs.iter().position(|doc| doc.path == path)).unwrap_or(0)
}

pub fn artifact_groups(docs: &[ArtifactDocument]) -> Vec<(String, Vec<usize>)> {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, doc) in docs.iter().enumerate().skip(1) {
        let name = doc.path.parent().map(file_name).unwrap_or_default();
        if let Some((_, indices)) = groups.iter_mut().find(|(group, _)| *group == name) {
            indices.push(i);
        } else {
            groups.push((name, vec![i]));
        }
    }
    groups
}

/// Объединённый список моделей: checkpoint прогона, trial fallback и импортированные файлы.
pub fn artifact_models(root: &Path, runs: &[Run]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for run in runs {
        let primary = run.path.join("best_model.pt");
        let fallback = run.best.as_ref().and_then(|v| v.get("best_trial")).and_then(|v| v.as_u64())
            .map(|best| run.path.join(format!("trial_{best:04}")).join("best.pt"));
        if primary.is_file() {
            out.push(primary);
        } else if let Some(path) = fallback.filter(|p| p.is_file()) {
            out.push(path);
        }
    }
    out.extend(imported_models(root));
    out.sort();
    out.dedup();
    out
}

pub fn read_artifact(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok().filter(|text| !text.trim().is_empty())
}

/// Копия .pt в models/; если имя занято — с суффиксом _2, _3…
pub fn import_model(root: &Path, src: &Path) -> Result<PathBuf, String> {
    if src.extension().is_none_or(|e| e != "pt") {
        return Err(format!("{}: нужен файл модели .pt", src.display()));
    }
    let dir = root.join(MODELS_DIR);
    if src.parent() == Some(dir.as_path()) {
        return Ok(src.to_path_buf());
    }
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let stem = src.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let dst = (1..)
        .map(|k| dir.join(if k == 1 { format!("{stem}.pt") } else { format!("{stem}_{k}.pt") }))
        .find(|p| !p.exists())
        .expect("бесконечный ряд");
    fs::copy(src, &dst).map_err(|e| format!("{}: {e}", src.display()))?;
    Ok(dst)
}

/// Каталог теста модели `key` на датасете: предсказания по сценам и scenes__test.json (predict_scene.py --out-dir).
pub fn test_dir(root: &Path, key: &str, dataset: &str) -> PathBuf {
    root.join(TESTS_DIR).join(format!("{key}__{dataset}"))
}

pub fn read_test(dir: &Path) -> Option<serde_json::Value> {
    read_json(&dir.join("scenes__test.json"))
}

/// Метрики по всем сценам теста вместе (сумма матриц ошибок): [F1, precision, recall, baseline F1].
pub fn pooled(eval: &serde_json::Value) -> Option<[f64; 4]> {
    let scenes = eval["scenes"].as_array().filter(|s| !s.is_empty())?;
    let mut c = [0f64; 4]; // tn, fp, fn, tp
    for s in scenes {
        for (k, (i, j)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
            c[k] += s["confusion"][i][j].as_f64()?;
        }
    }
    let [tn, fp, fneg, tp] = c;
    let prev = (tp + fneg) / (tn + fp + fneg + tp).max(1.);
    Some([2. * tp / (2. * tp + fp + fneg).max(1.), tp / (tp + fp).max(1.), tp / (tp + fneg).max(1.), 2. * prev / (1. + prev)])
}

pub fn scan_runs(runs_dir: &Path) -> Vec<Run> {
    let mut out: Vec<Run> = dirs(runs_dir)
        .into_iter()
        .map(|dir| Run {
            name: file_name(&dir),
            best: read_json(&dir.join("best.json")),
            split: read_json(&dir.join("split.json")),
            studio: read_json(&dir.join(STUDIO_FILE)),
            scene_eval: read_json(&dir.join("best.json"))
                .and_then(|b| b.get("best_trial")?.as_u64())
                .and_then(|t| read_json(&dir.join("pred").join(format!("scenes__trial{t}.json")))),
            trials: read_trials(&dir.join("trials.csv")),
            path: dir,
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Связь гиперпараметра с F1 по завершённым trials: точки (x, F1) и сила связи 0…1.
/// x — значение (для LogFloat — log10), у вариантов — номер варианта.
pub struct Influence {
    pub spec: usize,
    pub points: Vec<(f64, f64, String)>,
    /// |ρ| Спирмена для чисел, η (корреляционное отношение) для вариантов; None — мало данных.
    pub score: Option<f64>,
}

pub fn influence(run: &Run) -> Vec<Influence> {
    let done: Vec<(Params, f64, String)> = run
        .trials
        .iter()
        .filter(|t| t.state == "COMPLETE")
        .filter_map(|t| Some((serde_json::from_str::<Params>(&t.params).ok()?, t.f1?, t.number.clone())))
        .collect();
    let mut out: Vec<Influence> = SPACE
        .iter()
        .enumerate()
        .map(|(i, spec)| {
            let points: Vec<(f64, f64, String)> = done
                .iter()
                .filter_map(|(p, f1, n)| {
                    let v = p.get(spec.key)?;
                    let x = match spec.kind {
                        Kind::Choice(opts) => opts.iter().position(|o| serde_json::from_str::<serde_json::Value>(o).ok().as_ref() == Some(v))? as f64,
                        Kind::LogFloat(..) => v.as_f64()?.log10(),
                        _ => v.as_f64()?,
                    };
                    Some((x, *f1, n.clone()))
                })
                .collect();
            let (xs, ys): (Vec<f64>, Vec<f64>) = points.iter().map(|p| (p.0, p.1)).unzip();
            let score = match spec.kind {
                Kind::Choice(_) => eta(&xs, &ys),
                _ => spearman(&xs, &ys).map(f64::abs),
            };
            Influence { spec: i, points, score }
        })
        .collect();
    out.sort_by(|a, b| b.score.unwrap_or(-1.).total_cmp(&a.score.unwrap_or(-1.)));
    out
}

/// Ранги с усреднением одинаковых значений.
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        for k in i..=j {
            r[idx[k]] = (i + j) as f64 / 2.;
        }
        i = j + 1;
    }
    r
}

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let cov: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let (vx, vy): (f64, f64) = (x.iter().map(|a| (a - mx).powi(2)).sum(), y.iter().map(|b| (b - my).powi(2)).sum());
    (vx > 0. && vy > 0.).then(|| cov / (vx * vy).sqrt())
}

/// Ранговая корреляция; меньше 4 точек — None.
pub fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    (x.len() >= 4).then(|| pearson(&ranks(x), &ranks(y)))?
}

/// Корреляционное отношение η: доля разброса F1, объяснённая выбором варианта (√(межгрупповая / общая)).
pub fn eta(group: &[f64], y: &[f64]) -> Option<f64> {
    if y.len() < 4 {
        return None;
    }
    let my = y.iter().sum::<f64>() / y.len() as f64;
    let total: f64 = y.iter().map(|v| (v - my).powi(2)).sum();
    let mut groups: Vec<(f64, f64, f64)> = vec![]; // (метка, сумма, число)
    for (g, v) in group.iter().zip(y) {
        match groups.iter_mut().find(|e| e.0 == *g) {
            Some(e) => (e.1, e.2) = (e.1 + v, e.2 + 1.),
            None => groups.push((*g, *v, 1.)),
        }
    }
    let between: f64 = groups.iter().map(|(_, s, n)| n * (s / n - my).powi(2)).sum();
    (total > 0. && groups.len() > 1).then(|| (between / total).sqrt())
}

pub const STUDIO_FILE: &str = "studio.json";
/// Копия hpo_pointnet.py на момент первого запуска прогона.
pub const SNAPSHOT: &str = "hpo_pointnet.snapshot.py";

/// Дописывает запуск в studio.json прогона; при первом запуске сохраняет копию скрипта.
pub fn record_launch(root: &Path, out_dir: &Path, launch: serde_json::Value) -> std::io::Result<()> {
    fs::create_dir_all(out_dir)?;
    let snap = out_dir.join(SNAPSHOT);
    if !snap.exists() {
        fs::copy(root.join("hpo_pointnet.py"), snap)?;
    }
    update_studio(out_dir, |v| {
        if !v["launches"].is_array() {
            v["launches"] = serde_json::json!([]);
        }
        v["launches"].as_array_mut().unwrap().push(launch);
    })
}

/// Читает studio.json, меняет и записывает обратно; возвращает новое содержимое.
pub fn update_studio(out_dir: &Path, f: impl FnOnce(&mut serde_json::Value)) -> std::io::Result<()> {
    let file = out_dir.join(STUDIO_FILE);
    let mut v = read_json(&file).filter(|v| v.is_object()).unwrap_or_else(|| serde_json::json!({}));
    f(&mut v);
    fs::write(file, serde_json::to_vec_pretty(&v).unwrap())
}

/// Отличается ли текущий hpo_pointnet.py от копии в прогоне; None — копии нет.
pub fn script_changed(root: &Path, run_dir: &Path) -> Option<bool> {
    let snap = fs::read(run_dir.join(SNAPSHOT)).ok()?;
    Some(fs::read(root.join("hpo_pointnet.py")).ok().is_none_or(|cur| cur != snap))
}

/// Имя пользователя и компьютера — кто запустил прогон.
pub fn who() -> (String, String) {
    let env = |ks: &[&str]| ks.iter().find_map(|k| std::env::var(k).ok().filter(|v| !v.is_empty()));
    let user = env(&["USERNAME", "USER"]).unwrap_or_else(|| "?".into());
    let host = env(&["COMPUTERNAME", "HOSTNAME"])
        .or_else(|| fs::read_to_string("/etc/hostname").ok().map(|s| s.trim().to_string()))
        .unwrap_or_else(|| "?".into());
    (user, host)
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

fn read_trials(path: &Path) -> Vec<Trial> {
    read_csv(path)
        .into_iter()
        .map(|row| {
            let get = |k: &str| row.iter().find(|(h, _)| h == k).map(|(_, v)| v.clone()).unwrap_or_default();
            let num = |k: &str| get(k).parse().ok();
            Trial {
                number: get("number"),
                state: get("state"),
                f1: num("f1"),
                precision: num("precision"),
                recall: num("recall"),
                seconds: num("seconds"),
                params: get("params"),
                error: get("error"),
                baseline_f1: num("baseline_f1"),
            }
        })
        .collect()
}

/// F1 по эпохам из trial_XXXX/metrics.csv (пишется после конца trial).
pub fn read_epochs(run: &Path, trial: &str) -> Vec<(u32, f64)> {
    let n: u32 = trial.parse().unwrap_or(0);
    read_csv(&run.join(format!("trial_{n:04}")).join("metrics.csv"))
        .into_iter()
        .filter_map(|row| {
            let get = |k: &str| row.iter().find(|(h, _)| h == k)?.1.parse().ok();
            Some((get("epoch")? as u32, get("f1")?))
        })
        .collect()
}

/// Строки CSV как пары (заголовок, значение).
fn read_csv(path: &Path) -> Vec<Vec<(String, String)>> {
    let Ok(text) = fs::read_to_string(path) else { return vec![] };
    let mut rows = parse_csv(&text).into_iter();
    let Some(header) = rows.next() else { return vec![] };
    rows.map(|r| header.iter().cloned().zip(r).collect()).collect()
}

/// RFC 4180: кавычки, "" внутри кавычек, переводы строк внутри полей.
pub fn parse_csv(text: &str) -> Vec<Vec<String>> {
    let (mut rows, mut row, mut field, mut quoted) = (vec![], vec![], String::new(), false);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            ('"', _) => quoted = !quoted,
            (',', false) => row.push(std::mem::take(&mut field)),
            ('\r', false) => {}
            ('\n', false) => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

/// Из строки лога hpo_pointnet.py: `trial=3 epoch=2 F1=0.8123 best=… baseline=0.41 …` → (3, 2, 0.8123, Some(0.41)).
pub fn parse_epoch_line(line: &str) -> Option<(u32, u32, f64, Option<f64>)> {
    let val = |key: &str| {
        let i = line.find(key)? + key.len();
        line[i..].split_whitespace().next()
    };
    Some((
        val("trial=")?.parse().ok()?,
        val("epoch=")?.parse().ok()?,
        val(" F1=")?.parse().ok()?,
        val("baseline=").and_then(|s| s.parse().ok()),
    ))
}

/// Метрики сцены из строки predict_scene.py:
/// `=== F1=0.7130 precision=0.5540 recall=1.0000 baseline=0.7121 не покрыто=0 → …`.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneMetrics {
    pub f1: f64,
    pub precision: f64,
    pub recall: f64,
    pub baseline: Option<f64>,
    pub uncovered: u64,
}

pub fn parse_scene_line(line: &str) -> Option<SceneMetrics> {
    let val = |key: &str| {
        let i = line.find(key)? + key.len();
        line[i..].split_whitespace().next()
    };
    // Отличие от строк обучения (`trial 0 epoch 1 | Train: IoU=… F1=… P=…`): там нет precision/recall в одной строке с F1.
    line.find("precision=")?;
    line.find("recall=")?;
    Some(SceneMetrics {
        f1: val(" F1=")?.parse().ok()?,
        precision: val("precision=")?.parse().ok()?,
        recall: val("recall=")?.parse().ok()?,
        baseline: val("baseline=").and_then(|s| s.parse().ok()),
        uncovered: val("не покрыто=").and_then(|s| s.parse().ok()).unwrap_or(0),
    })
}

/// Заголовок сцены predict_scene.py: `=== сцена 3/10: scene_041_cleaned.npy` → (3, 10, "scene_041_cleaned.npy").
pub fn parse_scene_header(line: &str) -> Option<(u32, u32, String)> {
    let rest = line.strip_prefix("=== сцена ")?;
    let (counts, name) = rest.split_once(": ")?;
    let (k, n) = counts.split_once('/')?;
    Some((k.parse().ok()?, n.parse().ok()?, name.trim().to_string()))
}

/// Прогресс патчей predict_scene.py: `патч 140/144 213s` → (140, 144).
pub fn parse_patch_line(line: &str) -> Option<(u32, u32)> {
    let rest = line.trim_start().strip_prefix("патч ")?;
    let (k, n) = rest.split_once('/')?;
    Some((k.parse().ok()?, n.split_whitespace().next()?.parse().ok()?))
}

/// Заголовок прогона из run_all.py: `=== 12:00:00 ground_xyz (ground, xyz)` → ("ground_xyz", "ground", "xyz").
pub fn parse_run_header(line: &str) -> Option<(String, String, String)> {
    let rest = line.strip_prefix("=== ")?;
    let (head, tail) = rest.split_once(" (")?;
    let name = head.split_whitespace().nth(1)?.to_string();
    let (target, features) = tail.strip_suffix(')')?.split_once(", ")?;
    Some((name, target.into(), features.into()))
}

fn dirs(path: &Path) -> Vec<PathBuf> {
    fs::read_dir(path)
        .map(|r| r.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect())
        .unwrap_or_default()
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

pub fn human_bytes(b: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < units.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.1} {}", units[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape() {
        let h = "{'descr': '<f4', 'fortran_order': False, 'shape': (500000, 9), }";
        assert_eq!(parse_shape(h), Some(vec![500000, 9]));
        assert_eq!(parse_shape("{'shape': (7,), }"), Some(vec![7]));
    }

    #[test]
    fn csv() {
        let rows = parse_csv("a,b,c\r\n1,\"{\"\"x\"\": 1, \"\"y\"\": 2}\",\n");
        assert_eq!(rows, vec![vec!["a", "b", "c"], vec!["1", "{\"x\": 1, \"y\": 2}", ""]]);
    }

    #[test]
    fn epoch_line() {
        let l = "12:00:01 INFO trial=3 epoch=2 F1=0.8123 best=0.8123 baseline=0.4 lr=1e-3 время=0:01:02";
        assert_eq!(parse_epoch_line(l), Some((3, 2, 0.8123, Some(0.4))));
        assert_eq!(
            parse_run_header("=== 12:00:01 ground_xyz_hag (ground, xyz_hag)"),
            Some(("ground_xyz_hag".into(), "ground".into(), "xyz_hag".into()))
        );
        assert_eq!(parse_run_header("=== 12:00:01 ground_xyz за 0:10:00"), None);
        assert_eq!(parse_epoch_line("trial=3 epoch=2 batch 20/100 loss=0.1"), None);
    }

    /// Строки predict_scene.py: сцена, патчи и метрики сцены (таблица трансляции инференса).
    #[test]
    fn scene_lines() {
        let m = parse_scene_line("=== F1=0.7130 precision=0.5540 recall=1.0000 baseline=0.7121 не покрыто=0 → D:\\x.npy");
        assert_eq!(m, Some(SceneMetrics { f1: 0.7130, precision: 0.5540, recall: 1.0, baseline: Some(0.7121), uncovered: 0 }));
        assert_eq!(parse_scene_header("=== сцена 3/10: scene_041_cleaned.npy"), Some((3, 10, "scene_041_cleaned.npy".into())));
        assert_eq!(parse_patch_line("патч 140/144 213s"), Some((140, 144)));
        assert_eq!(parse_patch_line("=== 500000 точек, 144 патчей, xyz, cpu"), None);
        // Строка обучения не должна притворяться сценой: у неё нет precision/recall в одной строке.
        assert_eq!(parse_scene_line("trial=0 epoch=2 F1=0.6382 best=0.6382 baseline=0.5975 lr=0.00056"), None);
        assert_eq!(parse_scene_header("=== 12:00:01 ground_xyz (ground, xyz)"), None);
    }

    fn run(name: &str, target: &str, trials: &[(&str, f64, &str)]) -> Run {
        Run {
            name: name.into(),
            path: PathBuf::new(),
            best: Some(serde_json::json!({ "target": target, "search_space": SPACE_VERSION })),
            split: None,
            studio: None,
            scene_eval: None,
            trials: trials
                .iter()
                .map(|(state, f1, params)| Trial { state: (*state).into(), f1: Some(*f1), params: (*params).into(), ..Default::default() })
                .collect(),
        }
    }

    #[test]
    fn suggest_uses_best_of_class() {
        let p = |lr: f64, bs: u32, msg: bool| format!(r#"{{"lr": {lr}, "batch_size": {bs}, "use_MSG": {msg}, "nsample": 32}}"#);
        let (a, b, c, d, e) = (p(1e-3, 2, true), p(1e-3, 4, true), p(4e-3, 2, false), p(1e-2, 4, false), p(1e-4, 3, true));
        let runs = [
            run("ground_xyz", "ground", &[("COMPLETE", 0.9, &a), ("COMPLETE", 0.8, &b), ("COMPLETE", 0.85, &c), ("COMPLETE", 0.2, &d), ("PRUNED", 0.99, &e)]),
            run("water_xyz", "water", &[("COMPLETE", 0.95, &e)]),
        ];
        let s = suggest(&runs, "ground");
        assert!(!s.all_classes);
        assert_eq!((s.used, s.of, s.runs.clone()), (3, 4, vec!["ground_xyz".to_string()]));
        // лог-медиана из 1e-3, 1e-3, 4e-3 → 1e-3; batch 2 встречается дважды; MSG true дважды
        assert!((s.params["lr"].as_f64().unwrap() - 1e-3).abs() < 1e-9);
        assert_eq!(s.params["batch_size"], 2);
        assert_eq!(s.params["use_MSG"], true);
        assert_eq!(s.params["nsample"], 32);
        // параметра нет в trials → середина пространства
        assert_eq!(s.params["n_sa_layers"], 3);
        // своих trials нет → берутся все классы
        assert!(suggest(&runs, "noise").all_classes);
    }

    #[test]
    fn suggest_ignores_runs_of_other_space_versions() {
        let a = r#"{"lr": 0.001}"#;
        let mut old = run("old", "ground", &[("COMPLETE", 0.9, a)]);
        old.best = Some(serde_json::json!({ "target": "ground", "search_space": SPACE_VERSION - 1 }));
        let mut unversioned = run("none", "ground", &[("COMPLETE", 0.9, a)]);
        unversioned.best = Some(serde_json::json!({ "target": "ground" }));
        let s = suggest(&[old, unversioned], "ground");
        assert_eq!(s.of, 0);
    }

    #[test]
    fn suggest_clamps_bounds_and_survives_non_numbers() {
        let a = r#"{"lr": 0.01, "radius_l0": null}"#;
        let runs = [run("ground_xyz", "ground", &[("COMPLETE", 0.9, a)])];
        let s = suggest(&runs, "ground");
        assert_eq!(s.params["lr"], 0.01); // exp(ln(0.01)) > 0.01 — прижато к границе, а не середина
        assert_eq!(s.params["radius_l0"], SPACE[6].middle()); // null → середина, без паники
        assert_eq!(median(vec![]), None);
    }

    #[test]
    fn run_meta_from_split_before_best() {
        let r = Run {
            name: "x".into(),
            path: PathBuf::new(),
            best: None,
            split: Some(serde_json::json!({ "target": "water", "features": "xyz_hag" })),
            studio: None,
            scene_eval: None,
            trials: vec![Trial { baseline_f1: Some(0.4), ..Default::default() }],
        };
        assert_eq!((r.target().as_deref(), r.features().as_deref(), r.baseline()), (Some("water"), Some("xyz_hag"), Some(0.4)));
    }

    #[test]
    fn spec_parse() {
        assert_eq!(SPACE[0].parse("1e-3"), Ok(serde_json::json!(0.001)));
        assert!(SPACE[0].parse("0.5").is_err());
        assert_eq!(SPACE[2].parse("0,25"), Ok(serde_json::json!(0.25)));
        assert_eq!(SPACE[8].parse("true"), Ok(serde_json::json!(true)));
        assert!(SPACE[5].parse("100").is_err());
    }

    #[test]
    fn mini_dataset_picks_evenly_and_links() {
        let base = std::env::temp_dir().join("mlstudio_mini_test");
        _ = fs::remove_dir_all(&base);
        let src = base.join("exports");
        fs::create_dir_all(&src).unwrap();
        let scenes = (0..20)
            .map(|i| {
                let name = format!("scene_{i:03}.npy");
                fs::write(src.join(&name), [i as u8]).unwrap();
                Scene { name, bytes: 1, shape: vec![] }
            })
            .collect();
        let d = Dataset { name: "exports".into(), path: src, scenes };
        let dir = make_mini(&d, 4).unwrap();
        assert_eq!(dir, base.join("exports_mini"));
        let mut got: Vec<String> = fs::read_dir(&dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into()).collect();
        got.sort();
        assert_eq!(got, ["scene_000.npy", "scene_005.npy", "scene_010.npy", "scene_015.npy"]);
        assert_eq!(fs::read(dir.join("scene_010.npy")).unwrap(), [10]);
        make_mini(&d, 4).unwrap(); // повторно — без ошибки
    }

    /// Влияние параметров на настоящих прогонах: cargo test real_influence -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_influence() {
        for r in scan_runs(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../runs")) {
            let s: Vec<String> = influence(&r).iter().map(|f| format!("{}={}", SPACE[f.spec].key, f.score.map_or("—".into(), |v| format!("{v:.2}")))).collect();
            println!("{}: {}", r.name, s.join(" "));
        }
    }

    #[test]
    fn rank_stats() {
        // монотонная связь — |ρ| = 1 даже нелинейная
        assert!((spearman(&[1., 2., 3., 4., 5.], &[1., 4., 9., 16., 25.]).unwrap() - 1.).abs() < 1e-12);
        assert!((spearman(&[1., 2., 3., 4.], &[4., 3., 2., 1.]).unwrap() + 1.).abs() < 1e-12);
        assert_eq!(spearman(&[1., 2., 3.], &[1., 2., 3.]), None);
        assert_eq!(ranks(&[10., 20., 10., 30.]), vec![0.5, 2., 0.5, 3.]);
        // варианты целиком определяют F1 → η = 1; не влияют → 0
        assert!((eta(&[0., 0., 1., 1.], &[0.5, 0.5, 0.9, 0.9]).unwrap() - 1.).abs() < 1e-12);
        assert!(eta(&[0., 1., 0., 1.], &[0.5, 0.5, 0.9, 0.9]).unwrap().abs() < 1e-12);
    }

    #[test]
    fn launch_record_appends_and_snapshots() {
        let root = std::env::temp_dir().join(format!("mlstudio-launch-{}", std::process::id()));
        let out = root.join("runs").join("x");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("hpo_pointnet.py"), "v1").unwrap();
        record_launch(&root, &out, serde_json::json!({"n": 1})).unwrap();
        assert_eq!(script_changed(&root, &out), Some(false));
        fs::write(root.join("hpo_pointnet.py"), "v2").unwrap();
        record_launch(&root, &out, serde_json::json!({"n": 2})).unwrap();
        // копия — от первого запуска
        assert_eq!(script_changed(&root, &out), Some(true));
        update_studio(&out, |v| v["note"] = "заметка".into()).unwrap();
        let v = read_json(&out.join(STUDIO_FILE)).unwrap();
        assert_eq!(v["launches"].as_array().unwrap().len(), 2);
        assert_eq!(v["note"], "заметка");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn real_npy() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../exports_small/scene_001_cleaned.npy");
        if p.exists() {
            assert_eq!(npy_shape(&p), Some(vec![500000, 9]));
        }
    }

    #[test]
    fn pooled_and_spread() {
        let e = serde_json::json!({"scenes": [{"confusion": [[8, 2], [1, 1]]}, {"confusion": [[0, 0], [1, 3]]}]});
        // tn 8, fp 2, fn 2, tp 4: F1 = 8/12, P = 4/6, R = 4/6.
        let [f1, pr, rc, _] = super::pooled(&e).unwrap();
        assert!((f1 - 8. / 12.).abs() < 1e-9 && (pr - 4. / 6.).abs() < 1e-9 && (rc - 4. / 6.).abs() < 1e-9);
        assert!(super::pooled(&serde_json::json!({"scenes": []})).is_none());
        assert_eq!(super::spread(10, 3), [0, 3, 6]);
        assert_eq!(super::spread(2, 5), [0, 1]);
    }

    #[test]
    fn artifact_log_returns_latest_exact_run_log() {
        let root = std::env::temp_dir().join(format!("mlstudio-artifact-log-{}", std::process::id()));
        let logs = root.join("mlstudio-logs");
        fs::create_dir_all(&logs).unwrap();
        fs::write(logs.join("ground_xyz__2026-01-01.log"), "old").unwrap();
        fs::write(logs.join("ground_xyz__2026-01-02.log"), "latest").unwrap();
        fs::write(logs.join("ground_xyz_extra__2026-01-03.log"), "other").unwrap();

        let path = artifact_log_path(&root, "ground_xyz").unwrap();
        assert_eq!(path, logs.join("ground_xyz__2026-01-02.log"));
        assert!(artifact_log_path(&root, "missing").is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn artifact_selection_tracks_path_when_order_changes() {
        let docs = vec![
            ArtifactDocument { title: "Сводный отчёт".into(), path: "runs/report.md".into() },
            ArtifactDocument { title: "Данные".into(), path: "runs/b/data_summary.md".into() },
            ArtifactDocument { title: "Отчёт".into(), path: "runs/a/report.md".into() },
        ];
        assert_eq!(artifact_selected(&docs, Some(Path::new("runs/a/report.md"))), 2);
        assert_eq!(artifact_selected(&docs, Some(Path::new("missing"))), 0);
        assert_eq!(artifact_selected(&docs, None), 0);
    }

    #[test]
    fn artifact_sidebar_groups_documents_by_run() {
        let docs = vec![
            ArtifactDocument { title: "Сводный отчёт".into(), path: "runs/report.md".into() },
            ArtifactDocument { title: "Отчёт · a".into(), path: "runs/a/report.md".into() },
            ArtifactDocument { title: "Данные · a".into(), path: "runs/a/data_summary.md".into() },
            ArtifactDocument { title: "Данные · b".into(), path: "runs/b/data_summary.md".into() },
        ];
        assert_eq!(artifact_groups(&docs), vec![("a".to_string(), vec![1, 2]), ("b".to_string(), vec![3])]);
        assert!(artifact_groups(&docs[..1]).is_empty());
    }

    #[test]
    fn artifact_documents_include_summary_and_run_files() {
        let root = std::env::temp_dir().join(format!("mlstudio-artifacts-{}", std::process::id()));
        let runs = root.join("runs");
        fs::create_dir_all(runs.join("a")).unwrap();
        fs::write(runs.join("report.md"), "# общий").unwrap();
        fs::write(runs.join("a").join("report.md"), "# run").unwrap();
        fs::write(runs.join("a").join("data_summary.md"), "# data").unwrap();
        let docs = artifact_documents(&root, &runs);
        assert_eq!(docs.len(), 3);
                assert_eq!(docs[2].title, "Данные · a");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn artifact_models_deduplicate_run_and_imported_checkpoints() {
        let root = std::env::temp_dir().join(format!("mlstudio-models-{}", std::process::id()));
        let run = root.join("runs").join("r");
        fs::create_dir_all(run.join("trial_0001")).unwrap();
        fs::create_dir_all(root.join("models")).unwrap();
        fs::write(run.join("best_model.pt"), "best").unwrap();
        fs::write(run.join("trial_0001").join("best.pt"), "trial").unwrap();
        fs::write(root.join("models").join("import.pt"), "import").unwrap();
        let r = Run { name: "r".into(), path: run.clone(), best: Some(serde_json::json!({"best_trial": 1})), split: None, studio: None, scene_eval: None, trials: vec![] };
        let models = artifact_models(&root, &[r]);
        assert_eq!(models.len(), 2);
        assert!(models.contains(&run.join("best_model.pt")));
        assert!(!models.contains(&run.join("trial_0001").join("best.pt")));
        assert!(models.iter().any(|p| p.ends_with("import.pt")));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_artifact_rejects_missing_and_blank_files() {
        let root = std::env::temp_dir().join(format!("mlstudio-read-artifact-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let blank = root.join("blank.md");
        fs::write(&blank, " \n").unwrap();
        assert_eq!(read_artifact(&root.join("missing.md")), None);
        assert_eq!(read_artifact(&blank), None);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn import_model() {
        let root = std::env::temp_dir().join(format!("mlstudio_import_{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let src = root.join("m.pt");
        std::fs::write(&src, "x").unwrap();
        assert_eq!(super::import_model(&root, &src).unwrap(), root.join("models").join("m.pt"));
        assert_eq!(super::import_model(&root, &src).unwrap(), root.join("models").join("m_2.pt"));
        assert!(super::import_model(&root, &root.join("a.txt")).is_err());
        assert_eq!(super::imported_models(&root).len(), 2);
        std::fs::remove_dir_all(&root).unwrap();
    }
}