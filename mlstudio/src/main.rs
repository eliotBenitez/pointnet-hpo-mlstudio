#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
//! ML Studio: трансляция прогонов PointNet++ (hpo_pointnet.py / run_all.py), таблицы trials, запуск, датасеты.
mod artifact_markdown;
mod cloud;
mod data;
mod env;
mod monitor;
mod nevo;
mod projects;
mod scenario;

use std::borrow::Cow;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use data::*;
use gpui_kit::assets::IconName as Lucide;
use gpui_kit::component::button::*;
use gpui_kit::component::chart::LineChart;
use gpui_kit::component::checkbox::Checkbox;
use gpui_kit::component::input::*;
use gpui_kit::component::scroll::ScrollableElement;
use gpui_kit::component::tooltip::Tooltip;
use gpui_kit::component::*;
use gpui_kit::prelude::FluentBuilder;
use gpui_kit::*;

const TARGETS: [&str; 5] = ["ground", "vegetation", "noise", "building", "water"];
const FEATURES: [&str; 2] = ["xyz", "xyz_hag"];
const TASKS: [&str; 7] = ["2", "3", "4", "5", "6", "7", "final"];
/// Модель должна обгонять baseline хотя бы на столько (README): ниже — «зона вылета».
const MARGIN: f64 = 0.05;
// DirectWrite группирует PT Sans Narrow в семейство «PT Sans» (ширина condensed); встроен только этот файл.
#[cfg(windows)]
const DISPLAY: &str = "PT Sans";
#[cfg(not(windows))]
const DISPLAY: &str = "PT Sans Narrow";
const MONO: &str = "PT Mono";
const BODY: &str = "Golos Text";
const LINE_H: f32 = 18.;

// ---------- палитра «спортивной полосы» ----------

#[derive(Clone, Copy)]
struct Pal {
    bg: Hsla,
    surface: Hsla,
    ink: Hsla,
    muted: Hsla,
    rule: Hsla,
    rail: Hsla,
    rail_sel: Hsla,
    rail_ink: Hsla,
    /// Поверхность на ступень выше карточки: кнопки, выделение, наведение.
    raised: Hsla,
    live: Hsla,
    live_ink: Hsla,
    win: Hsla,
    loss: Hsla,
}

fn pal(dark: bool) -> Pal {
    if dark {
        Pal {
            bg: rgb(0x0E1013).into(),
            surface: rgb(0x15181D).into(),
            ink: rgb(0xE8EAED).into(),
            muted: rgb(0x8A919C).into(),
            rule: rgb(0x252A32).into(),
            rail: rgb(0x0B0D10).into(),
            rail_sel: rgb(0x1D2128).into(),
            rail_ink: rgb(0xE8EAED).into(),
            raised: rgb(0x1D2128).into(),
            live: rgb(0xF2A900).into(),
            live_ink: rgb(0xF5C04A).into(),
            win: rgb(0x3FB27F).into(),
            loss: rgb(0xE5484D).into(),
        }
    } else {
        Pal {
            bg: rgb(0xF6F6F7).into(),
            surface: rgb(0xFFFFFF).into(),
            ink: rgb(0x16181D).into(),
            muted: rgb(0x5F6672).into(),
            rule: rgb(0xE3E4E8).into(),
            rail: rgb(0xEFEFF1).into(),
            rail_sel: rgb(0xFFFFFF).into(),
            rail_ink: rgb(0x16181D).into(),
            raised: rgb(0xEDEEF0).into(),
            live: rgb(0xF2A900).into(),
            live_ink: rgb(0x8A5D00).into(),
            win: rgb(0x1F8A5B).into(),
            loss: rgb(0xD13438).into(),
        }
    }
}

/// Цвет класса — плоская заливка шапки (по мотивам условных знаков карт).
fn class_hex(target: &str) -> u32 {
    match target {
        "ground" => 0x8A5A2B,
        "vegetation" => 0x3B7A37,
        "noise" => 0x8E3B8C,
        "building" => 0x3D4C60,
        "water" => 0x2867A8,
        _ => 0x4A4F57,
    }
}

fn class_color(target: &str) -> Hsla {
    rgb(class_hex(target)).into()
}

fn class_ru(target: &str) -> &'static str {
    match target {
        "ground" => "Грунт",
        "vegetation" => "Растительность",
        "noise" => "Шум",
        "building" => "Здания",
        "water" => "Вода",
        _ => "Сценарий",
    }
}

fn tnum() -> FontFeatures {
    FontFeatures(Arc::new(vec![("tnum".into(), 1)]))
}

// ---------- процессы ----------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ArtifactView {
    #[default]
    Documents,
    Models,
    Logs,
}

impl ArtifactView {
    const ALL: [Self; 3] = [Self::Documents, Self::Models, Self::Logs];

    fn label(self) -> &'static str {
        match self {
            Self::Documents => "Документы",
            Self::Models => "Модели",
            Self::Logs => "Логи",
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Live,
    Runs,
    Launch,
    Datasets,
    Testing,
    Artifacts,
    Env,
    Projects,
    Scenarios,
}

/// Открытый редактор шага сценария: поля имени, trials, эпох и seed.
struct StepEditor {
    idx: usize,
    inputs: Vec<Entity<InputState>>,
    /// Поля гиперпараметров по порядку SPACE (None — вариант выбирается кнопками).
    params: Vec<Option<Entity<InputState>>>,
    _subs: Vec<Subscription>,
}

/// Идущий сценарий: копия шагов на момент запуска (правка файла не влияет на уже идущий).
struct Active {
    name: String,
    steps: Vec<scenario::Step>,
    /// Индекс следующего шага.
    next: usize,
    /// С какого шага запущен.
    from: usize,
    /// Процесс текущего шага в `jobs`.
    job: Option<usize>,
}

#[derive(Default)]
struct Output {
    lines: Vec<String>,
    /// Сколько строк выброшено с начала: в памяти только хвост лога, на диске он полный.
    dropped: usize,
    exit: Option<Result<(), String>>,
    /// Копия лога на диске: лог переживает закрытие приложения.
    file: Option<std::fs::File>,
}

/// Сколько строк лога держать в памяти: прогон идёт часами, а процессов в списке накапливается много.
const LOG_CAP: usize = 20_000;

impl Output {
    fn push(&mut self, line: String) {
        if let Some(f) = &mut self.file {
            use std::io::Write;
            _ = writeln!(f, "{line}");
        }
        if self.lines.len() >= LOG_CAP {
            let cut = LOG_CAP / 4;
            self.lines.drain(..cut);
            self.dropped += cut;
        }
        self.lines.push(line);
    }
}

/// Каталог логов процессов в корне проекта (не в runs/, чтобы не считаться прогоном).
const LOG_DIR: &str = "mlstudio-logs";

/// Имя лог-файла: `<название>__<UTC-время>.log`, время сортируется как строка.
fn log_file_name(title: &str) -> String {
    let safe: String = title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || "-_.+".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    format!("{safe}__{}.log", utc_stamp(secs))
}

/// UNIX-время → `YYYY-MM-DD_HH-MM-SSZ` (civil_from_days, Howard Hinnant).
fn utc_stamp(secs: i64) -> String {
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}_{:02}-{:02}-{:02}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

struct Epoch {
    trial: u32,
    epoch: u32,
    f1: f64,
    at: Instant,
}

struct Job {
    title: String,
    program: String,
    args: Vec<String>,
    pid: Option<u32>,
    out: Arc<Mutex<Output>>,
    started: Instant,
    ended: Option<Instant>,
    stopped: bool,
    /// Установка окружения (не обучение): в трансляции только лог.
    setup: bool,
    /// Файл лога на диске (None — не удалось создать).
    log_path: Option<PathBuf>,
    /// Это predict_scene.py: по окончании загрузить результат в превью.
    predict: Option<PredJob>,
    /// Номер последней разобранной строки, считая с начала лога: буфер лога обрезается, поэтому
    /// номер не совпадает с индексом в `Output.lines` (сдвиг — `Output.dropped`).
    seen: usize,
    /// Текущий прогон: для run_all меняется по строкам `=== … name (target, features)`.
    run: String,
    target: String,
    features: String,
    segment_start: Instant,
    epochs: Vec<Epoch>,
    baseline: Option<f64>,
    n_trials: u32,
    n_epochs: u32,
    /// Это predict_scene.py (тест модели или оценка по сценам): свой набор данных — сцены, а не эпохи.
    predicting: bool,
    /// Метрики сцен по порядку лога: (имя сцены, метрики).
    scenes: Vec<(String, SceneMetrics)>,
    /// Текущая сцена (k, n) из `=== сцена 3/10: …`.
    scene_progress: Option<(u32, u32)>,
    /// Прогресс патчей текущей сцены (i, n) из `патч 140/144 213s`.
    patch_progress: Option<(u32, u32)>,
}

impl Job {
    fn running(&self) -> bool {
        self.ended.is_none()
    }

    fn arg(args: &[String], key: &str) -> Option<String> {
        args.iter().skip_while(|a| *a != key).nth(1).cloned()
    }

    fn elapsed(&self) -> Duration {
        self.ended.unwrap_or_else(Instant::now) - self.started
    }

    /// (метка, цвет фона, цвет текста) статуса.
    fn status(&self, p: &Pal) -> (&'static str, Hsla, Hsla) {
        match &self.out.lock().unwrap().exit {
            None => ("В ЭФИРЕ", p.live, rgb(0x14171C).into()),
            Some(_) if self.stopped => ("ОСТАНОВЛЕН", p.muted, p.bg),
            Some(Ok(())) => ("ЗАВЕРШЁН", p.win, rgb(0xFFFFFF).into()),
            Some(Err(_)) => ("ОШИБКА", p.loss, rgb(0xFFFFFF).into()),
        }
    }

    /// Лучший F1 и длительность каждого trial текущего прогона, по убыванию F1.
    fn standings(&self) -> Vec<(u32, f64, u32, f64)> {
        let mut rows: Vec<(u32, f64, u32, f64)> = vec![];
        let mut prev = self.segment_start;
        for e in &self.epochs {
            match rows.last_mut() {
                Some(r) if r.0 == e.trial => {
                    r.1 = r.1.max(e.f1);
                    r.2 = e.epoch;
                    r.3 += (e.at - prev).as_secs_f64();
                }
                _ => rows.push((e.trial, e.f1, e.epoch, (e.at - prev).as_secs_f64())),
            }
            prev = e.at;
        }
        rows.sort_by(|a, b| b.1.total_cmp(&a.1));
        rows
    }

    /// Номер trial в этом запуске, с 1: Optuna продолжает нумерацию в продолженном study.
    fn trial_in_run(&self, trial: u32) -> u32 {
        trial - self.epochs.first().map_or(trial, |e| e.trial.min(trial)) + 1
    }

    /// Оценка остатка: среднее время эпохи × оставшиеся эпохи (верхняя граница: pruning обрывает trials).
    fn eta(&self) -> Option<Duration> {
        let last = self.epochs.last()?;
        let done = self.epochs.len() as f64;
        let per = (last.at - self.segment_start).as_secs_f64() / done;
        let left_trials = self.n_trials.saturating_sub(self.trial_in_run(last.trial)) as f64;
        let left =
            left_trials * self.n_epochs as f64 + self.n_epochs.saturating_sub(last.epoch) as f64;
        Some(Duration::from_secs_f64(per * left))
    }
}

/// 3D-превью сцены во вкладке «Датасеты».
struct Preview {
    scene: PathBuf,
    /// None — ещё грузится.
    cloud: Option<Result<Arc<cloud::Cloud>, String>>,
    view: cloud::View,
    by: cloud::ColorBy,
    image: Option<Arc<RenderImage>>,
    /// Последняя позиция мыши при перетаскивании.
    last: Option<gpui_kit::Point<Pixels>>,
    /// Вид поменялся — перерисовать в ближайшем кадре.
    dirty: bool,
    /// Предсказание модели на этой сцене.
    pred: Option<PredView>,
    /// Готовое предсказание, запрошенное до загрузки облака: подхватится, когда облако загрузится.
    pending: Option<PredJob>,
    /// Размер картинки на экране (px) — для перевода сдвига мыши в долю кадра.
    shown: Rc<std::cell::Cell<(f32, f32)>>,
}

/// Предсказание в превью: какая модель, была ли сцена в её val, результат (None — ещё считается).
struct PredView {
    request: Arc<()>,
    label: String,
    split: Option<&'static str>,
    result: Option<Result<cloud::Outcome, String>>,
}

/// Процесс predict_scene.py: что загрузить в превью, когда он закончится.
#[derive(Clone)]
struct PredJob {
    scene: PathBuf,
    out: PathBuf,
    target: String,
    request: Arc<()>,
}

/// Модель, которой можно проверить сцену: лучший trial прогона с сохранённым best.pt.
#[derive(Clone)]
struct Model {
    run: String,
    /// None — модель из файла: номера trial у неё нет.
    trial: Option<u64>,
    f1: Option<f64>,
    checkpoint: PathBuf,
    target: String,
    /// Где сцена была при обучении (по split.json прогона).
    split: Option<&'static str>,
}

impl Model {
    fn label(&self) -> String {
        match self.trial {
            Some(t) => format!("{} · trial {t}", self.run),
            None => self.run.clone(),
        }
    }
}

/// Что сделать по клику на сцену в таблице оценки (scene_table).
type OpenScene = Rc<dyn Fn(&mut Studio, PathBuf, &mut Context<Studio>)>;

/// Модель во вкладке «Тестирование»: модель прогона или загруженный .pt из models/.
struct TestModel {
    /// Выбор и заголовок процесса: run-<прогон> или file-<имя .pt>.
    key: String,
    /// Каталог тестов: у прогона с номером лучшего trial — новая лучшая модель не берёт старые предсказания.
    tag: String,
    label: String,
    file: PathBuf,
    run: Option<usize>,
}

/// Скрипт инференса едет внутри exe: если в проекте его нет, он кладётся рядом с hpo_pointnet.py.
const PREDICT_PY: &str = include_str!("../../predict_scene.py");

/// Разрешение картинки превью (масштабируется под панель).
/// Сцен в мини-копии датасета: hpo_pointnet.py делит 80/20, значит 8 train / 2 val.
const MINI_SCENES: usize = 10;
const PREVIEW_W: u32 = 1200;
const PREVIEW_H: u32 = 800;

// ---------- приложение ----------

#[derive(Default)]
struct ResourcesVisibility {
    collapsed: bool,
}

impl ResourcesVisibility {
    fn collapsed(&self) -> bool {
        self.collapsed
    }

    fn toggle(&mut self) {
        self.collapsed = !self.collapsed;
    }
}

struct Studio {
    root: PathBuf,
    /// Последний замер ЦП, ОЗУ и видеокарт.
    usage: monitor::Usage,
    /// Недавние и избранные проекты (конфиг пользователя).
    projects: projects::Projects,
    /// Почему выбранная папка не открылась.
    projects_note: Option<String>,
    dark: bool,
    resources: ResourcesVisibility,
    tab: Tab,
    datasets: Vec<Dataset>,
    sel_dataset: usize,
    runs: Vec<Run>,
    sel_run: usize,
    sel_trial: Option<String>,
    /// Прогоны, отмеченные для сравнения (по имени).
    compare: Vec<String>,
    scenarios: Vec<scenario::Scenario>,
    sel_scenario: usize,
    scenario_name: Entity<InputState>,
    scenario_name_for: Option<String>,
    active: Option<Active>,
    step_edit: Option<StepEditor>,
    /// Итог сценария или ошибка: (успех, текст).
    scenario_note: Option<(bool, String)>,
    _scenario_sub: Option<Subscription>,
    /// Заметка к выбранному прогону (сохраняется в его studio.json) и для какого прогона она загружена.
    note: Entity<InputState>,
    note_for: Option<String>,
    _note_sub: Option<Subscription>,
    /// Открытое сравнение: прогон и F1 по эпохам его лучшего trial.
    compare_view: Option<Vec<(String, Vec<(u32, f64)>)>>,
    trial_epochs: Vec<(u32, f64)>,
    jobs: Vec<Job>,
    sel_job: usize,
    autoscroll: bool,
    /// Взведённая кнопка «Остановить»: (индекс процесса, когда).
    confirm_stop: Option<(usize, Instant)>,
    /// Взведённая кнопка «Удалить» прогона: (имя, когда).
    confirm_delete: Option<(String, Instant)>,
    /// Почему не удалось удалить прогон (показывается в карточке прогона).
    delete_error: Option<String>,
    /// Ошибка действия на текущей вкладке (заметка, повтор, оценка по сценам, сохранение модели):
    /// рисуется в шапке вкладки, чтобы сообщение не пропадало на невидимом экране.
    tab_error: Option<String>,
    /// Почему редактор гиперпараметров недоступен (пространство поиска в hpo_pointnet.py изменилось).
    space_error: Option<String>,
    /// Идёт фоновое сканирование runs/.
    scanning: bool,
    /// Результат проверки окружения Python; None — ещё не проверено.
    env: Option<env::EnvInfo>,
    env_checking: bool,
    /// Сборка torch, выбранная вручную (иначе рекомендованная по драйверу).
    build: Option<&'static str>,
    preview: Option<Preview>,
    /// Путь к Python, который надо подставить в «Запуск» при следующей отрисовке (нужно окно).
    pending_python: Option<String>,
    /// То же для каталога данных (мини-копия готова в фоне).
    pending_data: Option<String>,
    /// Сообщение под шапкой датасета (мини-копия: готово / ошибка).
    ds_note: Option<(bool, String)>,
    /// Итог экспорта отчёта в Nevo: (успех, текст).
    nevo_note: Option<(bool, String)>,
    log_scroll: UniformListScrollHandle,
    run_all: bool,
    target: &'static str,
    features: &'static str,
    tasks: Vec<&'static str>,
    python: Entity<InputState>,
    data_dir: Entity<InputState>,
    final_data_dir: Entity<InputState>,
    runs_dir: Entity<InputState>,
    out_name: Entity<InputState>,
    trials: Entity<InputState>,
    epochs: Entity<InputState>,
    final_epochs: Entity<InputState>,
    patch_points: Entity<InputState>,
    patches_per_scene: Entity<InputState>,
    seed: Entity<InputState>,
    workers: Entity<InputState>,
    /// Раздел «Артефакты»: выбранный документ и режим просмотра.
    artifact_doc: Option<PathBuf>,
    artifact_view: ArtifactView,
    artifact_log: Option<String>,
    /// Класс модели из файла (checkpoint → цель): определяется один раз.
    model_targets: std::collections::HashMap<PathBuf, String>,
    /// Кэш текста открытого артефакта: путь, размер и время изменения → содержимое.
    artifact_text: Option<((PathBuf, u64, Option<std::time::SystemTime>), SharedString)>,
    /// «Тестирование»: ключ выбранной модели, датасет, сколько сцен, загруженные .pt, результат теста.
    test_model: Option<String>,
    /// Имя датасета (не номер: список датасетов пересканируется).
    test_dataset: Option<String>,
    test_scenes: Entity<InputState>,
    imported: Vec<PathBuf>,
    test_eval: Option<serde_json::Value>,
    test_note: Option<(bool, String)>,
    /// Функция потерь: "ce", "focal" или "focal_lovasz" (--loss в hpo_pointnet.py / run_all.py).
    loss: &'static str,
    focal_gamma: Entity<InputState>,
    lovasz_weight: Entity<InputState>,
    weight_power: Entity<InputState>,
    /// Баланс патчей Train (--equibatch): центр на точке цели или фона.
    equibatch: bool,
    positive_fraction: Entity<InputState>,
    boundary_fraction: Entity<InputState>,
    boundary_weight: Entity<InputState>,
    boundary_radius: Entity<InputState>,
    extra: Entity<InputState>,
    /// Своя конфигурация гиперпараметров (иначе всё подбирает Optuna).
    custom: bool,
    /// true — обучить ровно эту конфигурацию (1 trial); false — поставить её первой точкой подбора.
    custom_only: bool,
    /// Поля чисел и значения вариантов, по порядку SPACE.
    param_inputs: Vec<Option<Entity<InputState>>>,
    param_choice: Vec<serde_json::Value>,
    /// Откуда взяты текущие значения редактора.
    params_note: Option<String>,
    _tick: Task<()>,
    _quit: Subscription,
}

impl Studio {
    fn new(root: PathBuf, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let tick = cx.spawn(async move |this, cx| {
            let mut n = 0u32;
            loop {
                cx.background_executor()
                    .timer(Duration::from_millis(200))
                    .await;
                n += 1;
                if this
                    .update(cx, |this, cx| this.tick(n % 10 == 0, cx))
                    .is_err()
                {
                    break;
                }
            }
        });
        // Не оставлять обучение сиротой при закрытии окна.
        let quit = cx.on_app_quit(|this, _| {
            this.jobs.iter().filter(|j| j.running()).for_each(kill);
            async {}
        });
        let mut known = projects::Projects::load();
        let valid = projects::is_project(&root);
        if valid {
            known.touch(&root);
            known.save();
        }
        let datasets = scan_datasets(&root);
        let data = datasets
            .iter()
            .find(|d| d.name == "exports")
            .or(datasets.first())
            .map(|d| d.name.clone())
            .unwrap_or_else(|| "exports".into());
        // Windows: Scripts/python.exe, Linux/macOS: bin/python.
        let python = [".venv311/Scripts/python.exe", ".venv311/bin/python"]
            .iter()
            .map(|p| root.join(p))
            .find(|p| p.exists())
            .map_or_else(|| "python".into(), |p| p.to_string_lossy().into_owned());
        let dark = matches!(
            window.appearance(),
            WindowAppearance::Dark | WindowAppearance::VibrantDark
        );
        let mut input = |v: &str, ph: &str| {
            let (v, ph) = (v.to_string(), ph.to_string());
            cx.new(|cx| InputState::new(window, cx).default_value(v).placeholder(ph))
        };
        let param_inputs = SPACE
            .iter()
            .map(|s| match s.kind {
                Kind::Choice(_) => None,
                _ => Some(input(&s.show(&s.middle()), "")),
            })
            .collect();
        let param_choice = SPACE.iter().map(|s| s.middle()).collect();
        // Каталог результатов задаётся формой «Запуск». Стартовое сканирование — то же значение
        // по умолчанию «runs», с которым создаётся поле; повторные сканирования (reload_runs)
        // читают текущее значение поля.
        let runs_dir = input("runs", "каталог результатов");
        let runs = scan_runs(&root.join("runs"));
        let mut this = Self {
            projects: known,
            usage: monitor::latest(),
            projects_note: None,
            custom: false,
            custom_only: true,
            param_inputs,
            param_choice,
            params_note: None,
            runs,
            datasets,
            dark,
            resources: ResourcesVisibility::default(),
            sel_dataset: 0,
            sel_run: 0,
            sel_trial: None,
            compare: vec![],
            scenarios: scenario::scan(&root),
            sel_scenario: 0,
            scenario_name: input("", "название сценария"),
            scenario_name_for: None,
            active: None,
            step_edit: None,
            scenario_note: None,
            _scenario_sub: None,
            note: input("", "Заметка к прогону: что пробовали, что вышло"),
            note_for: None,
            _note_sub: None,
            compare_view: None,
            trial_epochs: vec![],
            jobs: vec![],
            sel_job: 0,
            autoscroll: true,
            confirm_stop: None,
            confirm_delete: None,
            delete_error: None,
            tab_error: None,
            space_error: match script_space_version(&root) {
                Some(SPACE_VERSION) => None,
                v => Some(format!(
                    "Редактор рассчитан на пространство поиска v{SPACE_VERSION}, а в hpo_pointnet.py {} — диапазоны могли измениться. Обновите SPACE в src/data.rs.",
                    v.map_or("версия не найдена".into(), |v| format!(
                        "v{v}"
                    ))
                )),
            },
            scanning: false,
            env: None,
            env_checking: false,
            build: None,
            pending_python: None,
            pending_data: None,
            ds_note: None,
            nevo_note: None,
            preview: None,
            log_scroll: UniformListScrollHandle::new(),
            tab: if valid { Tab::Live } else { Tab::Projects },
            run_all: false,
            target: "ground",
            features: "xyz",
            tasks: TASKS.to_vec(),
            python: input(&python, "python"),
            data_dir: input(&data, "каталог с scene_*.npy"),
            final_data_dir: input("exports_final", "каталог для финальных прогонов"),
            runs_dir,
            out_name: input("", "по умолчанию <класс>_<признаки>"),
            trials: input("25", ""),
            epochs: input("10", ""),
            final_epochs: input("30", ""),
            patch_points: input("16384", ""),
            patches_per_scene: input("2", ""),
            seed: input("42", ""),
            workers: input("0", ""),
            loss: "ce",
            artifact_doc: None,
            artifact_view: ArtifactView::Documents,
            artifact_log: None,
            model_targets: Default::default(),
            artifact_text: None,
            test_model: None,
            test_dataset: None,
            test_scenes: input("10", ""),
            imported: vec![],
            test_eval: None,
            test_note: None,
            focal_gamma: input("2", ""),
            lovasz_weight: input("0.5", ""),
            weight_power: input("1", ""),
            equibatch: false,
            positive_fraction: input("0.5", ""),
            boundary_fraction: input("0", ""),
            boundary_weight: input("1", ""),
            boundary_radius: input("0.5", ""),
            extra: input("", "например --patch-points 8192"),
            root,
            _tick: tick,
            _quit: quit,
        };
        this.select_best_trial();
        this._note_sub = Some(cx.subscribe(
            &this.note,
            |this, input, ev: &gpui_kit::component::input::InputEvent, cx| {
                if matches!(ev, gpui_kit::component::input::InputEvent::Change) {
                    let text = input.read(cx).value().to_string();
                    this.save_note(text);
                }
            },
        ));
        this._scenario_sub = Some(cx.subscribe(
            &this.scenario_name,
            |this, input, ev: &gpui_kit::component::input::InputEvent, cx| {
                use gpui_kit::component::input::InputEvent as E;
                if matches!(ev, E::PressEnter { .. } | E::Blur) {
                    let to = input.read(cx).value().to_string();
                    this.rename_scenario(to);
                    cx.notify();
                }
            },
        ));
        this.check_env(cx);
        Theme::change(
            if dark {
                ThemeMode::Dark
            } else {
                ThemeMode::Light
            },
            Some(window),
            cx,
        );
        this
    }

    /// Раз в 200 мс: новые строки процессов; раз в 2 с — пересканировать прогоны.
    fn tick(&mut self, rescan: bool, cx: &mut Context<Self>) {
        let (mut dirty, mut setup_done, mut preds_done) = (false, false, vec![]);
        for (i, job) in self.jobs.iter_mut().enumerate() {
            let out = job.out.lock().unwrap();
            let now = Instant::now();
            // seen — номер строки от начала лога; начало буфера сдвигается, когда он переполнился.
            let from = job.seen.saturating_sub(out.dropped);
            if out.lines.len() > from {
                for l in &out.lines[from..] {
                    if let Some((name, target, features)) = parse_run_header(l) {
                        // run_all: финальные прогоны — 1 trial и --final-epochs, остальные — общие --n-trials/--epochs.
                        let n = |k: &str, d: u32| {
                            Job::arg(&job.args, k)
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(d)
                        };
                        (job.n_trials, job.n_epochs) = if name.starts_with("final_") {
                            (1, n("--final-epochs", 30))
                        } else {
                            (n("--n-trials", 25), n("--epochs", 10))
                        };
                        (job.run, job.target, job.features) = (name, target, features);
                        job.epochs.clear();
                        job.baseline = None;
                        job.segment_start = now;
                        job.scenes.clear();
                        job.scene_progress = None;
                        job.patch_progress = None;
                    } else if let Some((trial, epoch, f1, base)) = parse_epoch_line(l) {
                        job.epochs.push(Epoch {
                            trial,
                            epoch,
                            f1,
                            at: now,
                        });
                        job.baseline = base.or(job.baseline);
                    } else if let Some((k, n, name)) = parse_scene_header(l) {
                        // Поля по отдельности: `out` держит ссылку на job.out, методы взяли бы весь job.
                        job.scene_progress = Some((k, n));
                        job.patch_progress = None;
                        job.scenes.push((
                            name,
                            SceneMetrics {
                                f1: f64::NAN,
                                precision: f64::NAN,
                                recall: f64::NAN,
                                baseline: None,
                                uncovered: 0,
                            },
                        ));
                    } else if let Some(m) = parse_scene_line(l) {
                        // Метрики сцены дописываются в последнюю начатую строку: заголовок и метрики идут парами.
                        match job.scenes.last_mut() {
                            Some((_, slot)) => *slot = m,
                            None => job.scenes.push((String::new(), m)),
                        }
                    } else if let Some((i, p)) = parse_patch_line(l) {
                        job.patch_progress = Some((i, p));
                    }
                }
                job.seen = out.dropped + out.lines.len();
                dirty = true;
                if self.autoscroll && i == self.sel_job {
                    self.log_scroll
                        .scroll_to_item(out.lines.len().saturating_sub(1), ScrollStrategy::Top);
                }
            }
            if job.ended.is_none() {
                dirty = true; // секундомер
                if out.exit.is_some() {
                    job.ended = Some(now);
                    setup_done |= job.setup;
                    if let Some(pj) = &job.predict {
                        preds_done.push((pj.clone(), matches!(out.exit, Some(Ok(())))));
                    }
                    if !job.stopped {
                        notify_end(job, &out);
                    }
                }
            }
        }
        let usage = monitor::latest();
        if usage.seq != self.usage.seq {
            self.usage = usage;
            dirty = true;
        }
        if setup_done {
            self.check_env(cx);
        }
        if self
            .active
            .as_ref()
            .is_some_and(|a| a.job.is_some_and(|j| !self.jobs[j].running()))
        {
            self.advance_scenario(cx);
        }
        for (pj, ok) in preds_done {
            if ok {
                self.load_prediction(pj, cx);
            } else if let Some(pv) = self
                .preview
                .as_mut()
                .filter(|p| p.scene == pj.scene)
                .and_then(|p| p.pred.as_mut())
            {
                pv.result = Some(Err(
                    "predict_scene.py завершился с ошибкой — подробности в «Трансляции»".into(),
                ));
            }
        }
        if rescan {
            self.reload_runs(cx);
            if self.tab == Tab::Testing {
                self.refresh_tests();
            }
        }
        self.prune_jobs();
        if dirty {
            cx.notify();
        }
    }

    /// Оставить все идущие процессы и последние завершённые: иначе список процессов и их логи
    /// растут всё время работы окна (сотни МБ за день).
    fn prune_jobs(&mut self) {
        const KEEP_DONE: usize = 10;
        // Индексы в Active.job и confirm_stop указывают в jobs — со сдвигом они стали бы чужими.
        if self.active.is_some() || self.confirm_stop.is_some() {
            return;
        }
        let mut left = self
            .jobs
            .iter()
            .filter(|j| !j.running())
            .count()
            .saturating_sub(KEEP_DONE);
        if left == 0 {
            return;
        }
        let old = std::mem::take(&mut self.jobs);
        // Куда переехал каждый процесс; usize::MAX — выброшен.
        let mut map = vec![usize::MAX; old.len()];
        for (i, j) in old.into_iter().enumerate() {
            if left > 0 && !j.running() {
                left -= 1;
                continue;
            }
            map[i] = self.jobs.len();
            self.jobs.push(j);
        }
        let sel = map.get(self.sel_job).copied().unwrap_or(usize::MAX);
        self.sel_job = if sel == usize::MAX {
            self.jobs.len().saturating_sub(1)
        } else {
            sel
        };
    }

    /// Пересканировать каталог результатов из формы (и графики выбранного trial) в фоне —
    /// диск не блокирует окно.
    fn reload_runs(&mut self, cx: &mut Context<Self>) {
        if self.scanning {
            return;
        }
        let field = self.runs_dir.read(cx).value().trim().to_string();
        // Пустое поле не сканируем: root.join("") — это корень проекта, и его подкаталоги
        // не должны выглядеть прогонами. Кнопка запуска требует непустой каталог.
        if field.is_empty() {
            return;
        }
        self.scanning = true;
        let dir = self.root.join(field);
        let name = self.runs.get(self.sel_run).map(|r| r.name.clone());
        let trial = self.sel_trial.clone();
        let (scan_name, scan_trial) = (name.clone(), trial.clone());
        cx.spawn(async move |this, cx| {
            let (runs, epochs) = cx
                .background_executor()
                .spawn(async move {
                    let runs = scan_runs(&dir);
                    let epochs = match (
                        runs.iter().find(|r| Some(&r.name) == scan_name.as_ref()),
                        &scan_trial,
                    ) {
                        (Some(r), Some(t)) => read_epochs(&r.path, t),
                        _ => vec![],
                    };
                    (runs, epochs)
                })
                .await;
            _ = this.update(cx, |this, cx| {
                this.scanning = false;
                let current = this.runs.get(this.sel_run).map(|r| r.name.clone());
                this.runs = runs;
                this.sel_run = this
                    .runs
                    .iter()
                    .position(|r| Some(&r.name) == current.as_ref())
                    .unwrap_or(0);
                // Выбор мог смениться, пока шло сканирование, — тогда графики этого trial перечитаем.
                // Прогон впервые появился (или сменился) — показать его лучший trial.
                if this.sel_trial.is_none() || current != name {
                    this.select_best_trial();
                } else if this.sel_trial == trial {
                    this.trial_epochs = epochs;
                } else {
                    this.load_trial_epochs();
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn reload_datasets(&mut self, cx: &mut Context<Self>) {
        let root = self.root.clone();
        cx.spawn(async move |this, cx| {
            let datasets = cx
                .background_executor()
                .spawn(async move { scan_datasets(&root) })
                .await;
            _ = this.update(cx, |this, cx| {
                this.datasets = datasets;
                this.sel_dataset = this.sel_dataset.min(this.datasets.len().saturating_sub(1));
                cx.notify();
            });
        })
        .detach();
    }

    fn select_best_trial(&mut self) {
        self.sel_trial = self
            .runs
            .get(self.sel_run)
            .and_then(|r| Some(r.best.as_ref()?.get("best_trial")?.as_u64()?.to_string()));
        self.load_trial_epochs();
    }

    fn load_trial_epochs(&mut self) {
        self.trial_epochs = match (self.runs.get(self.sel_run), &self.sel_trial) {
            (Some(r), Some(t)) => read_epochs(&r.path, t),
            _ => vec![],
        };
    }

    fn set_dark(&mut self, dark: bool, window: &mut Window, cx: &mut Context<Self>) {
        self.dark = dark;
        Theme::change(
            if dark {
                ThemeMode::Dark
            } else {
                ThemeMode::Light
            },
            Some(window),
            cx,
        );
        self.redraw_preview(Some(window), cx);
        cx.notify();
    }

    /// Заполнить редактор значениями (недостающие ключи не трогаются).
    fn apply_params(&mut self, params: &Params, window: &mut Window, cx: &mut Context<Self>) {
        for (i, spec) in SPACE.iter().enumerate() {
            let Some(v) = params.get(spec.key) else {
                continue;
            };
            match &self.param_inputs[i] {
                Some(input) => Self::set_input(input, spec.show(v), window, cx),
                None => self.param_choice[i] = v.clone(),
            }
        }
    }

    /// Значения редактора и ошибки проверки по индексам SPACE.
    fn custom_params(&self, cx: &App) -> (Params, Vec<(usize, String)>) {
        let (mut params, mut errs) = (Params::new(), vec![]);
        for (i, spec) in SPACE.iter().enumerate() {
            let raw = match &self.param_inputs[i] {
                Some(input) => input.read(cx).value().to_string(),
                None => self.param_choice[i].to_string(),
            };
            match spec.parse(&raw) {
                Ok(v) => _ = params.insert(spec.key.into(), v),
                Err(e) => errs.push((i, e)),
            }
        }
        (params, errs)
    }

    fn autopick(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let s = suggest(&self.runs, self.target);
        self.apply_params(&s.params, window, cx);
        let class = class_ru(self.target).to_lowercase();
        self.params_note = Some(if s.of == 0 {
            "Завершённых trials пока нет — выставлены середины диапазонов поиска.".into()
        } else if s.all_classes {
            format!(
                "Для класса «{class}» trials нет: медиана лучших {} из {} trials всех классов ({}).",
                s.used,
                s.of,
                s.runs.join(", ")
            )
        } else {
            format!(
                "Медиана лучших {} из {} завершённых trials класса «{class}» ({}).",
                s.used,
                s.of,
                s.runs.join(", ")
            )
        });
        cx.notify();
    }

    /// Настройки обучения из формы.
    fn train_opts(&self, cx: &App) -> TrainOpts {
        let v = |i: &Entity<InputState>| i.read(cx).value().trim().to_string();
        TrainOpts {
            loss: self.loss.into(),
            gamma: v(&self.focal_gamma),
            lovasz: v(&self.lovasz_weight),
            power: v(&self.weight_power),
            positive: self.equibatch.then(|| v(&self.positive_fraction)),
            bfrac: v(&self.boundary_fraction),
            bweight: v(&self.boundary_weight),
            bradius: v(&self.boundary_radius),
        }
    }

    /// Аргументы командной строки из формы запуска.
    fn command(&self, cx: &App) -> (String, Vec<String>, String) {
        let v = |i: &Entity<InputState>| i.read(cx).value().trim().to_string();
        let mut args = vec!["-u".to_string()];
        let title;
        if self.run_all {
            title = format!("run_all {}", self.tasks.join(" "));
            args.extend(["run_all.py", "--tasks"].map(String::from));
            args.extend(self.tasks.iter().map(|t| t.to_string()));
            args.extend([
                "--data".into(),
                v(&self.data_dir),
                "--final-data".into(),
                v(&self.final_data_dir),
                "--runs".into(),
                v(&self.runs_dir),
                "--final-epochs".into(),
                v(&self.final_epochs),
            ]);
        } else {
            let suffix = if self.custom { "_custom" } else { "" };
            // Study с другими настройками обучения hpo_pointnet.py не продолжит: по умолчанию свой каталог.
            let focal = self.train_opts(cx).suffix();
            let name = Some(v(&self.out_name))
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| format!("{}_{}{focal}{suffix}", self.target, self.features));
            title = name.clone();
            args.extend([
                "hpo_pointnet.py".into(),
                "--data-dir".into(),
                v(&self.data_dir),
            ]);
            let runs = v(&self.runs_dir);
            args.extend(["--output-dir".into(), cli_path(&runs, &name)]);
            args.extend(["--target", self.target, "--features", self.features].map(String::from));
            if self.custom {
                args.extend([
                    "--fixed-params".into(),
                    cli_path(&runs, &format!("{name}.params.json")),
                ]);
            }
        }
        let n_trials = if !self.run_all && self.custom && self.custom_only {
            "1".into()
        } else {
            v(&self.trials)
        };
        args.extend(["--n-trials".into(), n_trials]);
        for (k, i) in [
            ("--epochs", &self.epochs),
            ("--seed", &self.seed),
            ("--workers", &self.workers),
            ("--patch-points", &self.patch_points),
            ("--patches-per-scene", &self.patches_per_scene),
        ] {
            args.extend([k.into(), v(i)]);
        }
        args.extend(self.train_opts(cx).args());
        args.extend(v(&self.extra).split_whitespace().map(String::from));
        (v(&self.python), args, title)
    }

    fn spawn(&mut self, program: String, args: Vec<String>, title: String) {
        let out = Arc::new(Mutex::new(Output::default()));
        let mut cmd = Command::new(&program);
        cmd.args(&args)
            .current_dir(&self.root)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        std::os::windows::process::CommandExt::creation_flags(&mut cmd, 0x0800_0000); // CREATE_NO_WINDOW
        let log_path = std::fs::create_dir_all(self.root.join(LOG_DIR))
            .ok()
            .map(|_| self.root.join(LOG_DIR).join(log_file_name(&title)))
            .filter(|p| {
                std::fs::File::create(p)
                    .map(|f| out.lock().unwrap().file = Some(f))
                    .is_ok()
            });
        out.lock()
            .unwrap()
            .push(format!("$ {program} {}", args.join(" ")));
        let pid = match cmd.spawn() {
            Err(e) => {
                let mut o = out.lock().unwrap();
                o.push(format!("Не удалось запустить {program}: {e}. Проверьте путь к Python на вкладке «Запуск»."));
                o.exit = Some(Err("не запущен".into()));
                None
            }
            Ok(mut child) => {
                let pid = child.id();
                // shortcut: run_all.py сам выбирает каталоги прогонов, его запуски не записываются; нужно — передавать --output-dir из run_all.
                // Пишем только после удачного spawn: иначе неудачный запуск оставляет пустой каталог runs/<имя>.
                if args.iter().any(|a| a == "hpo_pointnet.py")
                    && let Some(dir) = Job::arg(&args, "--output-dir")
                {
                    let (author, host) = who();
                    let secs = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |d| d.as_secs()) as i64;
                    // Содержимое --fixed-params: файл рядом могут переписать, а повтор должен взять ту же конфигурацию.
                    let fixed = Job::arg(&args, "--fixed-params").and_then(|f| {
                        serde_json::from_str::<serde_json::Value>(
                            &std::fs::read_to_string(self.root.join(f)).ok()?,
                        )
                        .ok()
                    });
                    let launch = serde_json::json!({ "author": author, "host": host, "started": utc_stamp(secs), "program": program, "args": args, "fixed_params": fixed });
                    if let Err(e) = record_launch(&self.root, &self.root.join(&dir), launch) {
                        out.lock().unwrap().push(format!("Не удалось записать {dir}/{STUDIO_FILE}: {e} — запуск идёт, но без записи «кто и как»."));
                    }
                }
                let readers = [
                    pipe(child.stdout.take().unwrap(), out.clone()),
                    pipe(child.stderr.take().unwrap(), out.clone()),
                ];
                let out = out.clone();
                std::thread::spawn(move || {
                    let status = child.wait();
                    // Дочитать хвост вывода, но не ждать вечно: потомок, унаследовавший pipe, может пережить процесс.
                    let deadline = Instant::now() + Duration::from_secs(3);
                    while readers.iter().any(|r| !r.is_finished()) && Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    let mut o = out.lock().unwrap();
                    let (exit, msg) = match status {
                        Ok(s) if s.success() => (Ok(()), "=== процесс завершён".to_string()),
                        Ok(s) => {
                            let code = s.code().map_or("?".into(), |c| c.to_string());
                            (
                                Err(format!("код {code}")),
                                format!("=== процесс завершился с кодом {code}"),
                            )
                        }
                        Err(e) => (
                            Err(e.to_string()),
                            format!("=== ошибка ожидания процесса: {e}"),
                        ),
                    };
                    o.push(msg);
                    o.exit = Some(exit);
                });
                Some(pid)
            }
        };
        let n = |k: &str, d: u32| Job::arg(&args, k).and_then(|v| v.parse().ok()).unwrap_or(d);
        let now = Instant::now();
        // predict_scene.py — не обучение: у него сцены и патчи вместо trials и эпох (свой экран трансляции).
        let predicting = args.iter().any(|a| a == "predict_scene.py");
        self.jobs.push(Job {
            run: title.clone(),
            target: Job::arg(&args, "--target").unwrap_or_default(),
            features: Job::arg(&args, "--features").unwrap_or_default(),
            n_trials: n("--n-trials", 20),
            n_epochs: n("--epochs", 8),
            title,
            program,
            args,
            pid,
            out,
            started: now,
            ended: None,
            stopped: false,
            setup: false,
            log_path,
            predict: None,
            seen: 0,
            segment_start: now,
            epochs: vec![],
            baseline: None,
            predicting,
            scenes: vec![],
            scene_progress: None,
            patch_progress: None,
        });
        self.sel_job = self.jobs.len() - 1;
        self.tab = Tab::Live;
    }

    /// Открыть сцену в 3D: чтение .npy в фоне.
    fn open_preview(&mut self, scene: PathBuf, cx: &mut Context<Self>) {
        if self.preview.as_ref().is_some_and(|p| p.scene == scene) {
            return;
        }
        if let Some(img) = self.preview.take().and_then(|p| p.image) {
            cx.drop_image(img, None);
        }
        self.preview = Some(Preview {
            scene: scene.clone(),
            cloud: None,
            view: cloud::View::default(),
            by: cloud::ColorBy::Class,
            image: None,
            last: None,
            dirty: false,
            pred: None,
            shown: Rc::new(std::cell::Cell::new((PREVIEW_W as f32, PREVIEW_H as f32))),
            pending: None,
        });
        cx.spawn(async move |this, cx| {
            let path = scene.clone();
            let loaded = cx
                .background_executor()
                .spawn(async move { cloud::load(&path).map(Arc::new) })
                .await;
            _ = this.update(cx, |this, cx| {
                // Пока грузилось, могли выбрать другую сцену.
                if let Some(p) = this.preview.as_mut().filter(|p| p.scene == scene) {
                    p.cloud = Some(loaded);
                    let pending = p.pending.take();
                    this.redraw_preview(None, cx);
                    if let Some(job) = pending {
                        this.load_prediction(job, cx);
                    }
                }
            });
        })
        .detach();
    }

    /// Модели для проверки сцены: лучший trial каждого прогона, у которого есть best.pt.
    fn models(&self, scene: &Path) -> Vec<Model> {
        let file = scene
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut out: Vec<Model> = self
            .runs
            .iter()
            .filter_map(|r| {
                let trial = r.best.as_ref()?.get("best_trial")?.as_u64()?;
                let checkpoint = [
                    r.path.join("best_model.pt"),
                    r.path.join(format!("trial_{trial:04}")).join("best.pt"),
                ].into_iter().find(|p| p.exists())?;
                let in_list = |k: &str| {
                    r.split
                        .as_ref()?
                        .get(k)?
                        .as_array()
                        .map(|a| a.iter().any(|v| v.as_str() == Some(&file)))
                };
                let split = match (in_list("val"), in_list("train")) {
                    (Some(true), _) => Some("val"),
                    (_, Some(true)) => Some("train"),
                    _ => None,
                };
                checkpoint.exists().then(|| Model {
                    run: r.name.clone(),
                    trial: Some(trial),
                    f1: r.best_f1(),
                    checkpoint,
                    target: r.target().unwrap_or_default(),
                    split,
                })
            })
            .collect();
        for checkpoint in artifact_models(&self.root, &self.runs) {
            if out.iter().any(|m| m.checkpoint == checkpoint) {
                continue;
            }
            let label = checkpoint
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "модель".into());
            out.push(Model {
                run: format!("Файл · {label}"),
                trial: None,
                f1: None,
                checkpoint,
                target: String::new(),
                split: None,
            });
        }
        out.sort_by(|a, b| b.f1.unwrap_or(0.).total_cmp(&a.f1.unwrap_or(0.)));
        out
    }

    /// Проверить сцену превью моделью: готовое предсказание — сразу, иначе процесс predict_scene.py.
    fn predict(&mut self, mut m: Model, cx: &mut Context<Self>) {
        let Some(scene) = self.preview.as_ref().map(|p| p.scene.clone()) else {
            return;
        };
        let request = Arc::new(());
        if m.target.is_empty() {
            if let Some(t) = self.model_targets.get(&m.checkpoint) {
                m.target = t.clone();
            }
        }
        if m.target.is_empty() {
            let python = self.python.read(cx).value().trim().to_string();
            let checkpoint = m.checkpoint.clone();
            if let Some(p) = self.preview.as_mut() {
                p.pending = None;
                p.pred = Some(PredView { request: request.clone(), label: m.run.clone(), split: None, result: None });
            }
            let root = self.root.clone();
            cx.spawn(async move |this, cx| {
                let result = cx.background_executor().spawn(async move {
                    let output = Command::new(python).current_dir(root).args(["predict_scene.py", "--print-target", "--checkpoint"]).arg(&checkpoint).output().map_err(|e| e.to_string())?;
                    if !output.status.success() { return Err(String::from_utf8_lossy(&output.stderr).into_owned()); }
                    let target = String::from_utf8_lossy(&output.stdout).trim().to_owned();
                    if TARGETS.contains(&target.as_str()) { Ok(target) } else { Err("нет target".to_string()) }
                }).await;
                _ = this.update(cx, |this, cx| {
                    if !this.preview.as_ref().is_some_and(|p| p.scene == scene && p.pred.as_ref().is_some_and(|pred| Arc::ptr_eq(&pred.request, &request))) { return; }
                    match result {
                        Ok(target) => {
                            this.model_targets.insert(m.checkpoint.clone(), target.clone());
                            let mut m = m;
                            m.target = target;
                            this.predict(m, cx);
                        }
                        Err(e) => {
                            if let Some(pv) = this.preview.as_mut().and_then(|p| p.pred.as_mut()) {
                                pv.result = Some(Err(format!("Не удалось определить параметры модели: {e}")));
                            }
                            cx.notify();
                        }
                    }
                });
            }).detach();
            return;
        }
        let stem = scene
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        // Предсказание пишется в каталог того прогона, из которого взята модель
        // (поле «Каталог результатов» — только запасной вариант).
        let base = self
            .runs
            .iter()
            .find(|r| r.name == m.run)
            .map(|r| r.path.clone())
            .unwrap_or_else(|| {
                self.root.join("tests").join(format!("preview_{}", m.checkpoint.file_stem().unwrap_or_default().to_string_lossy()))
            });
        let name = prediction_cache_name(&scene, &m.checkpoint);
        remove_stale_predictions(&base.join("pred"), &name);
        let out = base.join("pred").join(name);
        let job = PredJob {
            scene: scene.clone(),
            out: out.clone(),
            target: m.target.clone(),
            request: request.clone(),
        };
        if let Some(p) = self.preview.as_mut() {
            p.pending = None;
            p.pred = Some(PredView {
                request,
                label: m.label(),
                split: m.split,
                result: None,
            });
        }
        if out.exists() {
            self.load_prediction(job, cx);
            return;
        }
        if let Err(e) = self.ensure_predict_script() {
            if let Some(pv) = self.preview.as_mut().and_then(|p| p.pred.as_mut()) {
                pv.result = Some(Err(e));
            }
            return;
        }
        let python = self.python.read(cx).value().trim().to_string();
        let args = [
            "-u",
            "predict_scene.py",
            "--checkpoint",
            &m.checkpoint.to_string_lossy(),
            "--scene",
            &scene.to_string_lossy(),
            "--out",
            &out.to_string_lossy(),
        ]
        .map(String::from)
        .to_vec();
        let source_tab = self.tab;
        self.spawn(python, args, format!("Предсказание {} · {stem}", m.run));
        if let Some(j) = self.jobs.last_mut() {
            j.predict = Some(job);
            j.target = m.target.clone();
        }
        self.tab = source_tab; // остаёмся у превью; лог — в «Трансляции»
        cx.notify();
    }

    /// Прочитать готовое предсказание в фоне и показать «Ошибки», если сцена всё ещё открыта.
    fn load_prediction(&mut self, pj: PredJob, cx: &mut Context<Self>) {
        if !self.preview.as_ref().is_some_and(|p| p.scene == pj.scene && p.pred.as_ref().is_some_and(|pred| Arc::ptr_eq(&pred.request, &pj.request))) {
            return;
        }
        let Some(Some(Ok(cloud))) = self
            .preview
            .as_ref()
            .filter(|p| p.scene == pj.scene)
            .map(|p| p.cloud.clone())
        else {
            if let Some(p) = self
                .preview
                .as_mut()
                .filter(|p| p.scene == pj.scene && p.cloud.is_none())
            {
                p.pending = Some(pj);
            }
            return;
        };
        cx.spawn(async move |this, cx| {
            let target = pj.target.clone();
            let out = pj.out.clone();
            let res = cx
                .background_executor()
                .spawn(async move { cloud::load_pred(&out, &cloud, &target) })
                .await;
            _ = this.update(cx, |this, cx| {
                let Some(p) = this.preview.as_mut().filter(|p| p.scene == pj.scene && p.pred.as_ref().is_some_and(|pred| Arc::ptr_eq(&pred.request, &pj.request))) else {
                    return;
                };
                if res.is_ok() {
                    p.by = cloud::ColorBy::Errors;
                }
                if let Some(pv) = p.pred.as_mut() {
                    pv.result = Some(res);
                }
                p.dirty = true;
                cx.notify();
            });
        })
        .detach();
    }

    /// Перерисовать картинку превью (вид, цвет или тема поменялись); старую текстуру освободить.
    fn redraw_preview(&mut self, window: Option<&mut Window>, cx: &mut Context<Self>) {
        let bg = if self.dark { 0x0E1013 } else { 0xF6F6F7 }; // = Pal::bg, чтобы картинка сливалась с панелью
        let Some(p) = self.preview.as_mut() else {
            return;
        };
        p.dirty = false;
        let Some(Ok(c)) = &p.cloud else { return };
        // Пока тянут мышью — половинное разрешение и каждая 3-я точка; отпустили — полное качество.
        let (w, h, stride) = if p.last.is_some() {
            (PREVIEW_W / 2, PREVIEW_H / 2, 3)
        } else {
            (PREVIEW_W, PREVIEW_H, 1)
        };
        let outcome = p
            .pred
            .as_ref()
            .and_then(|x| x.result.as_ref()?.as_ref().ok())
            .map(|o| o.sample.as_slice());
        let px = cloud::render(c, &p.view, w, h, stride, p.by, outcome, bg, class_hex);
        let frame = image::Frame::new(image::RgbaImage::from_raw(w, h, px).expect("размер буфера"));
        if let Some(old) = p.image.replace(Arc::new(RenderImage::new(vec![frame]))) {
            cx.drop_image(old, window);
        }
        cx.notify();
    }

    /// Проверить окружение Python в фоне (запуск python/nvidia-smi занимает секунды).
    fn check_env(&mut self, cx: &mut Context<Self>) {
        if self.env_checking {
            return;
        }
        self.env_checking = true;
        let root = self.root.clone();
        cx.spawn(async move |this, cx| {
            let info = cx
                .background_executor()
                .spawn(async move { env::detect(&root) })
                .await;
            _ = this.update(cx, |this, cx| {
                this.env_checking = false;
                // Окружение появилось — подставить его Python в «Запуск», если там общий python.
                if let Some(v) = &info.venv {
                    if info
                        .packages
                        .as_ref()
                        .is_some_and(|p| p.as_ref().is_ok_and(|p| p.missing().is_empty()))
                    {
                        this.pending_python = Some(v.to_string_lossy().into_owned());
                    }
                }
                this.env = Some(info);
                cx.notify();
            });
        })
        .detach();
    }

    /// Установочная задача: процесс с живым логом в «Трансляции».
    fn spawn_setup(&mut self, program: String, args: Vec<String>, title: &str) {
        self.spawn(program, args, title.into());
        if let Some(j) = self.jobs.last_mut() {
            j.setup = true;
        }
    }

    fn install_python(&mut self) {
        if let Some((p, a)) = self.env.as_ref().and_then(env::install_python_cmd) {
            self.spawn_setup(p, a, "Установка Python 3.11");
        }
    }

    /// venv + pip + torch нужной сборки + pyg-lib + остальное — одним процессом (шаги по очереди).
    fn install_packages(&mut self) {
        let Some(info) = &self.env else { return };
        let build = self.build.unwrap_or_else(|| info.recommended_build());
        let base = info.base.as_ref().map(|b| b.0.clone());
        let steps = env::setup_steps(&self.root, base.as_deref(), info.venv.is_some(), build);
        let runner = info
            .venv
            .clone()
            .or(base)
            .map(|p| p.to_string_lossy().into_owned());
        let Some(runner) = runner else { return };
        let args = vec![
            "-u".into(),
            "-c".into(),
            env::RUNNER.into(),
            serde_json::to_string(&steps).unwrap(),
        ];
        self.spawn_setup(
            runner,
            args,
            &format!("Окружение: torch {}+{build}", env::TORCH),
        );
    }

    /// Класс и признаки формы — из прогона (split.json → best.json).
    /// Мини-копия выбранного датасета (жёсткие ссылки) и выбор её для запуска.
    fn make_mini(&mut self, cx: &mut Context<Self>) {
        let Some(d) = self.datasets.get(self.sel_dataset) else {
            return;
        };
        let d = d.clone();
        self.ds_note = Some((true, format!("Создаю {}_mini…", d.name)));
        cx.spawn(async move |this, cx| {
            let res = cx.background_executor().spawn(async move { make_mini(&d, MINI_SCENES) }).await;
            _ = this.update(cx, |this, cx| {
                match res {
                    Ok(dir) => {
                        let name = dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                        this.ds_note = Some((true, format!(
                            "Готово: {name} — {MINI_SCENES} сцен ({} train / {} val). Выбран в «Запуске»; «Быстрая проверка» там же — 1 trial, 1 эпоха.",
                            MINI_SCENES - MINI_SCENES.div_ceil(5),
                            MINI_SCENES.div_ceil(5)
                        )));
                        this.pending_data = Some(name);
                    }
                    Err(e) => this.ds_note = Some((false, format!("Мини-копия не создана: {e}"))),
                }
                this.reload_datasets(cx);
            });
        })
        .detach();
    }

    /// Отчёт → новая заметка Nevo через его MCP-мост. Повторный экспорт создаёт ещё одну заметку.
    fn export_to_nevo(&mut self, title: String, path: PathBuf, cx: &mut Context<Self>) {
        let Some(markdown) = read_artifact(&path) else {
            self.nevo_note = Some((false, format!("Не читается: {}", path.display())));
            return cx.notify();
        };
        self.nevo_note = Some((true, format!("Отправляю «{title}» в Nevo…")));
        cx.notify();
        cx.spawn(async move |this, cx| {
            let t = title.clone();
            let res = cx.background_executor().spawn(async move { nevo::export(&t, &markdown) }).await;
            _ = this.update(cx, |this, cx| {
                this.nevo_note = Some(match res {
                    Ok(_) => (true, format!("Заметка «{title}» создана в Nevo.")),
                    Err(e) => (false, format!("Экспорт в Nevo не удался: {e}")),
                });
                cx.notify();
            });
        })
        .detach();
    }

    /// Прогон в корзину: папка runs/<имя> и её runs/<имя>.params.json. Восстановить можно из корзины ОС.
    fn delete_run(&mut self, idx: usize, cx: &mut Context<Self>) {
        let Some(r) = self.runs.get(idx) else { return };
        let name = r.name.clone();
        // Конфликт — по фактическому каталогу прогона (может лежать вне «runs» по умолчанию).
        let run_dir = r
            .path
            .strip_prefix(&self.root)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| r.path.to_string_lossy().replace('\\', "/"));
        if let Some(c) = self.conflict(Some(&run_dir)) {
            self.delete_error = Some(format!("Нельзя удалить {name}: {c}"));
            return;
        }
        // params-файл лежит рядом с каталогом прогона — его создают финальный прогон и повтор.
        let params = r
            .path
            .parent()
            .map(|p| p.join(format!("{name}.params.json")));
        let targets: Vec<PathBuf> = [Some(r.path.clone()), params.filter(|p| p.exists())]
            .into_iter()
            .flatten()
            .collect();
        match trash::delete_all(&targets) {
            Err(e) => {
                self.delete_error = Some(format!(
                    "Не удалось переместить {name} в корзину: {e}. Ничего не удалено."
                ))
            }
            Ok(()) => {
                self.delete_error = None;
                self.runs.remove(idx);
                self.sel_run = self.sel_run.min(self.runs.len().saturating_sub(1));
                self.select_best_trial();
                self.reload_runs(cx);
            }
        }
        self.confirm_delete = None;
        cx.notify();
    }

    fn start_final_run(&mut self, idx: usize, cx: &mut Context<Self>) {
        let Some(run) = self.runs.get(idx) else {
            return;
        };
        let Some(best) = run.best.as_ref() else {
            self.tab_error = Some("У прогона нет best.json — финальный прогон невозможен.".into());
            return;
        };
        if best.get("params").map_or(true, |p| !p.is_object()) {
            self.tab_error = Some(
                "В best.json нет параметров лучшего trial — финальный прогон невозможен.".into(),
            );
            return;
        }
        let final_data = self.final_data_dir.read(cx).value().trim().to_string();
        if final_data.is_empty() || !self.root.join(&final_data).is_dir() {
            self.tab_error =
                Some("Выберите существующий каталог данных для финального прогона.".into());
            return;
        }
        let epochs = self.final_epochs.read(cx).value().trim().to_string();
        let workers = self.workers.read(cx).value().trim().to_string();
        let runs_field = self.runs_dir.read(cx).value().trim().to_string();
        let params_path = cli_path(&runs_field, &format!("{}.params.json", run.name));
        let output = cli_path(&runs_field, &format!("final_{}", run.name));
        let title = format!("final_{}", run.name);
        // Все проверки — до записи params-файла: неудачная попытка не оставляет артефакт,
        // который заблокирует повтор (файл намеренно не перезаписывается).
        // Каталог проверяется целиком (как в run_all.py): даже без study.db он занят.
        if self.jobs.iter().any(|j| j.running() && j.title == title)
            || self.conflict(Some(&output)).is_some()
            || self.root.join(&output).exists()
        {
            self.tab_error = Some(format!(
                "Финальный прогон final_{} уже занят или существует. Выберите новый каталог результатов.",
                run.name
            ));
            return;
        }
        let params = best
            .get("params")
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        // Python проверяется до записи params-файла: неудачный spawn не оставит артефакт,
        // который заблокирует повтор (файл намеренно не перезаписывается).
        let program = self.python.read(cx).value().trim().to_string();
        if program.is_empty()
            || ((program.contains('/') || program.contains('\\')) && !Path::new(&program).exists())
        {
            self.tab_error = Some(
                "Проверьте путь к Python на вкладке «Запуск» — финальный прогон не запущен.".into(),
            );
            return;
        }
        let params_on_disk = self.root.join(&params_path);
        if params_on_disk.exists() {
            self.tab_error = Some(format!(
                "Файл {} уже существует; он не был перезаписан.",
                params_on_disk.display()
            ));
            return;
        }
        if let Err(e) = params_on_disk
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|_| {
                std::fs::write(
                    &params_on_disk,
                    serde_json::to_vec_pretty(&params).unwrap_or_default(),
                )
            })
        {
            self.tab_error = Some(format!("Не удалось записать параметры финала: {e}"));
            return;
        }
        let args = final_run_args(
            best,
            &TrainOpts::from_info(&run.loss(), |k| json_scalar(best.get(k))),
            &run.target().unwrap_or_default(),
            &run.features().unwrap_or_else(|| "xyz".into()),
            &final_data,
            &output,
            &params_path,
            &epochs,
            &workers,
        );
        self.spawn(program, args, title);
        cx.notify();
    }

    fn form_from_run(&mut self, idx: usize, window: &mut Window, cx: &mut Context<Self>) {
        let r = &self.runs[idx];
        // run_info пишется в split.json (и повторяется в best.json): чем именно обучался прогон.
        let info = |k: &str| {
            [&r.split, &r.best]
                .into_iter()
                .flatten()
                .find_map(|v| match v.get(k)? {
                    serde_json::Value::String(s) => Some(s.clone()),
                    v => v.is_number().then(|| v.to_string()),
                })
        };
        let (name, loss, target, features) = (r.name.clone(), r.loss(), r.target(), r.features());
        let (data, seed, epochs, patch_points, patches_per_scene) = (
            info("data_dir"),
            info("seed"),
            info("epochs"),
            info("patch_points"),
            info("patches_per_scene"),
        );
        let t = TrainOpts::from_info(&loss, &info);
        self.loss = match t.loss.as_str() {
            "focal" => "focal",
            "focal_lovasz" => "focal_lovasz",
            _ => "ce",
        };
        self.equibatch = t.positive.is_some();
        for (input, v) in [
            (&self.focal_gamma, t.gamma),
            (&self.lovasz_weight, t.lovasz),
            (&self.weight_power, t.power),
            (&self.positive_fraction, t.positive.unwrap_or_else(|| "0.5".into())),
            (&self.boundary_fraction, t.bfrac),
            (&self.boundary_weight, t.bweight),
            (&self.boundary_radius, t.bradius),
        ] {
            Self::set_input(input, v, window, cx);
        }
        if let Some(t) = target.and_then(|t| TARGETS.iter().find(|x| **x == t)) {
            self.target = t;
        }
        if let Some(f) = features.and_then(|f| FEATURES.iter().find(|x| **x == f)) {
            self.features = f;
        }
        // Продолжение пишется в тот же каталог и под тем же именем: иначе --output-dir укажет
        // на другой study, и «продолжение» молча создаст новый.
        if let Some(d) = r
            .path
            .parent()
            .and_then(|p| p.strip_prefix(&self.root).ok())
            .map(|p| p.to_string_lossy().replace('\\', "/"))
        {
            Self::set_input(&self.runs_dir, d, window, cx);
        }
        Self::set_input(&self.out_name, name.clone(), window, cx);
        // Продолжение study: --data-dir обязателен, а другие данные, seed или бюджет trial делают
        // trials несравнимыми (hpo_pointnet.py сверяет только версию пространства и loss).
        let (data_dir, seed_input, epochs_input) = (
            self.data_dir.clone(),
            self.seed.clone(),
            self.epochs.clone(),
        );
        let (points_input, patches_input) =
            (self.patch_points.clone(), self.patches_per_scene.clone());
        let mut changed: Vec<&str> = vec![];
        let mut missing: Option<String> = None;
        for (label, want, input) in [
            ("данные", data, &data_dir),
            ("seed", seed.clone(), &seed_input),
            ("эпохи на trial", epochs.clone(), &epochs_input),
            ("точки в патче", patch_points.clone(), &points_input),
            ("патчи на сцену", patches_per_scene.clone(), &patches_input),
        ] {
            let Some(v) =
                want.filter(|v| !v.is_empty() && v.as_str() != input.read(cx).value().trim())
            else {
                continue;
            };
            // Датасет мог жить на машине коллеги: абсолютный путь туда оставляем в предупреждении.
            let p = PathBuf::from(&v);
            if label == "данные" && p.is_absolute() && !p.exists() {
                missing = Some(show_path(&p));
                continue;
            }
            Self::set_input(input, v, window, cx);
            changed.push(label);
        }
        let mut note = vec![];
        if !changed.is_empty() {
            note.push(format!("Из прогона {name} взяты {} — иначе продолженный study смешал бы несравнимые trials.", changed.join(", ")));
        }
        if let Some(p) = missing {
            note.push(format!(
                "Каталог прогона {p} на этой машине недоступен — выберите данные вручную."
            ));
        }
        for (label, value) in [("seed", seed), ("эпохи на trial", epochs), ("точки в патче", patch_points), ("патчи на сцену", patches_per_scene)] {
            if value.is_none() {
                note.push(format!("В старом артефакте не записан параметр «{label}» — продолжение в прежнем study заблокировано."));
            }
        }
        self.params_note = (!note.is_empty()).then(|| note.join(" "));
        self.run_all = false;
    }

    /// Почему нельзя запускать в `out_dir` (None — сценарий run_all): туда уже пишет идущий процесс.
    fn conflict(&self, out_dir: Option<&str>) -> Option<String> {
        self.jobs.iter().filter(|j| j.running()).find_map(|j| {
            // У обучения каталог задаёт --output-dir (любой глубины, сравнивается как есть),
            // у predict_scene.py — --out-dir/--out или сам файл предсказания (поднимаем до прогона).
            let mut busy: Vec<String> = j
                .predict
                .iter()
                .map(|p| pred_run_path(&self.root, &p.out))
                .collect();
            for k in ["--output-dir"] {
                if let Some(a) = Job::arg(&j.args, k) {
                    busy.push(norm_path(&self.root, &PathBuf::from(a)));
                }
            }
            for k in ["--out-dir", "--out"] {
                if let Some(a) = Job::arg(&j.args, k) {
                    busy.push(pred_run_path(&self.root, &PathBuf::from(a)));
                }
            }
            if busy.is_empty() {
                // run_all.py пишет в <runs>/<текущий прогон>: каталог берём из его аргументов.
                let runs = Job::arg(&j.args, "--runs").map_or_else(|| "runs".into(), |s| s.replace('\\', "/"));
                busy.push(format!("{}/{}", runs.trim_end_matches('/'), j.run));
            }
            match out_dir {
                None => Some(format!("Идёт «{}» — run_all может писать в тот же прогон. Дождитесь окончания или остановите его.", j.title)),
                Some(d) if busy.iter().any(|b| b == d) || j.args.iter().any(|a| a == "run_all.py") => {
                    Some(format!("В {d} может писать идущий процесс «{}». Дождитесь окончания или задайте другое имя.", j.title))
                }
                _ => None,
            }
        })
    }

    fn save_note(&mut self, text: String) {
        let Some(r) = self
            .runs
            .iter_mut()
            .find(|r| Some(&r.name) == self.note_for.as_ref())
        else {
            return;
        };
        let stored = r
            .studio
            .as_ref()
            .and_then(|s| s.get("note")?.as_str())
            .unwrap_or_default();
        if stored == text {
            return;
        }
        match update_studio(&r.path, |v| v["note"] = text.clone().into()) {
            Ok(()) => {
                r.studio.get_or_insert_with(|| serde_json::json!({}))["note"] = text.into();
                self.tab_error = None;
            }
            Err(e) => self.tab_error = Some(format!("Заметка не сохранилась: {e}")),
        }
    }

    /// Новый прогон <имя>_repeat[N] с той же командой, что и последний запуск, и тем же --fixed-params.
    /// Python — текущий из «Запуска»: записанный путь может быть с машины коллеги.
    fn repeat_run(&mut self, idx: usize, cx: &App) {
        self.tab_error = None;
        let r = &self.runs[idx];
        let Some(last) = r
            .studio
            .as_ref()
            .and_then(|s| s["launches"].as_array()?.last().cloned())
        else {
            return;
        };
        let mut args: Vec<String> =
            serde_json::from_value(last["args"].clone()).unwrap_or_default();
        // Повтор лежит рядом с исходным прогоном — в его каталоге результатов.
        let dir = r
            .path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.root.join("runs"));
        let dir_str = dir
            .strip_prefix(&self.root)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| dir.to_string_lossy().replace('\\', "/"));
        let name = (1..)
            .map(|n| {
                if n == 1 {
                    format!("{}_repeat", r.name)
                } else {
                    format!("{}_repeat{n}", r.name)
                }
            })
            .find(|n| !dir.join(n).exists())
            .unwrap();
        let out_dir = cli_path(&dir_str, &name);
        if let Some(e) = self.conflict(Some(&out_dir)) {
            self.tab_error = Some(e);
            return;
        }
        set_arg(&mut args, "--output-dir", out_dir);
        if !last["fixed_params"].is_null() {
            let f = cli_path(&dir_str, &format!("{name}.params.json"));
            if let Err(e) = std::fs::write(
                self.root.join(&f),
                serde_json::to_vec_pretty(&last["fixed_params"]).unwrap(),
            ) {
                self.tab_error = Some(format!("Не удалось записать {f}: {e}"));
                return;
            }
            set_arg(&mut args, "--fixed-params", f);
        }
        let python = self.python.read(cx).value().trim().to_string();
        self.spawn(python, args, name);
    }

    /// Перезапуск с теми же аргументами, если он ещё сделает то же самое.
    fn restart(&mut self, idx: usize) {
        let j = &self.jobs[idx];
        let dir = Job::arg(&j.args, "--output-dir");
        let refusal = self.conflict(dir.as_deref()).or_else(|| {
            // --fixed-params ставится первым trial только в пустой study: при перезапуске Optuna выбрала бы другую конфигурацию.
            let fixed = Job::arg(&j.args, "--fixed-params")?;
            self.root.join(dir.as_deref()?).join("study.db").exists().then(|| {
                format!("study.db уже создан, поэтому {fixed} не применится. Запустите конфигурацию заново из «Запуска» с новым именем.")
            })
        });
        if let Some(r) = refusal {
            self.jobs[idx]
                .out
                .lock()
                .unwrap()
                .push(format!("=== перезапуск отменён: {r}"));
            return;
        }
        let (p, a, t) = (j.program.clone(), j.args.clone(), j.title.clone());
        // Признаки задачи не выводятся из аргументов: без них перезапущенная установка нарисуется как обучение,
        // а результат перезапущенного предсказания не попадёт в превью.
        let (setup, predict, target) = (j.setup, j.predict.clone(), j.target.clone());
        self.spawn(p, a, t);
        if let Some(j) = self.jobs.last_mut() {
            j.setup = setup;
            j.target = target;
            j.predict = predict;
        }
    }

    fn set_input(
        input: &Entity<InputState>,
        v: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        input.update(cx, |s, cx| s.set_value(v, window, cx));
    }

    // ---------- трансляция ----------

    /// Трансляция predict_scene.py: сцены и патчи вместо trials и эпох.
    /// У обучения счёт — F1 эпохи, у инференса — F1 последней посчитанной сцены.
    fn render_predicting(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let j = &self.jobs[self.sel_job];
        let (label, chip_bg, chip_fg) = j.status(&p);
        let running = j.running();
        let band = class_color(&j.target);
        let done = j.scenes.iter().filter(|(_, m)| !m.f1.is_nan()).count();
        let total = j
            .scene_progress
            .map_or(j.scenes.len(), |(_, n)| n as usize)
            .max(done);
        let last = j
            .scenes
            .iter()
            .rev()
            .find(|(_, m)| !m.f1.is_nan())
            .map(|(n, m)| (n.clone(), m.clone()));
        let score = last.as_ref().map(|(_, m)| m.f1);
        let line = last
            .as_ref()
            .and_then(|(_, m)| m.baseline)
            .map(|b| b + MARGIN);
        let diff = score.zip(line).map(|(s, l)| s - l);
        // Лучшая сцена: инференс не улучшает модель, поэтому «лучший в прогоне» — это лучшая сцена.
        let best = j
            .scenes
            .iter()
            .filter(|(_, m)| !m.f1.is_nan())
            .map(|(_, m)| m.f1)
            .fold(f64::NEG_INFINITY, f64::max);
        let best = best.is_finite().then_some(best);
        let patches = j.patch_progress;
        let (title, features) = (j.run.clone(), j.features.clone());
        let elapsed = hms(j.elapsed());
        let scenes: Vec<(String, SceneMetrics)> = j
            .scenes
            .iter()
            .filter(|(_, m)| !m.f1.is_nan())
            .cloned()
            .collect();
        let dark = self.dark;
        let (n_lines, dropped) = {
            let out = j.out.lock().unwrap();
            (out.lines.len(), out.dropped)
        };
        let autoscroll = self.autoscroll;

        let header = class_card(band, p)
            .child(v_flex().min_w_0().child(display(&title, 34., p.ink)).child(
                div().text_sm().child(format!(
                    "Предсказание модели · {}",
                    if features.is_empty() {
                        "—".into()
                    } else {
                        features
                    }
                )),
            ))
            .child(chip(label, chip_bg, chip_fg, running))
            .child(div().flex_1())
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        band_button("copy", Lucide::Copy, "Копировать лог", p.raised, p.ink)
                            .on_click(cx.listener(|this, _, _, cx| {
                                let text =
                                    this.jobs[this.sel_job].out.lock().unwrap().lines.join("\n");
                                cx.write_to_clipboard(ClipboardItem::new_string(text));
                            })),
                    )
                    .when_some(self.jobs[self.sel_job].log_path.clone(), |d, path| {
                        d.child(
                            band_button("log-file", Lucide::FileText, "Файл лога", p.raised, p.ink)
                                .on_click(move |_, _, _| open_path(&path)),
                        )
                    })
                    .when(running, |d| {
                        let armed = self.confirm_stop.is_some_and(|(i, t)| {
                            i == self.sel_job && t.elapsed() < Duration::from_secs(5)
                        });
                        let b = if armed {
                            band_button(
                                "stop",
                                Lucide::Square,
                                "Ещё раз — остановить",
                                p.loss,
                                rgb(0xFFFFFF).into(),
                            )
                        } else {
                            band_button("stop", Lucide::Square, "Остановить", p.raised, p.loss)
                        };
                        d.child(b.on_click(cx.listener(move |this, _, _, cx| {
                            if armed {
                                let j = &mut this.jobs[this.sel_job];
                                j.stopped = true;
                                kill(j);
                                this.confirm_stop = None;
                            } else {
                                this.confirm_stop = Some((this.sel_job, Instant::now()));
                            }
                            cx.notify();
                        })))
                    })
                    .when(!running, |d| {
                        d.child(
                            band_button("restart", Lucide::RotateCcw, "Перезапустить", p.ink, p.bg)
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.restart(this.sel_job);
                                    cx.notify();
                                })),
                        )
                    }),
            );

        let scoreline = card(p)
            .mx_6()
            .mt_3()
            .px_5()
            .py_4()
            .gap_10()
            .items_end()
            .child(
                v_flex()
                    .child(small_caps("F1 последней сцены", p.muted))
                    .child(
                        div()
                            .font_family(DISPLAY)
                            .font_weight(FontWeight::BOLD)
                            .text_size(px(64.))
                            .line_height(px(64.))
                            .font_features(tnum())
                            .text_color(p.ink)
                            .child(fmt(score)),
                    ),
            )
            .child(
                v_flex()
                    .gap_1()
                    .child(stat_line(
                        "Линия вылета",
                        line.map_or("—".into(), |l| format!("{l:.4}")),
                        p.ink,
                        p,
                    ))
                    .child(stat_line(
                        "Отрыв",
                        diff.map_or("—".into(), |d| format!("{d:+.4}")),
                        match diff {
                            Some(d) if d >= 0. => p.win,
                            Some(_) => p.loss,
                            None => p.ink,
                        },
                        p,
                    ))
                    .child(stat_line("Лучшая сцена", fmt(best), p.ink, p)),
            )
            .child(
                v_flex()
                    .gap_1()
                    .when_some(j.scene_progress, |d, (k, n)| {
                        d.child(stat_line("Сцена", format!("{k} из {n}"), p.ink, p))
                    })
                    .when_some(patches, |d, (i, n)| {
                        d.child(stat_line("Патчи", format!("{i} из {n}"), p.ink, p))
                    })
                    .when(total > 1, |d| {
                        d.child(stat_line(
                            "Посчитано сцен",
                            format!("{done} из {total}"),
                            p.ink,
                            p,
                        ))
                    })
                    .when(j.scene_progress.is_none() && patches.is_none(), |d| {
                        d.child(stat_line("Сцена", "—".into(), p.ink, p))
                    }),
            )
            .child(div().flex_1())
            .child(
                v_flex()
                    .items_end()
                    .gap_1()
                    .child(compact_stat_line("Идёт", elapsed, p.ink, p))
                    .child(compact_stat_line(
                        "Сцена",
                        last.as_ref().map_or("—".into(), |(n, _)| n.clone()),
                        p.muted,
                        p,
                    )),
            );

        // Таблица сцен: как «Таблица тура», только строки — сцены, а не trials.
        let mut table = v_flex()
            .id("scenes")
            .w(px(500.))
            .h_full()
            .flex_shrink_0()
            .rounded_xl()
            .bg(p.surface)
            .border_1()
            .border_color(p.rule)
            .overflow_y_scrollbar()
            .child(
                div()
                    .px_4()
                    .pt_4()
                    .pb_2()
                    .child(display("Сцены", 22., p.ink)),
            )
            .child(scene_row(
                ["#", "Сцена", "F1", "Точ.", "Полн."].map(String::from),
                true,
                p,
            ));
        for (i, (name, m)) in scenes.iter().enumerate() {
            table = table.child(
                scene_row(
                    [
                        format!("{}", i + 1),
                        name.replace(".npy", ""),
                        format!("{:.4}", m.f1),
                        format!("{:.4}", m.precision),
                        format!("{:.4}", m.recall),
                    ],
                    false,
                    p,
                )
                .when(m.baseline.is_some_and(|b| m.f1 >= b + MARGIN), |d| {
                    d.text_color(p.win).font_weight(FontWeight::SEMIBOLD)
                })
                .when(m.baseline.is_some_and(|b| m.f1 < b + MARGIN), |d| {
                    d.text_color(p.loss)
                }),
            );
        }
        if scenes.is_empty() {
            table = table.child(div().px_4().py_2().text_color(p.muted).child(if running {
                "Первая сцена ещё считается."
            } else {
                "Ни одна сцена не посчитана — подробности в ленте событий."
            }));
        }

        let pts: Vec<(u32, f64)> = scenes
            .iter()
            .enumerate()
            .map(|(i, (_, m))| (i as u32 + 1, m.f1))
            .collect();
        let chart = v_flex()
            .h(px(220.))
            .flex_shrink_0()
            .p_4()
            .rounded_xl()
            .bg(p.surface)
            .border_1()
            .border_color(p.rule)
            .child(chart_title(
                "F1 по сценам · у каждой сцены свой baseline",
                None,
                p,
            ))
            .child(if pts.is_empty() {
                div()
                    .flex_1()
                    .text_color(p.muted)
                    .child("График появится после первой сцены.")
                    .into_any_element()
            } else {
                f1_chart(
                    "live-scenes",
                    pts,
                    None,
                    None,
                    if dark { p.live } else { p.live_ink },
                )
            });

        let log = v_flex()
            .flex_1()
            .min_h_0()
            .gap_2()
            .child(
                h_flex()
                    .gap_2()
                    .child(small_caps("Лента событий", p.muted))
                    .when(dropped > 0, |d| {
                        d.child(div().text_xs().text_color(p.muted).child(format!(
                            "в памяти последние {n_lines} строк — полный лог в файле"
                        )))
                    })
                    .child(div().flex_1())
                    .child(
                        choice("auto", "Автопрокрутка", autoscroll).on_click(cx.listener(
                            |this, _, _, cx| {
                                this.autoscroll = !this.autoscroll;
                                cx.notify();
                            },
                        )),
                    ),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .rounded_xl()
                    .bg(p.surface)
                    .border_1()
                    .border_color(p.rule)
                    .p_3()
                    .child(
                        uniform_list(
                            "log",
                            n_lines,
                            cx.processor(move |this, range: std::ops::Range<usize>, _, _| {
                                let p = pal(dark);
                                let out = this.jobs[this.sel_job].out.lock().unwrap();
                                range
                                    .filter_map(|i| out.lines.get(i).map(|l| (i, l)))
                                    .map(|(i, l)| {
                                        let bad = ["Error", "ERROR", "Traceback", "exception"]
                                            .iter()
                                            .any(|k| l.contains(k));
                                        let score = l.contains(" F1=");
                                        div()
                                            .id(i)
                                            .h(px(LINE_H))
                                            .whitespace_nowrap()
                                            .text_color(if bad {
                                                p.loss
                                            } else if l.starts_with("===") || l.starts_with('$') {
                                                p.win
                                            } else if score {
                                                p.ink
                                            } else {
                                                p.muted
                                            })
                                            .when(score, |d| d.font_weight(FontWeight::BOLD))
                                            .child(l.clone())
                                    })
                                    .collect()
                            }),
                        )
                        .track_scroll(&self.log_scroll)
                        .size_full()
                        .font_family(MONO)
                        .text_xs(),
                    ),
            );

        v_flex()
            .size_full()
            .child(header)
            .child(scoreline)
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .px_6()
                    .pt_3()
                    .pb_6()
                    .gap_3()
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .h_full()
                            .gap_3()
                            .child(chart)
                            .child(log),
                    )
                    .child(table),
            )
            .into_any_element()
    }

    fn render_live(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let Some(j) = self.jobs.get(self.sel_job) else {
            let last = self
                .runs
                .iter()
                .filter(|r| r.best_f1().is_some())
                .max_by(|a, b| a.best_f1().unwrap().total_cmp(&b.best_f1().unwrap()));
            return v_flex()
                .size_full()
                .justify_center()
                .px(px(64.))
                .gap_4()
                .child(display("Сейчас ничего не обучается", 40., p.ink))
                .child(
                    div()
                        .max_w(px(560.))
                        .text_color(p.muted)
                        .child("Здесь будет идти трансляция прогона: счёт F1 против baseline, тур и эпоха, таблица trials и лог. Запустите подбор для класса или сценарий задач run_all.py."),
                )
                .when_some(last, |d, r| {
                    d.child(div().text_color(p.muted).child(format!("Лучший результат на диске: {} — F1 {}", r.name, fmt(r.best_f1()))))
                })
                .child(
                    h_flex()
                        .gap_2()
                        .child(Button::new("empty-go").icon(Icon::new(Lucide::Play)).label("Новый запуск").primary().on_click(cx.listener(|this, _, _, cx| {
                            this.tab = Tab::Launch;
                            cx.notify();
                        })))
                        .child(Button::new("empty-runs").label("Открыть прогоны").outline().on_click(cx.listener(|this, _, _, cx| {
                            this.tab = Tab::Runs;
                            cx.notify();
                        }))),
                )
                .into_any_element();
        };

        let (label, chip_bg, chip_fg) = j.status(&p);
        let running = j.running();
        if j.predicting {
            return self.render_predicting(p, cx);
        }
        let last = j.epochs.last();
        let score = last.map(|e| e.f1);
        let line = j.baseline.map(|b| b + MARGIN);
        let standings = j.standings();
        let best = standings.first().map(|r| r.1);
        let last_trial = last.map(|e| e.trial);
        let pts: Vec<(u32, f64)> = j
            .epochs
            .iter()
            .filter(|e| Some(e.trial) == last_trial)
            .map(|e| (e.epoch, e.f1))
            .collect();
        let (n, dropped) = {
            let out = j.out.lock().unwrap();
            (out.lines.len(), out.dropped)
        };
        let band = class_color(&j.target);
        let max_secs = standings.iter().map(|r| r.3).fold(0., f64::max);

        // Шапка матча: карточка с оттенком цвета класса.
        let header = class_card(band, p)
            .child(v_flex().min_w_0().child(display(&j.run, 34., p.ink)).child(
                div().text_sm().child(format!(
                    "{} · {}{}",
                    if j.setup {
                        "Окружение Python"
                    } else {
                        class_ru(&j.target)
                    },
                    if j.features.is_empty() {
                        "—"
                    } else {
                        &j.features
                    },
                    if j.run != j.title {
                        format!(" · сценарий {}", j.title)
                    } else {
                        String::new()
                    }
                )),
            ))
            .child(chip(label, chip_bg, chip_fg, running))
            .child(div().flex_1())
            .child(
                h_flex()
                    .gap_2()
                    .child(
                        band_button("copy", Lucide::Copy, "Копировать лог", p.raised, p.ink)
                            .on_click(cx.listener(|this, _, _, cx| {
                                let text =
                                    this.jobs[this.sel_job].out.lock().unwrap().lines.join("\n");
                                cx.write_to_clipboard(ClipboardItem::new_string(text));
                            })),
                    )
                    .when_some(j.log_path.clone(), |d, path| {
                        d.child(
                            band_button("log-file", Lucide::FileText, "Файл лога", p.raised, p.ink)
                                .on_click(move |_, _, _| open_path(&path)),
                        )
                    })
                    .when(running, |d| {
                        // Двойное нажатие: прогон идёт часами, случайный клик не должен его убить.
                        let armed = self.confirm_stop.is_some_and(|(i, t)| {
                            i == self.sel_job && t.elapsed() < Duration::from_secs(5)
                        });
                        let b = if armed {
                            band_button(
                                "stop",
                                Lucide::Square,
                                "Ещё раз — остановить",
                                p.loss,
                                rgb(0xFFFFFF).into(),
                            )
                        } else {
                            band_button("stop", Lucide::Square, "Остановить", p.raised, p.loss)
                        };
                        d.child(b.on_click(cx.listener(move |this, _, _, cx| {
                            if armed {
                                let j = &mut this.jobs[this.sel_job];
                                j.stopped = true;
                                kill(j);
                                this.confirm_stop = None;
                            } else {
                                this.confirm_stop = Some((this.sel_job, Instant::now()));
                            }
                            cx.notify();
                        })))
                    })
                    .when(!running, |d| {
                        d.child(
                            band_button("restart", Lucide::RotateCcw, "Перезапустить", p.ink, p.bg)
                                .on_click(cx.listener(|this, _, _, cx| {
                                    this.restart(this.sel_job);
                                    cx.notify();
                                })),
                        )
                    }),
            );

        // Счёт: F1 против линии baseline + 0.05.
        let diff = score.zip(line).map(|(s, l)| s - l);
        let scoreline = card(p)
            .mx_6()
            .mt_3()
            .px_5()
            .py_4()
            .gap_10()
            .items_end()
            .child(
                v_flex()
                    .child(small_caps(
                        if running {
                            "F1 сейчас"
                        } else {
                            "F1 последней эпохи"
                        },
                        p.muted,
                    ))
                    .child(
                        div()
                            .font_family(DISPLAY)
                            .font_weight(FontWeight::BOLD)
                            .text_size(px(64.))
                            .line_height(px(64.))
                            .font_features(tnum())
                            .text_color(p.ink)
                            .child(fmt(score)),
                    ),
            )
            .child(
                v_flex()
                    .gap_1()
                    .child(stat_line(
                        "Линия вылета",
                        line.map_or("—".into(), |l| format!("{l:.4}")),
                        p.ink,
                        p,
                    ))
                    .child(stat_line(
                        "Отрыв",
                        diff.map_or("—".into(), |d| format!("{d:+.4}")),
                        match diff {
                            Some(d) if d >= 0. => p.win,
                            Some(_) => p.loss,
                            None => p.ink,
                        },
                        p,
                    ))
                    .child(stat_line("Лучший в прогоне", fmt(best), p.ink, p)),
            )
            .child(
                v_flex()
                    .gap_1()
                    .child(stat_line(
                        "Тур (trial)",
                        last.map_or("—".into(), |e| {
                            format!("{} из {}", j.trial_in_run(e.trial), j.n_trials)
                        }),
                        p.ink,
                        p,
                    ))
                    .child(stat_line(
                        "Эпоха",
                        last.map_or("—".into(), |e| format!("{} из {}", e.epoch, j.n_epochs)),
                        p.ink,
                        p,
                    )),
            )
            .child(div().flex_1())
            .child(
                v_flex()
                    .items_end()
                    .gap_1()
                    .child(stat_line("Идёт", hms(j.elapsed()), p.ink, p))
                    .when(running, |d| {
                        d.child(stat_line(
                            "Осталось ≤",
                            j.eta().map_or("—".into(), hms),
                            p.live_ink,
                            p,
                        ))
                    }),
            );

        // Таблица тура: trials по убыванию F1, линия вылета, полосы длительности.
        let mut table = v_flex()
            .id("standings")
            .w(px(500.))
            .h_full()
            .flex_shrink_0()
            .rounded_xl()
            .bg(p.surface)
            .border_1()
            .border_color(p.rule)
            .overflow_y_scrollbar()
            .child(
                div()
                    .px_4()
                    .pt_4()
                    .pb_2()
                    .child(display("Таблица тура", 22., p.ink)),
            )
            .child(trow(
                ["#", "Trial", "F1", "Эп.", "Время"].map(String::from),
                true,
                p,
            ));
        let mut line_drawn = line.is_none();
        for (pos, (trial, f1, ep, secs)) in standings.iter().enumerate() {
            if !line_drawn && Some(*f1) < line {
                table = table.child(relegation(p));
                line_drawn = true;
            }
            table = table.child(
                trow(
                    [
                        format!("{}", pos + 1),
                        format!("{trial}"),
                        format!("{f1:.4}"),
                        format!("{ep}"),
                        hms(Duration::from_secs_f64(*secs)),
                    ],
                    false,
                    p,
                )
                .when(line.is_some_and(|l| *f1 >= l), |d| {
                    d.text_color(p.win).font_weight(FontWeight::SEMIBOLD)
                })
                .when(Some(*trial) == last_trial && running, |d| {
                    d.bg(p.live.opacity(0.18))
                })
                .child(duration_bar(*secs, max_secs, p)),
            );
        }
        if !line_drawn && !standings.is_empty() {
            table = table.child(relegation(p));
        }
        if standings.is_empty() {
            table = table.child(
                div()
                    .px_4()
                    .py_2()
                    .text_color(p.muted)
                    .child("Первая эпоха ещё не закончилась."),
            );
        }

        let chart = v_flex()
            .h(px(220.))
            .flex_shrink_0()
            .p_4()
            .rounded_xl()
            .bg(p.surface)
            .border_1()
            .border_color(p.rule)
            .child(chart_title(
                &format!(
                    "F1 по эпохам · trial {}",
                    last_trial.map_or("—".into(), |t| t.to_string())
                ),
                line,
                p,
            ))
            .child(if pts.is_empty() {
                div()
                    .flex_1()
                    .text_color(p.muted)
                    .child("График появится после первой эпохи.")
                    .into_any_element()
            } else {
                f1_chart(
                    "live-epochs",
                    pts,
                    line,
                    None,
                    if self.dark { p.live } else { p.live_ink },
                )
            });

        let dark = self.dark;
        let log =
            v_flex()
                .flex_1()
                .min_h_0()
                .gap_2()
                .child(
                    h_flex()
                        .gap_2()
                        .child(small_caps("Лента событий", p.muted))
                        .when(dropped > 0, |d| {
                            d.child(div().text_xs().text_color(p.muted).child(format!(
                                "в памяти последние {n} строк — полный лог в файле"
                            )))
                        })
                        .child(div().flex_1())
                        .child(choice("auto", "Автопрокрутка", self.autoscroll).on_click(
                            cx.listener(|this, _, _, cx| {
                                this.autoscroll = !this.autoscroll;
                                cx.notify();
                            }),
                        )),
                )
                .child(
                    div()
                        .flex_1()
                        .min_h_0()
                        .rounded_xl()
                        .bg(p.surface)
                        .border_1()
                        .border_color(p.rule)
                        .p_3()
                        .child(
                            uniform_list(
                                "log",
                                n,
                                cx.processor(move |this, range: std::ops::Range<usize>, _, _| {
                                    let p = pal(dark);
                                    let out = this.jobs[this.sel_job].out.lock().unwrap();
                                    range
                                        .filter_map(|i| out.lines.get(i).map(|l| (i, l)))
                                        .map(|(i, l)| {
                                            let bad = ["Error", "ERROR", "Traceback", "exception"]
                                                .iter()
                                                .any(|k| l.contains(k));
                                            let epoch = l.contains(" F1=");
                                            div()
                                                .id(i)
                                                .h(px(LINE_H))
                                                .whitespace_nowrap()
                                                .text_color(if bad {
                                                    p.loss
                                                } else if l.starts_with("===") || l.starts_with('$')
                                                {
                                                    p.win
                                                } else if epoch {
                                                    p.ink
                                                } else {
                                                    p.muted
                                                })
                                                .when(epoch, |d| d.font_weight(FontWeight::BOLD))
                                                .child(l.clone())
                                        })
                                        .collect()
                                }),
                            )
                            .track_scroll(&self.log_scroll)
                            .size_full()
                            .font_family(MONO)
                            .text_xs(),
                        ),
                );

        v_flex()
            .size_full()
            .child(header)
            .map(|d| {
                // Инференс (j.predicting) и установка окружения уходят своим путём — здесь только обучение.
                if j.setup {
                    d.child(v_flex().flex_1().min_h_0().px_6().pt_4().pb_6().child(log))
                } else {
                    d.child(scoreline).child(
                        h_flex()
                            .flex_1()
                            .min_h_0()
                            .px_6()
                            .pt_3()
                            .pb_6()
                            .gap_3()
                            .child(
                                v_flex()
                                    .flex_1()
                                    .min_w_0()
                                    .h_full()
                                    .gap_3()
                                    .child(chart)
                                    .child(log),
                            )
                            .child(table),
                    )
                }
            })
            .into_any_element()
    }

    // ---------- таблицы прогонов ----------

    // ---------- анализ прогона: гиперпараметры и сцены ----------

    /// Каталог со сценами прогона: data_dir из split.json (мог быть на машине коллеги), иначе одноимённый в проекте,
    /// иначе датасет проекта, где есть первая сцена валидации.
    fn val_scenes(&self, r: &Run) -> Result<Vec<PathBuf>, String> {
        let split = r.split.as_ref().ok_or("нет split.json")?;
        let names: Vec<String> = split["val"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let first = names.first().ok_or("в split.json нет сцен валидации")?;
        let data = split["data_dir"]
            .as_str()
            .map(PathBuf::from)
            .unwrap_or_default();
        // shortcut: из датасетов проекта берётся тот, где эта сцена крупнее всего (исходник, а не прореженная копия);
        // если прореженная копия окажется единственной, метрики будут по ней — каталог показан в карточке.
        let mut local: Vec<(u64, PathBuf)> = self
            .datasets
            .iter()
            .filter_map(|d| {
                Some((
                    std::fs::metadata(d.path.join(first)).ok()?.len(),
                    d.path.clone(),
                ))
            })
            .collect();
        local.sort_by(|a, b| b.0.cmp(&a.0));
        let dir = [
            self.root.join(&data),
            self.root.join(data.file_name().unwrap_or_default()),
        ]
        .into_iter()
        .chain(local.into_iter().map(|d| d.1))
        .find(|d| d.join(first).is_file())
        .ok_or_else(|| {
            format!(
                "сцен валидации ({first} …) нет на этом компьютере — нужен каталог {}",
                show_path(&data)
            )
        })?;
        Ok(names.iter().map(|n| dir.join(n)).collect())
    }

    /// predict_scene.py по всем сценам валидации лучшим trial: предсказания (их же откроет 3D) и scenes__trial<N>.json.
    fn evaluate_scenes(&mut self, idx: usize, cx: &mut Context<Self>) {
        let r = &self.runs[idx];
        let Some(trial) = r.best.as_ref().and_then(|b| b.get("best_trial")?.as_u64()) else {
            return;
        };
        let scenes = match self.val_scenes(r) {
            Ok(s) => s,
            Err(e) => {
                self.tab_error = Some(e);
                return;
            }
        };
        if let Err(e) = self.ensure_predict_script() {
            self.tab_error = Some(e);
            return;
        }
        let checkpoint = r.path.join(format!("trial_{trial:04}")).join("best.pt");
        let mut args: Vec<String> = ["-u", "predict_scene.py", "--checkpoint"]
            .map(String::from)
            .to_vec();
        args.push(checkpoint.to_string_lossy().into_owned());
        args.push("--scene".into());
        args.extend(scenes.iter().map(|s| s.to_string_lossy().into_owned()));
        args.extend([
            "--out-dir".into(),
            r.path.join("pred").to_string_lossy().into_owned(),
            "--tag".into(),
            format!("trial{trial}"),
        ]);
        let (title, python) = (
            format!("Оценка по сценам · {}", r.name),
            self.python.read(cx).value().trim().to_string(),
        );
        let target = self.runs[idx].target().unwrap_or_default();
        let tab = self.tab;
        self.tab_error = None;
        self.spawn(python, args, title);
        // predict_scene.py не принимает --target: класс нужен шапке трансляции (цвет и подпись).
        if let Some(j) = self.jobs.last_mut() {
            j.target = target;
        }
        self.tab = tab; // лог — в «Трансляции», но человека с текущей вкладки не выдёргиваем
    }

    /// Сцена из таблицы оценки — в 3D-превью «Датасетов», сразу с ошибками модели этого прогона.
    fn open_scene_errors(&mut self, scene: PathBuf, run: String, cx: &mut Context<Self>) {
        if let Some(i) = self
            .datasets
            .iter()
            .position(|d| scene.starts_with(&d.path))
        {
            self.sel_dataset = i;
        }
        self.open_preview(scene.clone(), cx);
        if let Some(m) = self.models(&scene).into_iter().find(|m| m.run == run) {
            self.predict(m, cx);
        }
        self.tab = Tab::Datasets;
        cx.notify();
    }

    // ---------- тестирование моделей ----------

    /// Модели для теста: модель каждого прогона (model_file), затем загруженные .pt.
    fn test_models(&self) -> Vec<TestModel> {
        let runs = self.runs.iter().enumerate().filter_map(|(i, r)| {
            let key = format!("run-{}", r.name);
            let trial = r.best.as_ref()?.get("best_trial")?.as_u64()?;
            Some(TestModel {
                tag: format!("{key}-trial{trial}"),
                key,
                label: r.name.clone(),
                file: model_file(r)?,
                run: Some(i),
            })
        });
        let files = self.imported.iter().map(|f| {
            let stem = f
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let key = format!("file-{stem}");
            TestModel {
                tag: key.clone(),
                key,
                label: stem,
                file: f.clone(),
                run: None,
            }
        });
        runs.chain(files).collect()
    }

    /// Выбранная модель (по ключу; иначе первая) и датасет теста (по имени; иначе первый).
    fn test_pick<'a>(&self, models: &'a [TestModel]) -> (Option<&'a TestModel>, Option<&Dataset>) {
        let m = models
            .iter()
            .find(|m| Some(&m.key) == self.test_model.as_ref())
            .or(models.first());
        let d = self
            .datasets
            .iter()
            .find(|d| Some(&d.name) == self.test_dataset.as_ref())
            .or(self.datasets.first());
        (m, d)
    }

    /// Перечитать models/ и результат теста выбранной модели (маленький JSON).
    fn refresh_tests(&mut self) {
        self.imported = imported_models(&self.root);
        let models = self.test_models();
        self.test_eval = match self.test_pick(&models) {
            (Some(m), Some(d)) => read_test(&test_dir(&self.root, &m.tag, &d.name)),
            _ => None,
        };
    }

    /// Скрипт инференса — помощник приложения: нет его или он старее встроенного (не пишет сведения о модели) — записать встроенный.
    fn ensure_predict_script(&self) -> Result<(), String> {
        let script = self.root.join("predict_scene.py");
        if std::fs::read_to_string(&script).is_ok_and(|s| s.contains("model=dict(")) {
            return Ok(());
        }
        std::fs::write(&script, PREDICT_PY)
            .map_err(|e| format!("не удалось записать {}: {e}", show_path(&script)))
    }

    fn import_models(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: true,
            directories: false,
            multiple: true,
            prompt: Some("Загрузить модель (.pt)".into()),
        });
        let root = self.root.clone();
        cx.spawn_in(window, async move |this, cx| {
            let Ok(Ok(Some(paths))) = rx.await else {
                return;
            };
            // Копирование больших .pt — в фоне, окно не замирает.
            let res: Vec<_> = cx
                .background_executor()
                .spawn(async move { paths.iter().map(|src| import_model(&root, src)).collect() })
                .await;
            _ = this.update(cx, |this, cx| {
                let (ok, errs): (Vec<PathBuf>, Vec<String>) = (
                    res.iter().filter_map(|r| r.clone().ok()).collect(),
                    res.into_iter().filter_map(Result::err).collect(),
                );
                if let Some(last) = ok.last() {
                    this.test_model = last
                        .file_stem()
                        .map(|s| format!("file-{}", s.to_string_lossy()));
                }
                this.test_note = Some(if errs.is_empty() {
                    (
                        true,
                        format!("Загружено моделей: {}. Они лежат в {MODELS_DIR}.", ok.len()),
                    )
                } else {
                    (false, format!("Не загружено: {}", errs.join("; ")))
                });
                this.refresh_tests();
                cx.notify();
            });
        })
        .detach();
    }

    fn test_title(m: &TestModel, d: &Dataset) -> String {
        format!("Тест {} · {}", m.key, d.name)
    }

    /// predict_scene.py по сценам датасета: предсказания и scenes__test.json в tests/<модель>__<датасет>/.
    fn run_test(&mut self, cx: &mut Context<Self>) {
        let models = self.test_models();
        let (Some(m), Some(d)) = self.test_pick(&models) else {
            return;
        };
        let n = self
            .test_scenes
            .read(cx)
            .value()
            .trim()
            .parse::<usize>()
            .unwrap_or(0);
        if n == 0 {
            self.test_note = Some((false, "Число сцен — целое больше нуля.".into()));
            return;
        }
        if let Err(e) = self.ensure_predict_script() {
            self.test_note = Some((false, e));
            return;
        }
        let out = test_dir(&self.root, &m.tag, &d.name);
        let mut args: Vec<String> = ["-u", "predict_scene.py", "--checkpoint"]
            .map(String::from)
            .to_vec();
        args.push(m.file.to_string_lossy().into_owned());
        args.push("--scene".into());
        args.extend(spread(d.scenes.len(), n).into_iter().map(|i| {
            d.path
                .join(&d.scenes[i].name)
                .to_string_lossy()
                .into_owned()
        }));
        args.extend([
            "--out-dir".into(),
            out.to_string_lossy().into_owned(),
            "--tag".into(),
            "test".into(),
        ]);
        let (title, python) = (
            Self::test_title(m, d),
            self.python.read(cx).value().trim().to_string(),
        );
        // predict_scene.py не принимает --target, поэтому класс модели ставим уже запущенному процессу:
        // от него зависят цвет шапки и подпись в трансляции.
        let target = m
            .run
            .and_then(|i| self.runs.get(i))
            .and_then(|r| r.target())
            .unwrap_or_default();
        self.test_note = None;
        self.spawn(python, args, title);
        if let Some(j) = self.jobs.last_mut() {
            j.target = target;
        }
        self.tab = Tab::Testing; // остаёмся на вкладке; лог — в «Трансляции»
    }

    /// Сцена из таблицы теста — в 3D-превью «Датасетов» с ошибками модели.
    fn open_test_scene(
        &mut self,
        scene: PathBuf,
        pred: PathBuf,
        target: String,
        label: String,
        cx: &mut Context<Self>,
    ) {
        if let Some(i) = self
            .datasets
            .iter()
            .position(|d| scene.starts_with(&d.path))
        {
            self.sel_dataset = i;
        }
        self.open_preview(scene.clone(), cx);
        let request = Arc::new(());
        if let Some(p) = self.preview.as_mut() {
            p.pending = None;
            p.pred = Some(PredView {
                request: request.clone(),
                label,
                split: None,
                result: None,
            });
        }
        self.load_prediction(
            PredJob {
                scene,
                out: pred,
                target,
                request,
            },
            cx,
        );
        self.tab = Tab::Datasets;
        cx.notify();
    }

    fn render_influence(&self, r: &Run, p: Pal) -> Option<Div> {
        let inf = influence(r);
        let best = r
            .best
            .as_ref()
            .and_then(|b| b.get("best_trial")?.as_u64())
            .map(|t| t.to_string());
        if inf.first()?.points.len() < 4 {
            return None;
        }
        let color = class_color(&r.target().unwrap_or_default());
        let cards = inf.into_iter().map(|f| {
            let spec = &SPACE[f.spec];
            let (lo, hi) = match spec.kind {
                Kind::Choice(o) => (0., (o.len() - 1) as f64),
                Kind::LogFloat(a, b) => (a.log10(), b.log10()),
                Kind::Float(a, b) => (a, b),
                Kind::Int(a, b) => (a as f64, b as f64),
            };
            let (ylo, yhi) = f
                .points
                .iter()
                .fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.1), b.max(p.1)));
            let h = 84.;
            let choice = matches!(spec.kind, Kind::Choice(_));
            let dots = f.points.iter().map(|(x, y, n)| {
                // Поле с отступом 4 %, чтобы крайние точки не ложились на оси.
                let fx = 0.04 + 0.92 * if hi > lo { (x - lo) / (hi - lo) } else { 0.5 };
                let fy = 0.06
                    + 0.88
                        * if yhi > ylo {
                            (y - ylo) / (yhi - ylo)
                        } else {
                            0.5
                        };
                let is_best = best.as_ref() == Some(n);
                let d = if is_best { 9. } else { 6. };
                // У вариантов все trials в одном столбике — раздвинуть по номеру trial, чтобы точки не слипались.
                let jitter = if choice {
                    (n.parse::<u32>().unwrap_or(0) % 5) as f32 * 3. - 6.
                } else {
                    0.
                };
                div()
                    .absolute()
                    .left(relative(fx as f32))
                    .ml(px(jitter - d / 2.))
                    .top(px(((1. - fy) * h) as f32 - d / 2.))
                    .size(px(d))
                    .rounded_full()
                    .bg(if is_best { color } else { p.muted.opacity(0.6) })
            });
            let axis = match spec.kind {
                Kind::Choice(o) => o
                    .iter()
                    .map(|o| match *o {
                        "true" => "да".to_string(),
                        "false" => "нет".to_string(),
                        o => o.to_string(),
                    })
                    .collect::<Vec<_>>(),
                Kind::LogFloat(a, b) => vec![format!("{a:.0e}"), format!("{b:.0e}")],
                _ => vec![format!("{lo}"), format!("{hi}")],
            };
            let score = f.score.map_or("мало данных".into(), |s| format!("{s:.2}"));
            v_flex()
                .flex_1()
                .min_w(px(240.))
                .p_3()
                .gap_2()
                .rounded_lg()
                .border_1()
                .border_color(p.rule)
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .text_xs()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(p.ink)
                                .child(spec.label),
                        )
                        .child(
                            div()
                                .text_xs()
                                .font_features(tnum())
                                .text_color(p.muted)
                                .child(score),
                        ),
                )
                .child(
                    div().h(px(3.)).w_full().rounded_full().bg(p.rule).child(
                        div()
                            .h_full()
                            .rounded_full()
                            .w(relative(f.score.unwrap_or(0.) as f32))
                            .bg(color),
                    ),
                )
                .child(
                    h_flex()
                        .mt_1()
                        .gap_2()
                        .child(
                            v_flex()
                                .h(px(h as f32))
                                .w(px(36.))
                                .flex_shrink_0()
                                .justify_between()
                                .items_end()
                                .text_xs()
                                .font_features(tnum())
                                .text_color(p.muted)
                                .child(format!("{yhi:.3}"))
                                .child(format!("{ylo:.3}")),
                        )
                        .child(
                            div()
                                .relative()
                                .flex_1()
                                .h(px(h as f32))
                                .border_b_1()
                                .border_l_1()
                                .border_color(p.rule)
                                .children(dots),
                        ),
                )
                .child(
                    h_flex()
                        .pl(px(44.))
                        .justify_between()
                        .text_xs()
                        .text_color(p.muted)
                        .children(axis),
                )
        });
        Some(
            v_flex()
                .mx_6()
                .mt_3()
                .p_4()
                .gap_3()
                .rounded_xl()
                .bg(p.surface)
                .border_1()
                .border_color(p.rule)
                .child(small_caps("Влияние гиперпараметров на F1", p.muted))
                .child(div().text_xs().text_color(p.muted).child(
                    "Точка — завершённый trial: по горизонтали значение, по вертикали F1 (слева — его пределы в этом прогоне); крупная — лучший trial. Число и полоса — сила связи с F1 (0…1): ранговая корреляция Спирмена, для вариантов — η. По 20–30 trials это ориентир, а не доказательство: Optuna чаще пробует удачные области.",
                ))
                .child(h_flex().gap_3().flex_wrap().children(cards)),
        )
    }

    fn render_scene_eval(&self, idx: usize, p: Pal, cx: &mut Context<Self>) -> Div {
        let r = &self.runs[idx];
        let title = format!("Оценка по сценам · {}", r.name);
        let running = self.jobs.iter().any(|j| j.running() && j.title == title);
        let trial = r.best.as_ref().and_then(|b| b.get("best_trial")?.as_u64());
        let has_ckpt = trial.is_some_and(|t| {
            r.path
                .join(format!("trial_{t:04}"))
                .join("best.pt")
                .exists()
        });
        let eval = r.scene_eval.as_ref();
        let (done, total) = eval.map_or((0, 0), |e| {
            (
                e["done"].as_u64().unwrap_or(0),
                e["total"].as_u64().unwrap_or(0),
            )
        });
        let scenes = self.val_scenes(r);
        let from = match &scenes {
            Ok(s) => s
                .first()
                .and_then(|f| f.parent())
                .map_or(String::new(), |d| {
                    format!(
                        " Сцены берутся из {}.",
                        show_path(d.strip_prefix(&self.root).unwrap_or(d))
                    )
                }),
            Err(e) => format!(" {e}."),
        };
        let button = (has_ckpt && scenes.is_ok() && !running && done < total.max(1)).then(|| {
            Button::new("run-eval")
                .icon(Icon::new(Lucide::ScanSearch))
                .label(if eval.is_some() {
                    "Досчитать"
                } else {
                    "Оценить по сценам"
                })
                .outline()
                .small()
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.evaluate_scenes(idx, cx);
                    cx.notify();
                }))
        });
        let note = if !has_ckpt {
            "Нет best.pt лучшего trial — оценивать нечем.".to_string()
        } else if running {
            format!(
                "Идёт оценка: {done} из {} сцен, лог — в «Трансляции». Таблица пополняется по ходу.",
                total.max(done)
            )
        } else if eval.is_none() {
            format!(
                "Лучший trial прогоняется по каждой сцене валидации (GPU, минуты). Худшие сцены окажутся сверху, клик откроет сцену в 3D с ошибками модели.{from}"
            )
        } else if done < total {
            format!(
                "Посчитано {done} из {total} сцен — оценку прервали; «Досчитать» продолжит с места остановки."
            )
        } else {
            format!("{total} сцен валидации, худшие сверху. Клик — сцена в 3D с ошибками модели.")
        };
        let run = r.name.clone();
        let table = self.scene_table(
            eval,
            Some(r),
            p,
            cx,
            Rc::new(move |this, path, cx| this.open_scene_errors(path, run.clone(), cx)),
        );
        v_flex()
            .mx_6()
            .mt_3()
            .p_4()
            .gap_2()
            .rounded_xl()
            .bg(p.surface)
            .border_1()
            .border_color(p.rule)
            .child(
                h_flex()
                    .gap_3()
                    .child(small_caps(
                        "Где модель ошибается · сцены валидации",
                        p.muted,
                    ))
                    .child(div().flex_1())
                    .children(button),
            )
            .child(div().text_xs().text_color(p.muted).child(note))
            .children(table)
    }

    /// Таблица оценки по сценам (scenes__*.json от predict_scene.py --out-dir), худшие сверху.
    /// Строка красная, если F1 ниже baseline этой сцены + MARGIN: там модель не лучше «всё = 1». Клик — `open`.
    fn scene_table(
        &self,
        eval: Option<&serde_json::Value>,
        run: Option<&Run>,
        p: Pal,
        cx: &mut Context<Self>,
        open: OpenScene,
    ) -> Option<Div> {
        let mut rows: Vec<serde_json::Value> = eval?["scenes"]
            .as_array()
            .filter(|r| !r.is_empty())?
            .clone();
        rows.sort_by(|a, b| {
            a["f1"]
                .as_f64()
                .unwrap_or(0.)
                .total_cmp(&b["f1"].as_f64().unwrap_or(0.))
        });
        let mut table = v_flex().child(
            trow(
                [
                    "#",
                    "Сцена",
                    "Выборка",
                    "F1",
                    "Precision",
                    "Recall",
                    "Класс",
                    "Не покрыто",
                ]
                .map(String::from),
                true,
                p,
            )
            .px_0(),
        );
        for (i, s) in rows.iter().enumerate() {
            let f1 = s["f1"].as_f64();
            let line = s["baseline_f1"].as_f64().map(|b| b + MARGIN);
            let share = s["val_positive_points"].as_f64().unwrap_or(0.)
                / s["points"].as_f64().unwrap_or(0.).max(1.);
            let name = s["scene"].as_str().unwrap_or("?").to_string();
            let path = PathBuf::from(s["path"].as_str().unwrap_or_default());
            let open = open.clone();
            table = table.child(
                trow(
                    [
                        format!("{}", i + 1),
                        name.clone(),
                        scene_split(run, &name),
                        fmt(f1),
                        fmt(s["precision"].as_f64()),
                        fmt(s["recall"].as_f64()),
                        format!("{:.0} %", share * 100.),
                        thousands(s["not_covered"].as_u64().unwrap_or(0)),
                    ],
                    false,
                    p,
                )
                .px_0()
                .cursor_pointer()
                .hover(|d| d.bg(p.raised))
                .when(f1.zip(line).is_some_and(|(f, l)| f < l), |d| {
                    d.text_color(p.loss)
                })
                .on_click(cx.listener(move |this, _, _, cx| open(this, path.clone(), cx))),
            );
        }
        Some(table)
    }

    // ---------- сравнение прогонов ----------

    /// Читает F1 по эпохам лучшего trial каждого отмеченного прогона (несколько маленьких CSV).
    fn open_compare(&mut self) {
        self.compare
            .retain(|n| self.runs.iter().any(|r| r.name == *n));
        if self.compare.len() < 2 {
            self.compare_view = None;
            return;
        }
        let view = self
            .compare
            .iter()
            .filter_map(|n| self.runs.iter().find(|r| r.name == *n))
            .map(|r| {
                let best = r.best.as_ref().and_then(|b| b.get("best_trial")?.as_u64());
                (
                    r.name.clone(),
                    best.map_or_else(Vec::new, |t| read_epochs(&r.path, &t.to_string())),
                )
            })
            .collect();
        self.compare_view = Some(view);
    }

    fn render_compare(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let view = self.compare_view.clone().unwrap_or_default();
        let runs: Vec<&Run> = view
            .iter()
            .filter_map(|(n, _)| self.runs.iter().find(|r| r.name == *n))
            .collect();
        let best = |r: &Run, k: &str| {
            r.best
                .as_ref()
                .and_then(|b| b.get(k))
                .cloned()
                .unwrap_or_default()
        };
        let margin = |r: &Run| r.best_f1().zip(r.baseline()).map(|(f, b)| f - b - MARGIN);
        const LABEL_W: f32 = 200.;
        const COL_W: f32 = 180.;

        let mut table = v_flex().min_w_0();
        // Заголовки колонок: цвет класса и имя прогона.
        table = table.child(
            h_flex()
                .px_8()
                .py_2()
                .border_b_1()
                .border_color(p.rule)
                .child(div().w(px(LABEL_W)).flex_shrink_0())
                .children(runs.iter().map(|r| {
                    let t = r.target().unwrap_or_default();
                    h_flex()
                        .w(px(COL_W))
                        .flex_shrink_0()
                        .gap_2()
                        .child(
                            div()
                                .size(px(10.))
                                .flex_shrink_0()
                                .rounded_full()
                                .bg(class_color(&t)),
                        )
                        .child(
                            div()
                                .min_w_0()
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_ellipsis()
                                .font_weight(FontWeight::SEMIBOLD)
                                .child(r.name.clone()),
                        )
                })),
        );
        let line = |label: &str, cells: Vec<(String, Hsla, bool)>| {
            h_flex()
                .px_8()
                .py_1p5()
                .border_b_1()
                .border_color(p.rule)
                .font_features(tnum())
                .child(
                    div()
                        .w(px(LABEL_W))
                        .flex_shrink_0()
                        .text_xs()
                        .text_color(p.muted)
                        .child(label.to_string()),
                )
                .children(cells.into_iter().map(|(v, color, bold)| {
                    div()
                        .w(px(COL_W))
                        .flex_shrink_0()
                        .overflow_hidden()
                        .whitespace_nowrap()
                        .text_ellipsis()
                        .text_color(color)
                        .when(bold, |d| d.font_weight(FontWeight::BOLD))
                        .child(v)
                }))
        };
        // Метрика: лучшее значение среди прогонов выделено.
        let metric = |label: &str, vals: Vec<Option<f64>>| {
            let top = vals
                .iter()
                .flatten()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let many = vals.iter().flatten().count() > 1;
            line(
                label,
                vals.iter()
                    .map(|v| {
                        (
                            fmt(*v),
                            if many && *v == Some(top) {
                                p.win
                            } else {
                                p.ink
                            },
                            many && *v == Some(top),
                        )
                    })
                    .collect(),
            )
        };
        // Текст: строки, где прогоны различаются, — ярче.
        let text = |label: &str, vals: Vec<String>| {
            let differ = vals.windows(2).any(|w| w[0] != w[1]);
            line(
                label,
                vals.into_iter()
                    .map(|v| (v, if differ { p.ink } else { p.muted }, differ))
                    .collect(),
            )
        };
        let section = |title: &str| div().px_8().pt_5().pb_1().child(small_caps(title, p.muted));

        table = table
            .child(section("Результат лучшего trial"))
            .child(metric("F1", runs.iter().map(|r| r.best_f1()).collect()))
            .child(metric(
                "Precision",
                runs.iter().map(|r| best(r, "precision").as_f64()).collect(),
            ))
            .child(metric(
                "Recall",
                runs.iter().map(|r| best(r, "recall").as_f64()).collect(),
            ))
            .child(line(
                "Baseline F1",
                runs.iter()
                    .map(|r| (fmt(r.baseline()), p.muted, false))
                    .collect(),
            ))
            .child(line(
                "Запас над линией вылета",
                runs.iter()
                    .map(|r| match margin(r) {
                        Some(m) => (
                            format!("{m:+.4}"),
                            if m >= 0. { p.win } else { p.loss },
                            true,
                        ),
                        None => ("—".into(), p.muted, false),
                    })
                    .collect(),
            ))
            .child(text(
                "Лучший trial",
                runs.iter()
                    .map(|r| best(r, "best_trial").to_string().replace("null", "—"))
                    .collect(),
            ))
            .child(text(
                "Trials завершено",
                runs.iter()
                    .map(|r| {
                        format!(
                            "{} / {}",
                            r.trials.iter().filter(|t| t.state == "COMPLETE").count(),
                            r.trials.len()
                        )
                    })
                    .collect(),
            ))
            .child(section("Постановка"))
            .child(text(
                "Класс",
                runs.iter()
                    .map(|r| class_ru(&r.target().unwrap_or_default()).to_string())
                    .collect(),
            ))
            .child(text(
                "Признаки",
                runs.iter()
                    .map(|r| r.features().unwrap_or_else(|| "—".into()))
                    .collect(),
            ))
            .child(text("Loss", runs.iter().map(|r| r.loss()).collect()));
        let show = |v: &serde_json::Value| match v {
            serde_json::Value::Null => "—".to_string(),
            serde_json::Value::String(s) if s.is_empty() => "—".into(),
            serde_json::Value::String(s) => Path::new(s)
                .file_name()
                .map_or(s.clone(), |n| n.to_string_lossy().into_owned()),
            serde_json::Value::Number(n) if n.is_f64() => {
                let f = n.as_f64().unwrap_or_default();
                if f != 0. && f.abs() < 0.01 {
                    format!("{f:.2e}")
                } else {
                    format!("{f:.4}")
                        .trim_end_matches('0')
                        .trim_end_matches('.')
                        .to_string()
                }
            }
            v => v.to_string(),
        };
        for (k, label) in [
            ("epochs", "Эпох"),
            ("patch_points", "Точек в патче"),
            ("patches_per_scene", "Патчей на сцену"),
            ("seed", "Seed"),
            ("data_dir", "Данные"),
            ("device", "Устройство"),
        ] {
            table = table.child(text(
                label,
                runs.iter().map(|r| show(&best(r, k))).collect(),
            ));
        }
        // Гиперпараметры: сначала в порядке SPACE, затем всё прочее из params.
        let mut keys: Vec<(String, String)> = SPACE
            .iter()
            .map(|s| (s.key.to_string(), s.label.to_string()))
            .collect();
        for r in &runs {
            if let Some(o) = best(r, "params").as_object() {
                for k in o.keys() {
                    if !keys.iter().any(|(kk, _)| kk == k) {
                        keys.push((k.clone(), k.clone()));
                    }
                }
            }
        }
        table = table.child(section("Гиперпараметры лучшего trial"));
        for (k, label) in keys {
            table = table.child(text(
                &label,
                runs.iter()
                    .map(|r| show(&best(r, "params")[k.as_str()]))
                    .collect(),
            ));
        }

        // Графики в одном масштабе: общая нижняя граница по всем точкам и линиям.
        let floor = view
            .iter()
            .flat_map(|(_, e)| e.iter().map(|x| x.1))
            .chain(runs.iter().filter_map(|r| r.baseline().map(|b| b + MARGIN)))
            .fold(1.0f64, f64::min);
        let charts = h_flex().px_8().pt_5().gap_4().flex_wrap().children(
            view.iter()
                .zip(&runs)
                .enumerate()
                .map(|(i, ((_, pts), r))| {
                    let line = r.baseline().map(|b| b + MARGIN);
                    v_flex()
                        .w(px(320.))
                        .gap_2()
                        .child(chart_title(
                            &format!("{} · trial {}", r.name, best(r, "best_trial")),
                            None,
                            p,
                        ))
                        .child(if pts.is_empty() {
                            div()
                                .h(px(160.))
                                .text_color(p.muted)
                                .child("Нет metrics.csv лучшего trial")
                                .into_any_element()
                        } else {
                            div()
                                .h(px(160.))
                                .child(f1_chart(
                                    ("cmp-chart", i),
                                    pts.clone(),
                                    line,
                                    Some(floor),
                                    class_color(&r.target().unwrap_or_default()),
                                ))
                                .into_any_element()
                        })
                }),
        );

        let head = h_flex()
            .px_8()
            .pt_6()
            .pb_4()
            .gap_4()
            .border_b_1()
            .border_color(p.rule)
            .child(
                v_flex()
                    .flex_1()
                    .child(display(&format!("Сравнение · {}", runs.len()), 34., p.ink))
                    .child(div().text_color(p.muted).child("Лучший trial каждого прогона. Зелёным — лучшее значение, жирным — строки, где прогоны различаются.")),
            )
            .child(Button::new("cmp-close").icon(Icon::new(Lucide::X)).label("Закрыть").outline().on_click(cx.listener(|this, _, _, cx| {
                this.compare_view = None;
                cx.notify();
            })));
        v_flex()
            .flex_1()
            .min_w_0()
            .h_full()
            .child(head)
            .child(
                v_flex()
                    .id("cmp-body")
                    .flex_1()
                    .overflow_scroll()
                    .pb_8()
                    .child(table)
                    .child(section(
                        "F1 по эпохам · общий масштаб, пунктир — линия вылета",
                    ))
                    .child(charts),
            )
            .into_any_element()
    }

    fn render_runs(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let live: Vec<String> = self
            .jobs
            .iter()
            .filter(|j| j.running())
            .map(|j| j.run.clone())
            .collect();
        let list = v_flex()
            .id("run-list")
            .w(px(300.))
            .h_full()
            .flex_shrink_0()
            .border_r_1()
            .border_color(p.rule)
            .overflow_y_scrollbar()
            .child(
                h_flex()
                    .px_4()
                    .pt_4()
                    .pb_2()
                    .child(display("Прогоны", 22., p.ink))
                    .child(div().flex_1())
                    .child(
                        Button::new("runs-reload")
                            .icon(Icon::new(Lucide::RefreshCw))
                            .ghost()
                            .xsmall()
                            .tooltip("Перечитать каталог результатов из формы «Запуск»")
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.reload_runs(cx);
                                cx.notify();
                            })),
                    ),
            )
            .child(div().px_4().pb_2().child(if self.compare.len() >= 2 {
                Button::new("runs-compare")
                    .icon(Icon::new(Lucide::Columns3))
                    .label(format!("Сравнить · {}", self.compare.len()))
                    .primary()
                    .small()
                    .w_full()
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.open_compare();
                        cx.notify();
                    }))
                    .into_any_element()
            } else {
                div()
                    .text_xs()
                    .text_color(p.muted)
                    .child("Отметьте два прогона и больше, чтобы сравнить их.")
                    .into_any_element()
            }))
            .children(self.runs.iter().enumerate().map(|(i, r)| {
                let target = &r.target().unwrap_or_default();
                let done = r.trials.iter().filter(|t| t.state == "COMPLETE").count();
                let is_live = live.contains(&r.name);
                h_flex()
                    .id(("run", i))
                    .mx_2()
                    .px_3()
                    .py_2()
                    .gap_3()
                    .rounded_lg()
                    .cursor_pointer()
                    .hover(|s| s.bg(p.raised))
                    .when(i == self.sel_run && self.compare_view.is_none(), |d| {
                        d.bg(p.raised)
                    })
                    .child({
                        let name = r.name.clone();
                        Checkbox::new(("cmp", i))
                            .checked(self.compare.contains(&r.name))
                            .tooltip("Добавить к сравнению")
                            .on_click(cx.listener(move |this, checked: &bool, _, cx| {
                                cx.stop_propagation();
                                this.compare.retain(|n| *n != name);
                                if *checked {
                                    this.compare.push(name.clone());
                                }
                                if this.compare_view.is_some() {
                                    this.open_compare();
                                }
                                cx.notify();
                            }))
                    })
                    .child(
                        div()
                            .size(px(10.))
                            .flex_shrink_0()
                            .rounded_full()
                            .bg(class_color(target)),
                    )
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .child(
                                div()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(p.ink)
                                    .child(r.name.clone()),
                            )
                            .child(div().text_xs().text_color(p.muted).child(format!(
                                "{} · trials {done}/{}",
                                class_ru(target),
                                r.trials.len()
                            ))),
                    )
                    .child(
                        v_flex()
                            .items_end()
                            .child(
                                div()
                                    .font_features(tnum())
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(p.ink)
                                    .child(fmt(r.best_f1())),
                            )
                            .when(is_live, |d| {
                                d.child(div().text_xs().text_color(p.live_ink).child("в эфире"))
                            }),
                    )
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.sel_run = i;
                        this.compare_view = None;
                        this.delete_error = None;
                        this.tab_error = None;
                        this.select_best_trial();
                        cx.notify();
                    }))
            }))
            .when(self.runs.is_empty(), |d| {
                d.child(div().px_4().py_2().text_color(p.muted).child(format!(
                    "В {} пока пусто. Результаты появятся после первого trial.",
                    self.root
                        .join(self.runs_dir.read(cx).value().trim())
                        .display()
                )))
            });

        if self.compare_view.is_some() {
            let body = self.render_compare(p, cx);
            return h_flex()
                .size_full()
                .child(list)
                .child(body)
                .into_any_element();
        }
        let Some(r) = self.runs.get(self.sel_run) else {
            return h_flex().size_full().child(list).into_any_element();
        };
        let b = r.best.as_ref();
        let get = |k: &str| b.and_then(|b| b.get(k)).cloned().unwrap_or_default();
        let target = r.target().unwrap_or_default();
        let features = r.features().unwrap_or_else(|| "—".into());
        let baseline = r.baseline();
        // Класс и признаки известны (split.json/best.json) — только тогда можно продолжить или взять параметры.
        let known = r.target().is_some_and(|t| TARGETS.contains(&t.as_str()))
            && r.features().is_some_and(|f| FEATURES.contains(&f.as_str()));
        let line = baseline.map(|b| b + MARGIN);
        let f1 = r.best_f1();
        let diff = f1.zip(line).map(|(f, l)| f - l);
        let params = get("params");
        let path = r.path.clone();

        let mut trials: Vec<&Trial> = r.trials.iter().collect();
        trials.sort_by(|a, b| b.f1.unwrap_or(-1.).total_cmp(&a.f1.unwrap_or(-1.)));
        let max_secs = trials.iter().filter_map(|t| t.seconds).fold(0., f64::max);
        let sel = self.sel_trial.clone();

        let header = class_card(class_color(&target), p)
            .flex_wrap()
            .child(
                v_flex()
                    .child(display(&r.name, 34., p.ink))
                    .child(div().text_sm().child(format!(
                        "{} · {features} · {}",
                        class_ru(&target),
                        r.loss()
                    ))),
            )
            .child(div().flex_1())
            .child(
                icon_button("run-open", Lucide::FolderOpen, "Открыть папку прогона", p)
                    .on_click(move |_, _, _| open_path(&path)),
            )
            .when_some(model_file(r), |d, src| {
                let name = r.name.clone();
                d.child(
                    icon_button(
                        "run-save-model",
                        Lucide::ArrowDownToLine,
                        "Сохранить модель…",
                        p,
                    )
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.save_model(src.clone(), &name, window, cx)
                    })),
                )
            })
            .when_some(artifact_log_path(&self.root, &r.name), |d, log| {
                d.child(
                    icon_button("run-log", Lucide::FileText, "Открыть последний лог", p)
                        .on_click(move |_, _, _| open_path(&log)),
                )
            })
            .child({
                // Двойное нажатие, как у «Остановить»: удаление прогона — часы обучения.
                let armed = self
                    .confirm_delete
                    .as_ref()
                    .is_some_and(|(n, t)| *n == r.name && t.elapsed() < Duration::from_secs(5));
                let b = if armed {
                    band_button(
                        "run-delete",
                        Lucide::Trash,
                        "Удалить?",
                        p.loss,
                        rgb(0xFFFFFF).into(),
                    )
                } else {
                    icon_button(
                        "run-delete",
                        Lucide::Trash,
                        "Удалить в корзину (нажать дважды)",
                        p,
                    )
                };
                let name = r.name.clone();
                b.on_click(cx.listener(move |this, _, _, cx| {
                    if armed {
                        this.delete_run(this.sel_run, cx);
                    } else {
                        this.delete_error = None;
                        this.tab_error = None;
                        this.confirm_delete = Some((name.clone(), Instant::now()));
                        cx.notify();
                    }
                }))
            })
            .when(
                known
                    && r.best
                        .as_ref()
                        .is_some_and(|b| b.get("params").is_some_and(|p| p.is_object())),
                |d| {
                    d.child(
                        band_button("run-final", Lucide::Play, "Финальный прогон", p.ink, p.bg)
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.start_final_run(this.sel_run, cx);
                            })),
                    )
                },
            )
            .when(known, |d| {
                d.child(
                    band_button("run-form", Lucide::Play, "Продолжить в форме", p.ink, p.bg)
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.form_from_run(this.sel_run, window, cx);
                            this.custom = false; // продолжение study — снова подбор Optuna
                            let name = this.runs[this.sel_run].name.clone();
                            Self::set_input(&this.out_name, name, window, cx);
                            this.tab = Tab::Launch;
                            cx.notify();
                        })),
                )
            })
            .when(known, |d| {
                d.child(
                    band_button(
                        "run-edit",
                        Lucide::SlidersHorizontal,
                        "Параметры в редактор",
                        p.ink,
                        p.bg,
                    )
                    .on_click(cx.listener(move |this, _, window, cx| {
                        // Выбранный trial, иначе лучший.
                        let r = &this.runs[this.sel_run];
                        let from_trial = this.sel_trial.as_ref().and_then(|n| {
                            let t = r.trials.iter().find(|t| &t.number == n)?;
                            Some((
                                format!("trial {n} прогона {} (F1 {})", r.name, fmt(t.f1)),
                                serde_json::from_str::<Params>(&t.params).ok()?,
                            ))
                        });
                        let from_best = || {
                            Some((
                                format!("лучшего trial прогона {}", r.name),
                                r.best.as_ref()?.get("params")?.as_object()?.clone(),
                            ))
                        };
                        let Some((src, params)) = from_trial.or_else(from_best) else {
                            return;
                        };
                        this.form_from_run(this.sel_run, window, cx);
                        this.custom = true;
                        this.apply_params(&params, window, cx);
                        this.params_note =
                            Some(format!("Параметры {src}. Поменяйте нужное и запустите."));
                        Self::set_input(&this.out_name, String::new(), window, cx);
                        this.tab = Tab::Launch;
                        cx.notify();
                    })),
                )
            });

        let scoreline = card(p)
            .mx_6()
            .mt_3()
            .px_5()
            .py_4()
            .flex_wrap()
            .gap_x_10()
            .gap_y_3()
            .items_end()
            .child(
                v_flex()
                    .child(small_caps("Лучший F1", p.muted))
                    .child(div().font_family(DISPLAY).font_weight(FontWeight::BOLD).text_size(px(56.)).line_height(px(56.)).font_features(tnum()).text_color(p.ink).child(fmt(f1))),
            )
            .child(
                v_flex()
                    .gap_1()
                    .child(stat_line("Precision", fmt(get("precision").as_f64()), p.ink, p))
                    .child(stat_line("Recall", fmt(get("recall").as_f64()), p.ink, p))
                    .child(stat_line("Лучший trial", get("best_trial").as_u64().map_or("—".into(), |n| n.to_string()), p.ink, p)),
            )
            .child(
                v_flex()
                    .gap_1()
                    .child(stat_line("Baseline F1", fmt(baseline), p.ink, p))
                    .child(stat_line("Линия вылета", fmt(line), p.ink, p))
                    .child(stat_line(
                        "Отрыв",
                        diff.map_or("—".into(), |d| format!("{d:+.4}")),
                        match diff {
                            Some(d) if d >= 0. => p.win,
                            Some(_) => p.loss,
                            None => p.ink,
                        },
                        p,
                    )),
            )
            .when(diff.is_some_and(|d| d < 0.), |d| {
                // flex_1 + min_w: текст занимает остаток строки и переносится, а в узком окне уходит вниз.
                d.child(div().flex_1().min_w(px(200.)).max_w(px(320.)).text_sm().text_color(p.loss).child("Модель обгоняет «всё = 1» меньше чем на 0.05 — результат под вопросом."))
            });

        // Train/Val и матрицы ошибок лучшего trial — есть только в best.json новых прогонов.
        let (train_m, val_m) = (get("train"), get("val"));
        let splits = (train_m.is_object() && val_m.is_object()).then(|| {
            let color = class_color(&target);
            let cell = |t: String, w: f32| div().w(px(w)).flex().justify_end().child(t);
            let row = |name: &str, m: &serde_json::Value| {
                h_flex().py_1().border_b_1().border_color(p.rule).font_features(tnum()).child(div().w(px(56.)).child(name.to_string()))
                    .children(["iou", "f1", "precision", "recall"].map(|k| cell(fmt(m[k].as_f64()), 76.)))
            };
            card(p)
                .mx_6()
                .mt_3()
                .px_5()
                .py_4()
                .gap_10()
                .flex_wrap()
                .items_start()
                .child(
                    v_flex()
                        .gap_1()
                        .child(small_caps(&format!("Лучший trial · эпоха {}", get("epoch").as_u64().map_or("—".into(), |e| e.to_string())), p.muted))
                        .child(
                            h_flex().py_1().border_b_1().border_color(p.rule).text_xs().font_weight(FontWeight::SEMIBOLD).text_color(p.muted).child(div().w(px(56.)))
                                .children(["IoU", "F1", "Precision", "Recall"].map(|k| cell(k.to_uppercase(), 76.))),
                        )
                        .child(row("Train", &train_m))
                        .child(row("Val", &val_m))
                        .child(div().pt_1().max_w(px(360.)).text_xs().text_color(p.muted).child("Train — по шагам обучения эпохи (режим train), Val — отдельным проходом.")),
                )
                .child(confusion_matrix("Train", &train_m, color, p))
                .child(confusion_matrix("Val", &val_m, color, p))
        });

        // ---- кто и как запускал ----
        let launches = r
            .studio
            .as_ref()
            .and_then(|s| s["launches"].as_array().cloned())
            .unwrap_or_default();
        let last = launches.last().cloned();
        let s = |v: &serde_json::Value| v.as_str().unwrap_or("?").to_string();
        let versions = get("versions");
        let env_line = [get("device").as_str().map(String::from)]
            .into_iter()
            .chain(
                ["python", "torch", "optuna"]
                    .map(|k| versions[k].as_str().map(|v| format!("{k} {v}"))),
            )
            .flatten()
            .collect::<Vec<_>>()
            .join(" · ");
        let data_dir = get("data_dir").as_str().map(String::from).or_else(|| {
            last.as_ref().and_then(|l| {
                Job::arg(
                    &serde_json::from_value::<Vec<String>>(l["args"].clone()).ok()?,
                    "--data-dir",
                )
            })
        });
        let data_missing = data_dir
            .as_ref()
            .is_some_and(|d| !self.root.join(d).exists());
        let script = script_changed(&self.root, &r.path);
        let snap = r.path.join(SNAPSHOT);
        let provenance = v_flex()
            .mx_6()
            .mt_3()
            .p_4()
            .gap_1p5()
            .rounded_xl()
            .bg(p.surface)
            .border_1()
            .border_color(p.rule)
            .child(
                h_flex()
                    .gap_3()
                    .child(small_caps("Кто и как запускал", p.muted))
                    .child(div().flex_1())
                    .when(last.is_some() && known, |d| {
                        d.child(
                            band_button(
                                "run-repeat",
                                Lucide::Repeat,
                                "Повторить точно",
                                p.ink,
                                p.bg,
                            )
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.repeat_run(this.sel_run, cx);
                                cx.notify();
                            })),
                        )
                    }),
            )
            .child(stat_line(
                "Запустил",
                match &last {
                    Some(l) => format!(
                        "{} @ {} · {}{}",
                        s(&l["author"]),
                        s(&l["host"]),
                        s(&l["started"]),
                        if launches.len() > 1 {
                            format!(" · запусков {}", launches.len())
                        } else {
                            String::new()
                        }
                    ),
                    None => "неизвестно — прогон запущен не из ML Studio".into(),
                },
                if last.is_some() { p.ink } else { p.muted },
                p,
            ))
            .child(stat_line(
                "Окружение",
                if env_line.is_empty() {
                    "—".into()
                } else {
                    env_line
                },
                p.ink,
                p,
            ))
            .child(stat_line(
                "Seed",
                get("seed").as_u64().map_or("—".into(), |v| v.to_string()),
                p.ink,
                p,
            ))
            .child(stat_line(
                "Данные",
                data_dir.map_or("—".into(), |d| {
                    if data_missing {
                        format!("{d} · нет на этом компьютере")
                    } else {
                        d
                    }
                }),
                if data_missing { p.loss } else { p.ink },
                p,
            ))
            .child(
                h_flex()
                    .gap_3()
                    .child(
                        stat_line(
                            "Скрипт",
                            match script {
                                Some(false) => {
                                    "Код движка (hpo_pointnet.py, hpo_core/) не менялся с первого запуска".into()
                                }
                                Some(true) => {
                                    "Код движка (hpo_pointnet.py, hpo_core/) изменён после запуска — повтор не будет точным"
                                        .into()
                                }
                                None => "копии нет — прогон запущен не из ML Studio".into(),
                            },
                            match script {
                                Some(false) => p.win,
                                Some(true) => p.loss,
                                None => p.muted,
                            },
                            p,
                        )
                        .flex_1()
                        .min_w_0(),
                    )
                    .when(script.is_some(), |d| {
                        d.child(
                            icon_button("run-snap", Lucide::FileCode, "Открыть копию скрипта", p)
                                .on_click(move |_, _, _| open_path(&snap)),
                        )
                    }),
            )
            .when_some(last.as_ref(), |d, l| {
                let args = l["args"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str())
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_default();
                d.child(
                    div()
                        .pt_1()
                        .font_family(MONO)
                        .text_xs()
                        .text_color(p.muted)
                        .child(format!("$ {} {args}", s(&l["program"]))),
                )
            })
            .child(div().pt_2().child(Input::new(&self.note)));

        let mut detail = v_flex()
            .id("run-detail")
            .flex_1()
            .min_w_0()
            .h_full()
            .overflow_y_scrollbar()
            .child(header)
            .child(scoreline)
            .children(splits)
            .child(provenance)
            .children(self.render_influence(r, p))
            .child(self.render_scene_eval(self.sel_run, p, cx))
            .when_some(self.delete_error.clone(), |d, e| {
                d.child(
                    div()
                        .px_6()
                        .py_2()
                        .border_b_1()
                        .border_color(p.rule)
                        .text_sm()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(p.loss)
                        .child(e),
                )
            })
            .when(!params.is_null(), |d| {
                d.child(
                    div()
                        .px_6()
                        .pt_3()
                        .font_family(MONO)
                        .text_xs()
                        .text_color(p.muted)
                        .child(format!("params {params}")),
                )
            });

        if !self.trial_epochs.is_empty() {
            detail = detail.child(
                v_flex()
                    .h(px(220.))
                    .flex_shrink_0()
                    .mx_6()
                    .mt_3()
                    .p_4()
                    .rounded_xl()
                    .bg(p.surface)
                    .border_1()
                    .border_color(p.rule)
                    .child(chart_title(
                        &format!("F1 по эпохам · trial {}", sel.clone().unwrap_or_default()),
                        line,
                        p,
                    ))
                    .child(f1_chart(
                        "trial-chart",
                        self.trial_epochs.clone(),
                        line,
                        None,
                        class_color(&target),
                    )),
            );
        }

        detail = detail
            .child(
                div()
                    .px_6()
                    .pt_4()
                    .pb_2()
                    .child(display("Таблица trials", 22., p.ink)),
            )
            .child(trow(
                ["#", "Trial", "F1", "Precision", "Recall", "Время", "Итог"].map(String::from),
                true,
                p,
            ));
        let mut line_drawn = line.is_none();
        for (pos, tr) in trials.iter().enumerate() {
            if !line_drawn && tr.f1.is_some() && tr.f1 < line {
                detail = detail.child(relegation(p));
                line_drawn = true;
            }
            let n = tr.number.clone();
            let state = match tr.state.as_str() {
                "COMPLETE" => "сыгран",
                "PRUNED" => "снят",
                "FAIL" => "ошибка",
                "RUNNING" => "идёт",
                s => s,
            };
            detail = detail.child(
                trow(
                    [
                        format!("{}", pos + 1),
                        tr.number.clone(),
                        fmt(tr.f1),
                        fmt(tr.precision),
                        fmt(tr.recall),
                        tr.seconds
                            .map_or("—".into(), |s| hms(Duration::from_secs_f64(s))),
                        state.into(),
                    ],
                    false,
                    p,
                )
                .id(("trial", pos))
                .cursor_pointer()
                .hover(|s| s.bg(p.surface))
                .when(sel.as_ref() == Some(&tr.number), |d| {
                    d.bg(p.surface).font_weight(FontWeight::SEMIBOLD)
                })
                .when(line.is_some() && tr.f1 >= line, |d| d.text_color(p.win))
                .when(tr.state == "FAIL", |d| d.text_color(p.loss))
                .when(tr.state == "PRUNED", |d| d.text_color(p.muted))
                .child(duration_bar(tr.seconds.unwrap_or(0.), max_secs, p))
                .when(!tr.error.is_empty() || !tr.params.is_empty(), |d| {
                    d.tooltip({
                        let e: SharedString = if tr.error.is_empty() {
                            format!("params {}", tr.params)
                        } else {
                            tr.error.clone()
                        }
                        .into();
                        move |w, cx| Tooltip::new(e.clone()).build(w, cx)
                    })
                })
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.sel_trial = Some(n.clone());
                    this.load_trial_epochs();
                    cx.notify();
                })),
            );
        }
        h_flex()
            .size_full()
            .child(list)
            .child(detail.child(div().h(px(24.))))
            .into_any_element()
    }

    // ---------- запуск ----------

    fn render_launch(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let (program, args, title) = self.command(cx);
        let field = |label: &str, input: &Entity<InputState>| {
            v_flex()
                .gap_1()
                .child(
                    div()
                        .text_xs()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(p.muted)
                        .child(label.to_string()),
                )
                .child(Input::new(input))
        };
        let run_all = self.run_all;
        let section = |title: &str| {
            div()
                .pt_4()
                .pb_1()
                .border_b_1()
                .border_color(p.rule)
                .child(display(title, 20., p.ink))
        };

        let targets = h_flex()
            .gap_1()
            .flex_wrap()
            .children(TARGETS.iter().map(|&o| {
                let selected = self.target == o;
                h_flex()
                    .id(ElementId::Name(format!("tg-{o}").into()))
                    .gap_2()
                    .px_3()
                    .py_1p5()
                    .rounded_lg()
                    .cursor_pointer()
                    .border_1()
                    .border_color(if selected { class_color(o) } else { p.rule })
                    .bg(if selected { class_color(o) } else { p.surface })
                    .text_color(if selected {
                        rgb(0xFFFFFF).into()
                    } else {
                        p.ink
                    })
                    .child(div().size(px(10.)).rounded_full().bg(if selected {
                        rgb(0xFFFFFF).into()
                    } else {
                        class_color(o)
                    }))
                    .child(format!("{} · {o}", class_ru(o)))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.target = o;
                        cx.notify();
                    }))
            }));
        let features = h_flex().gap_1().children(FEATURES.iter().map(|&o| {
            choice(
                ElementId::Name(format!("ft-{o}").into()),
                o,
                self.features == o,
            )
            .on_click(cx.listener(move |this, _, _, cx| {
                this.features = o;
                cx.notify();
            }))
        }));
        let tasks = h_flex()
            .gap_1()
            .flex_wrap()
            .children(TASKS.iter().map(|&o| {
                let label = match o {
                    "7" => "№7 отчёт".to_string(),
                    "final" => "final".to_string(),
                    n => format!(
                        "№{n} {}",
                        class_ru(TARGETS[n.parse::<usize>().unwrap() - 2])
                    ),
                };
                choice(
                    ElementId::Name(format!("tk-{o}").into()),
                    label,
                    self.tasks.contains(&o),
                )
                .on_click(cx.listener(move |this, _, _, cx| {
                    match this.tasks.iter().position(|x| *x == o) {
                        Some(i) if this.tasks.len() > 1 => _ = this.tasks.remove(i),
                        Some(_) => {}
                        None => {
                            this.tasks.push(o);
                            this.tasks
                                .sort_by_key(|x| TASKS.iter().position(|y| y == x));
                        }
                    }
                    cx.notify();
                }))
            }));

        // ---- гиперпараметры: Optuna или своя конфигурация ----
        let custom = self.custom && !run_all;
        let (params, errs) = self.custom_params(cx);
        let out_dir = Job::arg(&args, "--output-dir").unwrap_or_default();
        let params_file = Job::arg(&args, "--fixed-params").map(|p| self.root.join(p));
        // --fixed-params ставится первым trial только в пустой study (hpo_pointnet.py).
        let final_data = self.final_data_dir.read(cx).value().trim().to_string();
        let data = self.data_dir.read(cx).value().trim().to_string();
        let train_error = self.train_opts(cx).error();
        let patch_valid = [&self.patch_points, &self.patches_per_scene]
            .iter()
            .all(|i| {
                i.read(cx)
                    .value()
                    .trim()
                    .parse::<usize>()
                    .is_ok_and(|n| n > 0)
            });
        let blocked: Option<String> = if run_all
            && self.tasks.contains(&"final")
            && (final_data.is_empty() || !self.root.join(&final_data).is_dir())
        {
            Some("Выберите существующий каталог данных для финального прогона.".into())
        } else if self.runs_dir.read(cx).value().trim().is_empty() {
            Some("Укажите каталог результатов.".into())
        } else if data.is_empty() || !self.root.join(&data).is_dir() {
            Some("Выберите существующий каталог данных.".into())
        } else if !patch_valid {
            Some("Число точек в патче и патчей на сцену должно быть положительным.".into())
        } else if let Some(e) = train_error {
            Some(e.into())
        } else if [&self.trials, &self.epochs, &self.final_epochs]
            .iter()
            .any(|i| {
                !i.read(cx)
                    .value()
                    .trim()
                    .parse::<usize>()
                    .is_ok_and(|n| n > 0)
            })
        {
            Some("Число trials и эпох должно быть положительным целым.".into())
        } else if let Some(c) = self.conflict((!run_all).then_some(out_dir.as_str())) {
            Some(c)
        } else if custom && self.space_error.is_some() {
            self.space_error.clone()
        } else if custom && !errs.is_empty() {
            Some("Исправьте гиперпараметры, отмеченные красным.".into())
        } else if custom && self.root.join(&out_dir).join("study.db").exists() {
            Some(format!(
                "В {out_dir} уже есть study.db — своя конфигурация применяется только к новому study. Задайте другое имя прогона."
            ))
        } else {
            None
        };
        let mut best_runs: Vec<(String, f64, Params)> = self
            .runs
            .iter()
            .filter(|r| r.target().as_deref() == Some(self.target))
            .filter_map(|r| {
                let b = r.best.as_ref()?;
                Some((
                    r.name.clone(),
                    b.get("f1")?.as_f64()?,
                    b.get("params")?.as_object()?.clone(),
                ))
            })
            .collect();
        best_runs.sort_by(|a, b| b.1.total_cmp(&a.1));
        best_runs.truncate(3);

        let cell = |i: usize, cx: &mut Context<Self>| {
            let spec = &SPACE[i];
            let err = errs.iter().find(|e| e.0 == i).map(|e| e.1.clone());
            let control = match (&self.param_inputs[i], &spec.kind) {
                (Some(input), _) => Input::new(input).into_any_element(),
                (None, Kind::Choice(opts)) => h_flex()
                    .gap_1()
                    .children(opts.iter().map(|&o| {
                        let v: serde_json::Value = serde_json::from_str(o).unwrap();
                        let label = match o {
                            "true" => "да",
                            "false" => "нет",
                            o => o,
                        };
                        choice(
                            ElementId::Name(format!("hp-{}-{o}", spec.key).into()),
                            label,
                            self.param_choice[i] == v,
                        )
                        .on_click(cx.listener(move |this, _, _, cx| {
                            this.param_choice[i] = v.clone();
                            cx.notify();
                        }))
                    }))
                    .into_any_element(),
                _ => div().into_any_element(),
            };
            v_flex()
                .flex_1()
                .min_w_0()
                .gap_1()
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(p.muted)
                                .child(spec.label),
                        )
                        .child(
                            div()
                                .text_xs()
                                .font_family(MONO)
                                .text_color(p.muted)
                                .opacity(0.7)
                                .child(spec.key),
                        ),
                )
                .child(control)
                .child(match err {
                    Some(e) => div().text_xs().text_color(p.loss).child(e),
                    None => div()
                        .text_xs()
                        .text_color(p.muted)
                        .child(if spec.hint.is_empty() {
                            spec.range()
                        } else {
                            format!("{} · {}", spec.range(), spec.hint)
                        }),
                })
        };
        let custom_only = self.custom_only;
        let hyper = v_flex()
            .gap_3()
            .child(section("Гиперпараметры"))
            .child(
                h_flex()
                    .gap_1()
                    .child(choice("hp-optuna", "Подбирает Optuna", !custom).on_click(cx.listener(|this, _, _, cx| {
                        this.custom = false;
                        cx.notify();
                    })))
                    .child(choice("hp-custom", "Своя конфигурация", custom).on_click(cx.listener(|this, _, window, cx| {
                        this.custom = true;
                        if this.params_note.is_none() {
                            this.autopick(window, cx);
                        }
                        cx.notify();
                    }))),
            )
            .when(!custom, |d| {
                d.child(div().text_sm().text_color(p.muted).child("Optuna (TPE) сама выбирает значения в пределах пространства поиска hpo_pointnet.py: learning rate, глубина сети, радиусы и т. д."))
                    .when_some(self.params_note.clone(), |d, n| d.child(div().text_xs().text_color(p.muted).child(n)))
            })
            .when(custom, |d| {
                d.child(
                    h_flex()
                        .gap_2()
                        .flex_wrap()
                        .child(Button::new("autopick").icon(Icon::new(Lucide::WandSparkles)).label("Автоподбор по прогонам").outline().small().on_click(cx.listener(|this, _, window, cx| this.autopick(window, cx))))
                        .children(best_runs.into_iter().enumerate().map(|(k, (name, f1, params))| {
                            Button::new(("best-run", k)).label(format!("Лучший из {name} · F1 {f1:.4}")).ghost().small().on_click(cx.listener(move |this, _, window, cx| {
                                this.apply_params(&params, window, cx);
                                this.params_note = Some(format!("Параметры лучшего trial прогона {name} (F1 {f1:.4})."));
                                cx.notify();
                            }))
                        })),
                )
                .when_some(self.params_note.clone(), |d, n| d.child(div().text_xs().text_color(p.muted).child(n)))
                .when_some(self.space_error.clone(), |d, e| d.child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(p.loss).child(e)))
                .child(h_flex().gap_4().items_start().child(cell(0, cx)).child(cell(1, cx)).child(cell(2, cx)))
                .child(h_flex().gap_4().items_start().child(cell(3, cx)).child(cell(4, cx)).child(cell(8, cx)))
                .child(h_flex().gap_4().items_start().child(cell(5, cx)).child(cell(6, cx)).child(cell(7, cx)))
                .child(
                    h_flex()
                        .gap_1()
                        .child(choice("use-only", "Обучить только эту конфигурацию (1 trial)", custom_only).on_click(cx.listener(|this, _, _, cx| {
                            this.custom_only = true;
                            cx.notify();
                        })))
                        .child(choice("use-seed", "Начать с неё подбор Optuna", !custom_only).on_click(cx.listener(|this, _, _, cx| {
                            this.custom_only = false;
                            cx.notify();
                        }))),
                )
                .child(div().text_xs().text_color(p.muted).child(format!(
                    "Конфигурация сохранится в {}.params.json и пойдёт первым trial нового study.",
                    out_dir
                )))
            });

        let form = v_flex()
            .id("launch")
            .size_full()
            .overflow_y_scrollbar()
            .child(
                v_flex()
                    .max_w(px(880.))
                    .px_8()
                    .py_6()
                    .gap_3()
                    .child(display("Новый запуск", 40., p.ink))
                    .child(
                        h_flex()
                            .gap_1()
                            .child(choice("m-single", "Подбор для одного класса", !run_all).on_click(cx.listener(|this, _, _, cx| {
                                this.run_all = false;
                                cx.notify();
                            })))
                            .child(choice("m-all", "Сценарий задач run_all.py", run_all).on_click(cx.listener(|this, _, _, cx| {
                                this.run_all = true;
                                cx.notify();
                            }))),
                    )
                    .when(!run_all, |d| {
                        d.child(section("Класс"))
                            .child(targets)
                            .child(section("Признаки"))
                            .child(features)
                            .child(hyper)
                            .child(section("Прогон"))
                            .child(field("Имя прогона", &self.out_name))
                            .child(div().text_xs().text_color(p.muted).child("Результаты пишутся в <каталог результатов>/<имя>. С тем же именем подбор продолжится из study.db."))
                    })
                    .when(run_all, |d| {
                        d.child(section("Задачи"))
                            .child(div().text_sm().text_color(p.muted).child("№2–6 — подбор по классам, №7 — сводный отчёт, final — финальные прогоны с лучшими параметрами."))
                            .child(tasks)
                            .child(field("Эпох в финальном прогоне", &self.final_epochs))
                            .child(field("Данные для финала", &self.final_data_dir))
                    })
                    .child(section("Обучение"))
                    .child(
                        h_flex()
                            .gap_3()
                            .child(field(if custom && custom_only { "Trials (своя конфигурация — 1)" } else { "Trials" }, &self.trials).flex_1())
                            .child(field("Эпох на trial", &self.epochs).flex_1())
                            .child(field("Seed", &self.seed).flex_1())
                            .child(field("Workers", &self.workers).flex_1()),
                    )
                    .child(
                        h_flex()
                            .gap_3()
                            .items_end()
                            .child(
                                h_flex()
                                    .gap_1()
                                    .child(choice("loss-ce", "CrossEntropy", self.loss == "ce").on_click(cx.listener(|this, _, _, cx| {
                                        this.loss = "ce";
                                        cx.notify();
                                    })))
                                    .child(choice("loss-focal", "Focal Loss", self.loss == "focal").on_click(cx.listener(|this, _, _, cx| {
                                        this.loss = "focal";
                                        cx.notify();
                                    })))
                                    .child(choice("loss-lovasz", "Focal + Lovász", self.loss == "focal_lovasz").on_click(cx.listener(|this, _, _, cx| {
                                        this.loss = "focal_lovasz";
                                        cx.notify();
                                    }))),
                            )
                            .when(self.loss != "ce", |d| d.child(field("Gamma", &self.focal_gamma).w(px(120.))))
                            .when(self.loss == "focal_lovasz", |d| d.child(field("Доля Lovász", &self.lovasz_weight).w(px(120.)))),
                    )
                    .child(
                        h_flex()
                            .gap_3()
                            .items_end()
                            .child(field("Степень весов классов", &self.weight_power).w(px(200.)))
                            .child(div().pb_2().text_xs().text_color(p.muted).child("1 — обратные частоты классов, меньше — мягче (для редких классов 0.4–0.5), 0 — без весов.")),
                    )
                    .child(
                        h_flex()
                            .gap_2()
                            .child(Button::new("quick").icon(Icon::new(Lucide::Zap)).label("Быстрая проверка: 1 trial, 1 эпоха").outline().small().on_click(cx.listener(|this, _, window, cx| {
                                Self::set_input(&this.trials, "1".into(), window, cx);
                                Self::set_input(&this.epochs, "1".into(), window, cx);
                                cx.notify();
                            })))
                            .child(Button::new("rare").label("Редкий класс: Focal + Lovász, баланс и граница").outline().small().tooltip("Focal + Lovász 0.5, степень весов 0.47, 8 патчей на сцену: половина на границе, остальные поровну на цели и фоне, вес граничных точек 2, признаки xyz_hag").on_click(cx.listener(|this, _, window, cx| {
                                this.loss = "focal_lovasz";
                                this.equibatch = true;
                                this.features = "xyz_hag";
                                for (input, v) in [
                                    (&this.focal_gamma, "2"),
                                    (&this.lovasz_weight, "0.5"),
                                    (&this.weight_power, "0.47"),
                                    (&this.positive_fraction, "0.5"),
                                    (&this.boundary_fraction, "0.5"),
                                    (&this.boundary_weight, "2"),
                                    (&this.boundary_radius, "0.5"),
                                    (&this.patches_per_scene, "8"),
                                ] {
                                    Self::set_input(input, v.into(), window, cx);
                                }
                                cx.notify();
                            }))),
                    )
                    .child(h_flex().gap_3().child(field("Точек в патче", &self.patch_points).flex_1()).child(field("Патчей на сцену", &self.patches_per_scene).flex_1()))
                    .child(
                        h_flex()
                            .gap_1()
                            .child(choice("patch-random", "Центр патча случайный", !self.equibatch).on_click(cx.listener(|this, _, _, cx| {
                                this.equibatch = false;
                                cx.notify();
                            })))
                            .child(choice("patch-balance", "Баланс: цель / фон", self.equibatch).on_click(cx.listener(|this, _, _, cx| {
                                this.equibatch = true;
                                cx.notify();
                            }))),
                    )
                    .when(self.equibatch, |d| {
                        d.child(
                            h_flex()
                                .gap_3()
                                .child(field("Доля патчей с центром на цели", &self.positive_fraction).flex_1())
                                .child(field("Доля граничных патчей", &self.boundary_fraction).flex_1()),
                        )
                    })
                    .child(
                        h_flex()
                            .gap_3()
                            .child(field("Вес граничных точек", &self.boundary_weight).flex_1())
                            .child(field("Радиус границы (ед. координат)", &self.boundary_radius).flex_1()),
                    )
                    .child(div().text_xs().text_color(p.muted).child("Только Train, Val не меняется. Граничная точка — та, у которой в радиусе есть точка другого класса. Вес 1 и доля 0 — выключено."))
                    .child(section("Окружение"))
                    .child(h_flex().gap_3().child(field("Данные для подбора", &self.data_dir).flex_1()).child(field("Python", &self.python).flex_1()))
                    .child(field("Каталог результатов", &self.runs_dir))
                    .child(field("Дополнительные аргументы", &self.extra))
                    .child(div().h(px(24.))),
            );
        // Команда и запуск всегда на виду, под формой.
        let footer = h_flex()
            .flex_shrink_0()
            .px_8()
            .py_3()
            .gap_4()
            .border_t_1()
            .border_color(p.rule)
            .bg(p.surface)
            .child(
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .gap_1()
                    .child(small_caps("Команда", p.muted))
                    .child(
                        div()
                            .font_family(MONO)
                            .text_xs()
                            .text_color(p.ink)
                            .child(format!("{program} {}", args.join(" "))),
                    )
                    .when_some(blocked.clone(), |d, b| {
                        d.child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(p.loss)
                                .child(b),
                        )
                    }),
            )
            .child({
                let step = scenario::Step {
                    title: title.clone(),
                    args: args.clone(),
                    fixed_params: params_file.as_ref().map_or(serde_json::Value::Null, |_| {
                        serde_json::Value::Object(params.clone())
                    }),
                };
                // Занятый каталог и study.db проверяются при запуске шага; здесь — только ошибки в самой конфигурации.
                let bad = custom && (self.space_error.is_some() || !errs.is_empty());
                Button::new("to-scenario")
                    .icon(Icon::new(Lucide::ListPlus))
                    .label(match self.scenarios.get(self.sel_scenario) {
                        Some(s) => format!("В «{}» · {}", s.name, s.steps.len()),
                        None => "В новый сценарий".into(),
                    })
                    .outline()
                    .large()
                    .disabled(bad)
                    .tooltip("Добавить эту команду шагом в сценарий (вкладка «Сценарии»)")
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.add_step(step.clone());
                        cx.notify();
                    }))
            })
            .child(
                Button::new("go")
                    .icon(Icon::new(Lucide::Play))
                    .label("Запустить")
                    .primary()
                    .large()
                    .disabled(blocked.is_some())
                    .on_click(cx.listener(move |this, _, _, cx| {
                        if let Some(path) = &params_file {
                            let json = serde_json::to_string_pretty(&params).unwrap();
                            if let Err(e) = path
                                .parent()
                                .map_or(Ok(()), std::fs::create_dir_all)
                                .and_then(|_| std::fs::write(path, json))
                            {
                                this.params_note =
                                    Some(format!("Не удалось записать {}: {e}", show_path(path)));
                                cx.notify();
                                return;
                            }
                        }
                        this.spawn(program.clone(), args.clone(), title.clone());
                        cx.notify();
                    })),
            );
        v_flex()
            .size_full()
            .child(div().flex_1().min_h_0().overflow_hidden().child(form))
            .child(footer)
            .into_any_element()
    }

    // ---------- датасеты ----------

    fn render_datasets(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let list = v_flex()
            .id("ds-list")
            .w(px(300.))
            .h_full()
            .flex_shrink_0()
            .border_r_1()
            .border_color(p.rule)
            .overflow_y_scrollbar()
            .child(
                div()
                    .px_4()
                    .pt_4()
                    .pb_2()
                    .child(display("Датасеты", 22., p.ink)),
            )
            .children(self.datasets.iter().enumerate().map(|(i, d)| {
                v_flex()
                    .id(("ds", i))
                    .mx_2()
                    .px_3()
                    .py_2()
                    .rounded_lg()
                    .cursor_pointer()
                    .hover(|s| s.bg(p.raised))
                    .when(i == self.sel_dataset, |d| d.bg(p.raised))
                    .child(
                        div()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(p.ink)
                            .child(d.name.clone()),
                    )
                    .child(
                        div()
                            .text_xs()
                            .font_features(tnum())
                            .text_color(p.muted)
                            .child(format!(
                                "{} сцен · {} · {:.1} млн точек",
                                d.scenes.len(),
                                human_bytes(d.bytes()),
                                d.points() as f64 / 1e6
                            )),
                    )
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.sel_dataset = i;
                        cx.notify();
                    }))
            }))
            .when(self.datasets.is_empty(), |d| {
                d.child(
                    div()
                        .px_4()
                        .text_color(p.muted)
                        .child("В корне проекта нет каталогов с .npy."),
                )
            });

        let Some(d) = self.datasets.get(self.sel_dataset) else {
            return h_flex().size_full().child(list).into_any_element();
        };
        let name = d.name.clone();
        let path = d.path.clone();
        let max = d
            .scenes
            .iter()
            .filter_map(|s| s.shape.first())
            .max()
            .copied()
            .unwrap_or(1)
            .max(1);
        let min = d
            .scenes
            .iter()
            .filter_map(|s| s.shape.first())
            .min()
            .copied()
            .unwrap_or(0);
        let cols = d
            .scenes
            .first()
            .and_then(|s| s.shape.get(1))
            .copied()
            .unwrap_or(0);
        let dark = self.dark;
        let scenes: Vec<(String, u64, u64)> = d
            .scenes
            .iter()
            .map(|s| {
                (
                    s.name.clone(),
                    s.shape.first().copied().unwrap_or(0),
                    s.bytes,
                )
            })
            .collect();
        let n = scenes.len();
        let dir = d.path.clone();

        let detail = v_flex()
            .flex_1()
            .min_w_0()
            .h_full()
            .child(
                h_flex()
                    .px_6()
                    .py_4()
                    .flex_wrap()
                    .gap_4()
                    .border_b_1()
                    .border_color(p.rule)
                    // Имя не переносится по буквам: в узком окне кнопки уходят на следующую строку.
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w(px(220.))
                            .overflow_hidden()
                            .whitespace_nowrap()
                            .text_ellipsis()
                            .child(display(&d.name, 34., p.ink))
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(p.muted)
                                    .overflow_hidden()
                                    .text_ellipsis()
                                    .child(show_path(&d.path)),
                            ),
                    )
                    .child(
                        band_button("ds-open", Lucide::FolderOpen, "Папка", p.raised, p.ink)
                            .on_click(move |_, _, _| open_path(&path)),
                    )
                    .when(
                        !d.name.ends_with("_mini") && d.scenes.len() > MINI_SCENES,
                        |h| {
                            h.child(
                                band_button("ds-mini", Lucide::Zap, "Мини-копия", p.raised, p.ink)
                                    .on_click(cx.listener(|this, _, _, cx| this.make_mini(cx))),
                            )
                        },
                    )
                    .child(
                        band_button("ds-use", Lucide::Play, "Обучать на нём", p.ink, p.bg)
                            .on_click(cx.listener(move |this, _, window, cx| {
                                Self::set_input(&this.data_dir, name.clone(), window, cx);
                                this.tab = Tab::Launch;
                                cx.notify();
                            })),
                    ),
            )
            .when_some(self.ds_note.clone(), |d, (ok, note)| {
                d.child(
                    div()
                        .px_6()
                        .py_2()
                        .border_b_1()
                        .border_color(p.rule)
                        .text_sm()
                        .text_color(if ok { p.win } else { p.loss })
                        .child(note),
                )
            })
            .child(
                h_flex()
                    .px_6()
                    .py_4()
                    .flex_wrap()
                    .gap_x_10()
                    .gap_y_3()
                    .border_b_1()
                    .border_color(p.rule)
                    .child(stat_big("Сцен", d.scenes.len().to_string(), p))
                    .child(stat_big(
                        "Точек",
                        format!("{:.1} млн", d.points() as f64 / 1e6),
                        p,
                    ))
                    .child(stat_big("На диске", human_bytes(d.bytes()), p))
                    .child(stat_big(
                        "На сцену",
                        format!("{}–{} тыс.", min / 1000, max / 1000),
                        p,
                    ))
                    .child(stat_big("Колонок", cols.to_string(), p)),
            )
            .child(
                h_flex()
                    .flex_1()
                    .min_h_0()
                    .child(
                        // Таблица и 3D-превью делят ширину; таблица не шире 480.
                        v_flex()
                            .flex_1()
                            .min_w(px(320.))
                            .max_w(px(480.))
                            .h_full()
                            .child(trow(
                                ["Сцена", "Точек", "Размер"].map(String::from),
                                true,
                                p,
                            ))
                            .child(
                                uniform_list(
                                    "scenes",
                                    n,
                                    cx.processor(
                                        move |this, range: std::ops::Range<usize>, _, cx| {
                                            let p = pal(dark);
                                            let open =
                                                this.preview.as_ref().map(|pv| pv.scene.clone());
                                            range
                                                .map(|i| {
                                                    let (name, pts, bytes) = &scenes[i];
                                                    let scene = dir.join(name);
                                                    trow(
                                                        [
                                                            name.clone(),
                                                            thousands(*pts),
                                                            human_bytes(*bytes),
                                                        ],
                                                        false,
                                                        p,
                                                    )
                                                    .cursor_pointer()
                                                    .hover(|s| s.bg(p.surface))
                                                    .when(open.as_ref() == Some(&scene), |d| {
                                                        d.bg(p.surface)
                                                            .font_weight(FontWeight::SEMIBOLD)
                                                    })
                                                    .child(duration_bar(*pts as f64, max as f64, p))
                                                    .on_click(cx.listener(move |this, _, _, cx| {
                                                        this.open_preview(scene.clone(), cx)
                                                    }))
                                                })
                                                .collect()
                                        },
                                    ),
                                )
                                .flex_1(),
                            ),
                    )
                    .child(self.render_preview(p, cx)),
            );
        h_flex()
            .size_full()
            .child(list)
            .child(detail)
            .into_any_element()
    }

    /// Текст файла с кэшем: перечитывается только при смене пути, размера или времени изменения.
    /// Показываются последние `max_lines` строк (лог растёт часами); копия `SharedString` дешёвая.
    fn artifact_text(&mut self, path: &Path, max_lines: usize) -> Option<SharedString> {
        let meta = std::fs::metadata(path).ok()?;
        let key = (path.to_path_buf(), meta.len(), meta.modified().ok());
        if let Some((k, text)) = &self.artifact_text {
            if *k == key {
                return Some(text.clone());
            }
        }
        let mut text = read_artifact(path)?;
        let n = text.lines().count();
        if n > max_lines {
            let tail: Vec<&str> = text.lines().skip(n - max_lines).collect();
            text = format!("… показаны последние {max_lines} строк из {n}\n{}", tail.join("\n"));
        }
        let text = SharedString::from(text);
        self.artifact_text = Some((key, text.clone()));
        Some(text)
    }

    fn render_artifacts(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let sidebar = |id: &'static str| {
            v_flex().id(id).w(px(250.)).flex_shrink_0().h_full().overflow_y_scrollbar()
                .border_r_1().border_color(p.rule).p_3().gap_1()
        };
        let tabs = h_flex()
            .px_8()
            .pt_6()
            .border_b_1().border_color(p.rule)
            .children(ArtifactView::ALL.into_iter().enumerate().map(|(i, view)| {
                let active = self.artifact_view == view;
                div().id(("artifact-tab", i)).px_3().py_3().cursor_pointer()
                    .border_b_2().border_color(if active { p.ink } else { p.rule })
                    .text_color(if active { p.ink } else { p.muted })
                    .when(active, |d| d.font_weight(FontWeight::SEMIBOLD))
                    .child(view.label())
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.artifact_view = view;
                        if view == ArtifactView::Models {
                            this.imported = imported_models(&this.root);
                        }
                        cx.notify();
                    }))
            }));
        if self.artifact_view == ArtifactView::Models {
            // Тот же список, что в «Тестировании»: модель лучшего trial каждого прогона и models/*.pt.
            let models = self.test_models();
            let col = |w: f32| div().w(px(w)).flex_shrink_0();
            let header = h_flex().px_8().py_2().gap_4().border_b_1().border_color(p.rule).text_xs().text_color(p.muted)
                .child(div().flex_1().min_w_0().child("МОДЕЛЬ"))
                .child(col(90.).child("F1"))
                .child(col(90.).child("РАЗМЕР"))
                .child(col(210.));
            let rows = models.iter().enumerate().map(|(i, m)| {
                let run = m.run.and_then(|i| self.runs.get(i));
                let sub = match run {
                    Some(r) => format!(
                        "прогон · {} · {}",
                        r.target().map_or("?".into(), |t| class_ru(&t).to_string()),
                        r.features().unwrap_or_else(|| "?".into())
                    ),
                    None => format!("загружена · {MODELS_DIR}"),
                };
                let size = std::fs::metadata(&m.file).map_or("—".into(), |meta| human_bytes(meta.len()));
                let key = m.key.clone();
                let folder = m.file.parent().map(Path::to_path_buf).unwrap_or_default();
                h_flex().id(("artifact-model", i)).px_8().py_2().gap_4().items_center().border_b_1().border_color(p.rule)
                    .child(v_flex().flex_1().min_w_0()
                        .child(div().font_weight(FontWeight::SEMIBOLD).text_color(p.ink).child(m.label.clone()))
                        .child(div().text_xs().text_color(p.muted).child(sub))
                        .child(div().text_xs().text_color(p.muted).overflow_hidden().whitespace_nowrap().text_ellipsis().child(show_path(&m.file))))
                    .child(col(90.).font_features(tnum()).text_color(p.ink).child(fmt(run.and_then(|r| r.best_f1()))))
                    .child(col(90.).font_features(tnum()).text_color(p.muted).child(size))
                    .child(col(210.).child(h_flex().gap_2().justify_end()
                        .child(Button::new(("artifact-model-test", i)).icon(Icon::new(Lucide::FlaskConical)).label("Тестировать").outline().small()
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.test_model = Some(key.clone());
                                this.tab = Tab::Testing;
                                this.refresh_tests();
                                cx.notify();
                            })))
                        .child(Button::new(("artifact-model-dir", i)).icon(Icon::new(Lucide::FolderOpen)).ghost().small()
                            .tooltip("Открыть папку")
                            .on_click(move |_, _, _| open_path(&folder)))))
            }).collect::<Vec<_>>();
            let body = v_flex().id("artifact-models").flex_1().min_h_0().overflow_y_scrollbar()
                .child(header)
                .children(rows)
                .when(models.is_empty(), |d| d.child(div().px_8().py_4().text_color(p.muted)
                    .child("Моделей пока нет: обучите прогон или загрузите .pt во вкладке «Тестирование».")));
            return v_flex().size_full().child(tabs).child(body).into_any_element();
        }
        if self.artifact_view == ArtifactView::Logs {
            let logs: Vec<(String, PathBuf)> = self.runs.iter().filter_map(|r| artifact_log_path(&self.root, &r.name).map(|path| (r.name.clone(), path))).collect();
            let selected = self.artifact_log.as_ref().and_then(|name| logs.iter().position(|(n, _)| n == name)).unwrap_or(0);
            let list = sidebar("artifact-log-runs")
                .child(div().px_3().text_xs().text_color(p.muted).child("ПРОГОНЫ С ЛОГАМИ"))
                .when(logs.is_empty(), |d| d.child(div().p_3().text_color(p.muted).child("Сохранённые логи не найдены")))
                .children(logs.iter().enumerate().map(|(i, (name, _))| {
                    let name = name.clone();
                    div().id(("artifact-log-run", i)).px_3().py_2().rounded_md().cursor_pointer()
                        .when(i == selected, |d| d.bg(p.raised).font_weight(FontWeight::SEMIBOLD))
                        .child(name.clone()).on_click(cx.listener(move |this, _, _, cx| {
                            this.artifact_log = Some(name.clone());
                            cx.notify();
                        }))
                }));
            let detail = if let Some((name, path)) = logs.get(selected) {
                let text = self.artifact_text(path, 5000);
                let path = path.clone();
                let mut header = h_flex().p_4().gap_3().items_center()
                    .child(div().flex_1().min_w_0().child(name.clone()));
                if text.is_some() {
                    let full = path.clone();
                    header = header.child(band_button("artifact-log-copy", Lucide::Copy, "Копировать", p.raised, p.ink)
                        .on_click(move |_, _, cx| {
                            if let Some(all) = read_artifact(&full) {
                                cx.write_to_clipboard(ClipboardItem::new_string(all));
                            }
                        }));
                }
                let header = header.child(band_button("artifact-log-open", Lucide::FileText, "Открыть файл", p.raised, p.ink)
                    .on_click(move |_, _, _| open_path(&path)));
                v_flex().flex_1().min_w_0().h_full()
                    .child(header)
                    .child(div().id("artifact-log-content").flex_1().min_h_0().overflow_scroll().p_4().font_family(MONO).font_features(tnum()).text_sm().whitespace_nowrap()
                        .child(text.unwrap_or_else(|| "Лог пуст или не читается.".into())))
                    .into_any_element()
            } else {
                div().flex_1().p_6().text_color(p.muted).child("Нет доступных сохранённых логов прогонов.").into_any_element()
            };
            return v_flex().size_full().child(tabs).child(h_flex().flex_1().min_h_0().child(list).child(detail)).into_any_element();
        }
        let runs_dir = PathBuf::from(self.runs_dir.read(cx).value().trim());
        let runs_dir = if runs_dir.is_absolute() { runs_dir } else { self.root.join(runs_dir) };
        let docs = artifact_documents(&self.root, &runs_dir);
        let selected = artifact_selected(&docs, self.artifact_doc.as_deref());
        let document = docs.get(selected);
        let document_text = document.and_then(|d| self.artifact_text(&d.path, usize::MAX));
        let document_item = |i: usize, label: String| {
            let path = docs[i].path.clone();
            let icon = if path.file_name().is_some_and(|name| name == "data_summary.md") { Lucide::Database } else { Lucide::FileText };
            h_flex()
                .id(("artifact-document", i))
                .px_3().py_2().rounded_md().cursor_pointer()
                .when(i == selected, |d| d.bg(p.raised).font_weight(FontWeight::SEMIBOLD))
                .gap_2().items_center()
                .child(Icon::new(icon).size_4().text_color(p.muted))
                .child(div().min_w_0().overflow_hidden().whitespace_nowrap().text_ellipsis().child(label))
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.artifact_doc = Some(path.clone());
                    this.nevo_note = None;
                    cx.notify();
                }))
        };
        let selector = sidebar("artifact-document-list")
            .child(document_item(0, "Сводный отчёт".into()))
            .children(artifact_groups(&docs).into_iter().map(|(name, indices)| {
                v_flex().pt_3().gap_1()
                    .child(div().px_3().text_xs().text_color(p.muted).overflow_hidden().whitespace_nowrap().text_ellipsis().child(name))
                    .children(indices.into_iter().map(|i| {
                        let label = if docs[i].path.file_name().is_some_and(|n| n == "report.md") { "Отчёт" } else { "Данные" };
                        document_item(i, label.into())
                    }))
            }));
        let content = match document_text.as_ref().map(|t| t.as_ref()) {
            Some(text) => v_flex()
                .gap_2()
                .font_features(tnum())
                .children(artifact_markdown::parse_markdown(text).into_iter().map(|block| {
                    use artifact_markdown::Block;
                    match block {
                        Block::Heading { level, text } => div().pt_2().font_weight(FontWeight::SEMIBOLD).text_size(px(match level { 1 => 24., 2 => 20., 3 => 17., _ => 15. })).child(text),
                        Block::Paragraph(text) => div().child(text),
                        Block::List { ordered, start, items } => v_flex().gap_1().children(items.into_iter().enumerate().map(|(i, item)| {
                            let pad = item.chars().take_while(|c| *c == '\u{a0}').count();
                            div().pl(px(pad as f32 * 6.)).child(format!("{} {}", if ordered { format!("{}.", start + i) } else { "•".into() }, item.trim_start_matches('\u{a0}')))
                        })),
                        Block::Table { headers, rows } => {
                            let row = |cells: Vec<String>| h_flex().border_b_1().border_color(p.rule).children(cells.into_iter().map(|cell| div().flex_1().min_w(px(100.)).p_2().child(cell)));
                            v_flex().border_1().border_color(p.rule).child(row(headers).font_weight(FontWeight::SEMIBOLD)).children(rows.into_iter().map(row))
                        }
                        Block::Code { language: _, text } => div().p_3().bg(p.raised).font_family(MONO).text_sm().child(text),
                    }
                }))
                .into_any_element(),
            None => v_flex()
                .gap_2()
                .child(display("Документ недоступен", 24., p.ink))
                .child(div().text_color(p.muted).child(document.map_or("Файл не найден".into(), |d| format!("Файл пуст или не читается: {}", d.path.display()))))
                .into_any_element(),
        };
        v_flex()
            .size_full()
            .child(tabs)
            .child(h_flex().flex_1().min_h_0().pt_4()
                .child(selector)
                .child(v_flex().flex_1().min_w_0().h_full()
                    .when_some(document, |d, doc| {
                        let (title, path) = (doc.title.clone(), doc.path.clone());
                        d.child(h_flex().px_8().pt_2().gap_3().items_center()
                            .child(v_flex().flex_1().min_w_0()
                                .child(div().font_weight(FontWeight::SEMIBOLD).child(doc.title.clone()))
                                .child(div().pt_1().text_xs().text_color(p.muted).child(show_path(&doc.path))))
                            .when(document_text.is_some(), |h| h.child(
                                band_button("artifact-nevo", Lucide::Share, "В Nevo", p.raised, p.ink)
                                    .on_click(cx.listener(move |this, _, _, cx| this.export_to_nevo(title.clone(), path.clone(), cx))))))
                        .when_some(self.nevo_note.clone(), |d, (ok, note)| d.child(
                            div().px_8().pt_2().text_sm().text_color(if ok { p.win } else { p.loss }).child(note)))
                    })
                    .child(div().id("artifact-content").flex_1().min_h_0().overflow_y_scrollbar().px_8().py_4().child(content))))
            .into_any_element()
    }

    fn render_testing(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let models = self.test_models();
        let (sel, dataset) = self.test_pick(&models);
        let sel_key = sel.map(|m| m.key.clone());
        let list = v_flex()
            .id("tm-list")
            .w(px(300.))
            .h_full()
            .flex_shrink_0()
            .border_r_1()
            .border_color(p.rule)
            .overflow_y_scrollbar()
            .child(
                v_flex()
                    .px_4()
                    .pt_4()
                    .pb_2()
                    .gap_2()
                    .child(display("Модели", 22., p.ink))
                    .child(Button::new("tm-import").icon(Icon::new(Lucide::Upload)).label("Загрузить .pt").outline().small().on_click(cx.listener(|this, _, window, cx| this.import_models(window, cx))))
                    .child(div().text_xs().text_color(p.muted).child("Модели ваших прогонов и загруженные файлы .pt — например, обученные на другом компьютере.")),
            )
            .children(models.iter().map(|m| {
                let sub = match m.run.and_then(|i| self.runs.get(i)) {
                    Some(r) => format!("прогон · {} · F1 {}", r.target().map_or("?".into(), |t| class_ru(&t).to_string()), fmt(r.best_f1())),
                    None => format!("загружена · {MODELS_DIR}/{}", m.file.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()),
                };
                let key = m.key.clone();
                v_flex()
                    .id(SharedString::from(format!("tm-{}", m.key)))
                    .mx_2()
                    .px_3()
                    .py_2()
                    .rounded_lg()
                    .cursor_pointer()
                    .hover(|s| s.bg(p.raised))
                    .when(Some(&m.key) == sel_key.as_ref(), |d| d.bg(p.raised))
                    .child(div().font_weight(FontWeight::SEMIBOLD).text_color(p.ink).child(m.label.clone()))
                    .child(div().text_xs().text_color(p.muted).child(sub))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.test_model = Some(key.clone());
                        this.refresh_tests();
                        cx.notify();
                    }))
            }))
            .when(models.is_empty(), |d| d.child(div().px_4().text_color(p.muted).child("Пока нет ни одной модели: обучите прогон или загрузите .pt.")));

        let Some(m) = sel else {
            return h_flex().size_full().child(list).into_any_element();
        };
        let run = m.run.and_then(|i| self.runs.get(i));
        let eval = self.test_eval.clone();
        let info = |k: &str| {
            eval.as_ref()
                .and_then(|e| e["model"][k].as_str().map(String::from))
        };
        let target = run.and_then(|r| r.target()).or_else(|| info("target"));
        let features = run.and_then(|r| r.features()).or_else(|| info("features"));
        let folder = m.file.parent().map(Path::to_path_buf).unwrap_or_default();

        let sel_ds = dataset.map(|d| d.name.clone());
        let datasets = h_flex()
            .gap_1()
            .flex_wrap()
            .children(self.datasets.iter().map(|d| {
                let name = d.name.clone();
                choice(
                    SharedString::from(format!("td-{}", d.name)),
                    format!("{} · {} сцен", d.name, d.scenes.len()),
                    Some(&d.name) == sel_ds.as_ref(),
                )
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.test_dataset = Some(name.clone());
                    this.refresh_tests();
                    cx.notify();
                }))
            }));
        let running = dataset.is_some_and(|d| {
            let t = Self::test_title(m, d);
            self.jobs.iter().any(|j| j.running() && j.title == t)
        });
        let (done, total) = eval.as_ref().map_or((0, 0), |e| {
            (
                e["done"].as_u64().unwrap_or(0),
                e["total"].as_u64().unwrap_or(0),
            )
        });
        let note = if dataset.is_none() {
            "Нет датасетов: положите каталог со scene_*.npy в корень проекта.".to_string()
        } else if running {
            format!(
                "Идёт тест: {done} из {} сцен, лог — в «Трансляции». Таблица пополняется по ходу.",
                total.max(done)
            )
        } else if eval.is_none() {
            "Модель предсказывает каждую выбранную сцену целиком (GPU, минуты на сцену). Сцены берутся равномерно по датасету.".to_string()
        } else if done < total {
            format!(
                "Посчитано {done} из {total} сцен — тест прервали; «Досчитать» продолжит с места остановки."
            )
        } else {
            format!(
                "{total} сцен, худшие сверху. Клик — сцена в 3D с ошибками модели. Другое число сцен — «Тестировать» заново."
            )
        };

        let dir = dataset
            .map(|d| test_dir(&self.root, &m.tag, &d.name))
            .unwrap_or_default();
        let (t, label) = (
            target.clone().unwrap_or_default(),
            format!("{} · тест", m.label),
        );
        let open: OpenScene = Rc::new(move |this, path, cx| {
            let pred = dir.join(format!(
                "{}__test.npy",
                path.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            ));
            this.open_test_scene(path, pred, t.clone(), label.clone(), cx)
        });
        let table = self.scene_table(eval.as_ref(), run, p, cx, open);
        let pooled = eval.as_ref().and_then(pooled);
        let can_run = dataset.is_some() && !running;

        let detail = v_flex()
            .id("tm-detail")
            .flex_1()
            .min_w_0()
            .h_full()
            .overflow_y_scrollbar()
            .child(
                h_flex()
                    .px_6()
                    .py_4()
                    .flex_wrap()
                    .gap_4()
                    .border_b_1()
                    .border_color(p.rule)
                    .child(v_flex().flex_1().min_w(px(220.)).overflow_hidden().whitespace_nowrap().text_ellipsis().child(display(&m.label, 34., p.ink)).child(div().text_xs().text_color(p.muted).overflow_hidden().text_ellipsis().child(show_path(m.file.strip_prefix(&self.root).unwrap_or(&m.file)))))
                    .child(band_button("tm-open", Lucide::FolderOpen, "Папка", p.raised, p.ink).on_click(move |_, _, _| open_path(&folder)))
                    .when(can_run, |h| {
                        h.child(band_button("tm-run", Lucide::Play, if done > 0 && done < total { "Досчитать" } else { "Тестировать" }, p.ink, p.bg).on_click(cx.listener(|this, _, _, cx| {
                            this.run_test(cx);
                            cx.notify();
                        })))
                    }),
            )
            .when_some(self.test_note.clone(), |d, (ok, note)| {
                d.child(div().px_6().py_2().border_b_1().border_color(p.rule).text_sm().text_color(if ok { p.win } else { p.loss }).child(note))
            })
            .child(
                h_flex()
                    .px_6()
                    .py_4()
                    .flex_wrap()
                    .gap_x_10()
                    .gap_y_3()
                    .border_b_1()
                    .border_color(p.rule)
                    .child(stat_big("Класс", target.as_deref().map_or("после теста".into(), |t| class_ru(t).to_string()), p))
                    .child(stat_big("Признаки", features.unwrap_or_else(|| "—".into()), p))
                    .when_some(run, |d, r| d.child(stat_big("F1 при обучении", fmt(r.best_f1()), p)))
                    .when_some(pooled, |d, [f1, pr, rc, base]| {
                        d.child(stat_big("F1 теста", fmt(Some(f1)), p))
                            .child(stat_big("Precision", fmt(Some(pr)), p))
                            .child(stat_big("Recall", fmt(Some(rc)), p))
                            .child(stat_big("Baseline F1", fmt(Some(base)), p))
                    }),
            )
            .child(
                v_flex()
                    .px_6()
                    .py_4()
                    .gap_3()
                    .child(small_caps("Датасет для теста", p.muted))
                    .child(datasets)
                    .child(
                        h_flex()
                            .gap_3()
                            .items_center()
                            .child(div().text_sm().text_color(p.muted).child("Сцен"))
                            .child(div().w(px(90.)).child(Input::new(&self.test_scenes)))
                            .child(div().text_sm().text_color(p.muted).child(dataset.map_or(String::new(), |d| format!("из {}", d.scenes.len())))),
                    )
                    .child(div().text_xs().text_color(p.muted).child(note))
                    .when(run.is_some() && table.is_some(), |d| {
                        d.child(div().text_xs().text_color(p.muted).child("Выборка: train — сцена была в обучении (метрики завышены), val — в валидации, новая — модель её не видела."))
                    })
                    .children(table),
            );
        h_flex()
            .size_full()
            .child(list)
            .child(detail)
            .into_any_element()
    }

    fn render_preview(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let panel = v_flex()
            .flex_1()
            .min_w_0()
            .h_full()
            .border_l_1()
            .border_color(p.rule);
        let Some(pv) = &self.preview else {
            return panel
                .justify_center()
                .items_center()
                .gap_2()
                .child(display("3D-превью", 28., p.ink))
                .child(
                    div()
                        .text_color(p.muted)
                        .child("Выберите сцену в таблице — покажу облако точек."),
                )
                .into_any_element();
        };
        let scene = pv
            .scene
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let by = pv.by;
        let toolbar = h_flex()
            .px_4()
            .py_2()
            .flex_wrap()
            .gap_2()
            .border_b_1()
            .border_color(p.rule)
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .overflow_hidden()
                    .whitespace_nowrap()
                    .text_ellipsis()
                    .font_weight(FontWeight::SEMIBOLD)
                    .child(scene),
            )
            .child(
                choice("pv-class", "Классы", by == cloud::ColorBy::Class).on_click(cx.listener(
                    |this, _, window, cx| {
                        if let Some(p) = this.preview.as_mut() {
                            p.by = cloud::ColorBy::Class;
                        }
                        this.redraw_preview(Some(window), cx);
                    },
                )),
            )
            .child(
                choice("pv-height", "Высота", by == cloud::ColorBy::Height).on_click(cx.listener(
                    |this, _, window, cx| {
                        if let Some(p) = this.preview.as_mut() {
                            p.by = cloud::ColorBy::Height;
                        }
                        this.redraw_preview(Some(window), cx);
                    },
                )),
            )
            .when(
                matches!(
                    pv.pred.as_ref().and_then(|x| x.result.as_ref()),
                    Some(Ok(_))
                ),
                |d| {
                    d.child(
                        choice("pv-errors", "Ошибки", by == cloud::ColorBy::Errors).on_click(
                            cx.listener(|this, _, window, cx| {
                                if let Some(p) = this.preview.as_mut() {
                                    p.by = cloud::ColorBy::Errors;
                                }
                                this.redraw_preview(Some(window), cx);
                            }),
                        ),
                    )
                },
            )
            .child(
                Button::new("pv-reset")
                    .icon(Icon::new(Lucide::RotateCcw))
                    .ghost()
                    .small()
                    .tooltip("Исходный вид")
                    .on_click(cx.listener(|this, _, window, cx| {
                        if let Some(p) = this.preview.as_mut() {
                            p.view = cloud::View::default();
                        }
                        this.redraw_preview(Some(window), cx);
                    })),
            );

        // Строка моделей: лучший trial каждого прогона с best.pt.
        let models = self.models(&pv.scene);
        let current = pv.pred.as_ref().map(|x| x.label.clone());
        let running = self
            .jobs
            .iter()
            .any(|j| j.running() && j.predict.as_ref().is_some_and(|pj| pj.scene == pv.scene));
        let models_row = h_flex()
            .id("pv-models")
            .max_h(px(100.))
            .overflow_y_scrollbar()
            .px_4()
            .py_2()
            .gap_2()
            .flex_wrap()
            .border_b_1()
            .border_color(p.rule)
            .text_xs()
            .child(div().text_color(p.muted).child("Проверить моделью:"))
            .when(models.is_empty(), |d| {
                d.child(
                    div()
                        .text_color(p.muted)
                        .child("нет прогонов с сохранённым best.pt"),
                )
            })
            .children(models.into_iter().enumerate().map(|(k, m)| {
                let label = m.label();
                let text = format!(
                    "{label} · F1 {}",
                    m.f1.map_or("—".into(), |f| format!("{f:.3}"))
                );
                choice(
                    ("pv-model", k),
                    text,
                    current.as_deref() == Some(label.as_str()),
                )
                .disabled(running)
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.predict(
                        Model {
                            run: m.run.clone(),
                            checkpoint: m.checkpoint.clone(),
                            target: m.target.clone(),
                            ..m
                        },
                        cx,
                    );
                }))
            }));

        let body = match (&pv.cloud, &pv.image) {
            (None, _) => div()
                .flex_1()
                .flex()
                .items_center()
                .justify_center()
                .text_color(p.muted)
                .child("Читаю сцену…")
                .into_any_element(),
            (Some(Err(e)), _) => div()
                .flex_1()
                .p_4()
                .text_color(p.loss)
                .child(format!("Не удалось прочитать: {e}"))
                .into_any_element(),
            (Some(Ok(_)), None) => div().flex_1().into_any_element(),
            (Some(Ok(_)), Some(image)) => div()
                .on_children_prepainted({
                    let shown = pv.shown.clone();
                    move |b, _, _| {
                        if let Some(r) = b.first() {
                            // ObjectFit::Contain: кадр вписан в элемент с сохранением пропорций.
                            let k = (f32::from(r.size.width) / PREVIEW_W as f32).min(f32::from(r.size.height) / PREVIEW_H as f32);
                            shown.set((PREVIEW_W as f32 * k, PREVIEW_H as f32 * k));
                        }
                    }
                })
                .id("pv-canvas")
                .flex_1()
                .min_h_0()
                .cursor_grab()
                .child(
                    img(image.clone())
                        .size_full()
                        .object_fit(ObjectFit::Contain),
                )
                // Перетаскивание — вращение, колесо — масштаб.
                // Обработчики только меняют вид; картинка пересчитывается раз за кадр в render().
                .on_mouse_move(cx.listener(|this, e: &MouseMoveEvent, _, cx| {
                    let Some(p) = this.preview.as_mut() else {
                        return;
                    };
                    if !matches!(e.pressed_button, Some(MouseButton::Left) | Some(MouseButton::Middle)) {
                        if p.last.take().is_some() {
                            p.dirty = true; // конец перетаскивания — полное качество
                            cx.notify();
                        }
                        return;
                    }
                    if let Some(last) = p.last {
                        let d = e.position - last;
                        if e.pressed_button == Some(MouseButton::Middle) {
                            let (sw, sh) = p.shown.get();
                            p.view.pan_x += f32::from(d.x) / sw.max(1.);
                            p.view.pan_y += f32::from(d.y) / sh.max(1.);
                        } else {
                            p.view.yaw -= f32::from(d.x) * 0.01;
                            p.view.pitch = (p.view.pitch + f32::from(d.y) * 0.01).clamp(0., 1.55);
                        }
                    }
                    p.last = Some(e.position);
                    p.dirty = true;
                    cx.notify();
                }))
                .on_mouse_up(
                    MouseButton::Left,
                    cx.listener(|this, _, _, cx| {
                        if let Some(p) = this.preview.as_mut().filter(|p| p.last.is_some()) {
                            p.last = None;
                            p.dirty = true;
                            cx.notify();
                        }
                    }),
                )
                .on_mouse_up(
                    MouseButton::Middle,
                    cx.listener(|this, _, _, cx| {
                        if let Some(p) = this.preview.as_mut().filter(|p| p.last.is_some()) {
                            p.last = None;
                            p.dirty = true;
                            cx.notify();
                        }
                    }),
                )
                .on_scroll(cx.listener(|this, e: &ScrollEvent, _, cx| {
                    let Some(p) = this.preview.as_mut() else {
                        return;
                    };
                    let dy = f32::from(e.delta.pixel_delta(px(20.)).y);
                    p.view.zoom = (p.view.zoom * (dy * 0.002).exp()).clamp(0.3, 20.);
                    p.dirty = true;
                    cx.notify();
                }))
                .into_any_element(),
        };

        let pred_line = pv.pred.as_ref().map(|x| {
            let split = match x.split {
                Some("val") => "сцена из val этого прогона — честная оценка",
                Some("train") => "сцена из train — модель её видела, оценка завышена",
                _ => "сцена не из датасета этого прогона",
            };
            match &x.result {
                None => div().text_color(p.muted).child(format!(
                    "{}: считаю предсказание… (лог — в «Трансляции»)",
                    x.label
                )),
                Some(Err(e)) => div().text_color(p.loss).child(format!("{}: {e}", x.label)),
                Some(Ok(o)) => {
                    let [tn, tp, fp, fn_] = o.counts;
                    let all = (tn + tp + fp + fn_).max(1) as f64;
                    let pct = |n: usize| n as f64 * 100. / all;
                    h_flex()
                        .gap_4()
                        .flex_wrap()
                        .child(
                            div()
                                .font_weight(FontWeight::SEMIBOLD)
                                .font_features(tnum())
                                .child(format!("{} · F1 на сцене {:.4}", x.label, o.f1())),
                        )
                        .children(
                            [
                                ("верно найдено", 0x2EAD5B, tp),
                                ("ложная тревога", 0xE8452C, fp),
                                ("пропуск", 0xF2C200, fn_),
                                ("верно отброшено", 0x5B6068, tn),
                            ]
                            .map(|(name, hex, n)| {
                                h_flex()
                                    .gap_1()
                                    .child(div().size(px(10.)).rounded_full().bg(rgb(hex)))
                                    .child(format!("{name} {:.1}%", pct(n)))
                            }),
                        )
                        .child(div().text_color(p.muted).child(split))
                        .when(o.uncovered > 0, |d| {
                            d.child(div().text_color(p.loss).child(format!(
                                "не покрыто {} точек",
                                thousands(o.uncovered as u64)
                            )))
                        })
                }
            }
        });

        let legend = match &pv.cloud {
            Some(Ok(c)) => {
                let shown: usize = c.classes.iter().map(|k| k.1).sum::<usize>().max(1);
                v_flex()
                    .px_4()
                    .py_2()
                    .gap_1()
                    .border_t_1()
                    .border_color(p.rule)
                    .text_xs()
                    .when_some(pred_line, |d, l| d.child(l))
                    .child(h_flex().gap_4().flex_wrap().children(c.classes.iter().take(8).map(|&(k, n)| {
                        let t = cloud::target_of(k);
                        h_flex()
                            .gap_1()
                            .child(div().size(px(10.)).rounded_full().bg(if t.is_empty() { rgb(0x8A8F98).into() } else { class_color(t) }))
                            .child(format!("{} {k} · {:.0}%", if t.is_empty() { "класс" } else { class_ru(t) }, n as f64 * 100. / shown as f64))
                    })))
                    .child(div().text_color(p.muted).font_features(tnum()).whitespace_nowrap().overflow_hidden().text_ellipsis().child(format!(
                        "{} из {} точек · {:.0}×{:.0}×{:.0} м · левая кнопка — вращать, средняя — двигать, колесо — масштаб",
                        thousands(c.pts.len() as u64),
                        thousands(c.total as u64),
                        c.extent[0],
                        c.extent[1],
                        c.extent[2]
                    )))
                    .into_any_element()
            }
            _ => div().into_any_element(),
        };
        panel
            .child(toolbar)
            .when(matches!(pv.cloud, Some(Ok(_))), |d| d.child(models_row))
            .child(body)
            .child(legend)
            .into_any_element()
    }
}

impl Studio {
    /// Окружение готово к обучению (пока проверка не прошла — не тревожим).
    fn env_ready(&self) -> bool {
        self.env.as_ref().is_none_or(|e| {
            e.packages
                .as_ref()
                .is_some_and(|p| p.as_ref().is_ok_and(|p| p.missing().is_empty()))
        })
    }

    // ---------- сценарии ----------

    fn reload_scenarios(&mut self) {
        let cur = self
            .scenarios
            .get(self.sel_scenario)
            .map(|s| s.name.clone());
        self.scenarios = scenario::scan(&self.root);
        self.sel_scenario = self
            .scenarios
            .iter()
            .position(|s| Some(&s.name) == cur.as_ref())
            .unwrap_or(0);
    }

    fn new_scenario(&mut self) -> Option<usize> {
        let s = scenario::Scenario {
            name: scenario::new_name(&self.root),
            steps: vec![],
        };
        if let Err(e) = s.save(&self.root) {
            self.scenario_note = Some((false, format!("Не удалось создать сценарий: {e}")));
            return None;
        }
        self.scenarios.push(s);
        self.sel_scenario = self.scenarios.len() - 1;
        Some(self.sel_scenario)
    }

    /// Шаг из текущей формы «Запуска» — в выбранный сценарий (или в новый, если сценариев нет).
    fn add_step(&mut self, step: scenario::Step) {
        let idx = match self.scenarios.get(self.sel_scenario) {
            Some(_) => self.sel_scenario,
            None => match self.new_scenario() {
                Some(i) => i,
                None => return,
            },
        };
        let s = &mut self.scenarios[idx];
        s.steps.push(step);
        if let Err(e) = s.save(&self.root) {
            s.steps.pop();
            self.scenario_note = Some((false, format!("Не удалось сохранить «{}»: {e}", s.name)));
        }
    }

    /// Изменить шаги выбранного сценария и сохранить файл.
    fn edit_steps(&mut self, f: impl FnOnce(&mut Vec<scenario::Step>)) {
        let Some(s) = self.scenarios.get_mut(self.sel_scenario) else {
            return;
        };
        f(&mut s.steps);
        if let Err(e) = s.save(&self.root) {
            self.scenario_note = Some((false, format!("Не удалось сохранить «{}»: {e}", s.name)));
        }
    }

    fn rename_scenario(&mut self, to: String) {
        let Some(s) = self
            .scenarios
            .iter_mut()
            .find(|s| Some(&s.name) == self.scenario_name_for.as_ref())
        else {
            return;
        };
        let old = s.name.clone();
        match s.rename(&self.root, &to) {
            Ok(()) => {
                self.scenario_note = None;
                self.scenario_name_for = Some(s.name.clone());
                if let Some(a) = self.active.as_mut().filter(|a| a.name == old) {
                    a.name = s.name.clone();
                }
            }
            Err(e) => self.scenario_note = Some((false, e)),
        }
    }

    fn start_scenario(&mut self, from: usize, cx: &mut Context<Self>) {
        let Some(s) = self.scenarios.get(self.sel_scenario) else {
            return;
        };
        if self.active.is_some() || from >= s.steps.len() {
            return;
        }
        self.active = Some(Active {
            name: s.name.clone(),
            steps: s.steps.clone(),
            next: from,
            from,
            job: None,
        });
        self.scenario_note = None;
        self.advance_scenario(cx);
    }

    /// Шаг закончился — запустить следующий; ошибка или ручная остановка обрывают сценарий, как в run_all.py.
    fn advance_scenario(&mut self, cx: &mut Context<Self>) {
        let Some(mut a) = self.active.take() else {
            return;
        };
        let mut fail = |this: &mut Self, msg: String| {
            this.scenario_note = Some((false, msg));
            cx.notify();
        };
        if let Some(j) = a.job {
            let job = &self.jobs[j];
            if job.running() {
                self.active = Some(a);
                return;
            }
            let ok = !job.stopped && matches!(job.out.lock().unwrap().exit, Some(Ok(())));
            if !ok {
                let why = if job.stopped {
                    "процесс остановлен вручную"
                } else {
                    "шаг завершился с ошибкой, лог в «Трансляции»"
                };
                return fail(
                    self,
                    format!(
                        "«{}» прерван на шаге {}: {why}. Продолжить можно кнопкой «Отсюда» у этого шага.",
                        a.name, a.next
                    ),
                );
            }
        }
        if a.next >= a.steps.len() {
            let (summary, body) = (
                format!("Сценарий «{}» выполнен", a.name),
                format!("Шагов: {}", a.steps.len() - a.from),
            );
            self.scenario_note = Some((true, format!("{summary}. {body}.")));
            std::thread::spawn(move || {
                _ = notify_rust::Notification::new()
                    .appname("ML Studio")
                    .summary(&summary)
                    .body(&body)
                    .show()
            });
            cx.notify();
            return;
        }
        let step = a.steps[a.next].clone();
        if let Some(c) = self.conflict(Job::arg(&step.args, "--output-dir").as_deref()) {
            return fail(self, format!("«{}», шаг {}: {c}", a.name, a.next + 1));
        }
        if !step.fixed_params.is_null()
            && let Some(f) = Job::arg(&step.args, "--fixed-params")
            && let Err(e) = std::fs::write(
                self.root.join(&f),
                serde_json::to_vec_pretty(&step.fixed_params).unwrap(),
            )
        {
            return fail(
                self,
                format!(
                    "«{}», шаг {}: не удалось записать {f}: {e}",
                    a.name,
                    a.next + 1
                ),
            );
        }
        let python = self.python.read(cx).value().trim().to_string();
        let tab = self.tab;
        self.spawn(python, step.args, step.title);
        // Первый шаг открывает «Трансляцию», следующие не выдёргивают человека с текущей вкладки.
        if a.job.is_some() {
            self.tab = tab;
        }
        a.job = Some(self.jobs.len() - 1);
        a.next += 1;
        self.active = Some(a);
        cx.notify();
    }

    fn stop_scenario(&mut self) {
        if let Some(j) = self.active.as_ref().and_then(|a| a.job) {
            let job = &mut self.jobs[j];
            if job.running() {
                job.stopped = true;
                kill(job);
            }
        }
    }

    /// Новый шаг подбора: копия последнего шага hpo (обычно меняют только класс), иначе значения по умолчанию.
    fn add_hpo_step(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let last = self.scenarios.get(self.sel_scenario).and_then(|s| {
            s.steps
                .iter()
                .rev()
                .find(|st| st.args.iter().any(|a| a == "hpo_pointnet.py"))
                .cloned()
        });
        let step = last.unwrap_or_else(|| {
            let data = self
                .datasets
                .iter()
                .find(|d| d.name == "exports")
                .or(self.datasets.first())
                .map_or("exports".into(), |d| d.name.clone());
            let args = [
                "-u",
                "hpo_pointnet.py",
                "--data-dir",
                &data,
                "--output-dir",
                "runs/ground_xyz",
                "--target",
                "ground",
                "--features",
                "xyz",
            ]
            .into_iter()
            .chain([
                "--n-trials",
                "25",
                "--epochs",
                "10",
                "--seed",
                "42",
                "--workers",
                "0",
            ])
            .map(String::from)
            .collect();
            scenario::Step {
                title: "ground_xyz".into(),
                args,
                fixed_params: serde_json::Value::Null,
            }
        });
        let mut step = step;
        // Копия последнего шага писала бы в тот же runs/<имя>: берём первый класс, которого в сценарии ещё нет.
        let used: Vec<String> = self.scenarios.get(self.sel_scenario).map_or(vec![], |s| {
            s.steps
                .iter()
                .filter_map(|st| Job::arg(&st.args, "--output-dir"))
                .collect()
        });
        let free = TARGETS.iter().find(|t| {
            let mut a = step.args.clone();
            set_arg(&mut a, "--target", t.to_string());
            !used.contains(&format!("runs/{}", auto_name(&a)))
        });
        if let Some(t) = free {
            set_arg(&mut step.args, "--target", t.to_string());
            let name = auto_name(&step.args);
            set_arg(&mut step.args, "--output-dir", format!("runs/{name}"));
            if step.args.iter().any(|a| a == "--fixed-params") {
                set_arg(
                    &mut step.args,
                    "--fixed-params",
                    format!("runs/{name}.params.json"),
                );
            }
            step.title = name;
        }
        self.add_step(step);
        if let Some(s) = self.scenarios.get(self.sel_scenario) {
            self.open_step_editor(s.steps.len() - 1, window, cx);
        }
    }

    /// Шаг №7: сводный отчёт по всем прогонам, как в run_all.py.
    fn add_report_step(&mut self, cx: &App) {
        let runs = self.runs_dir.read(cx).value().trim().to_string();
        let args = vec![
            "-u".to_string(),
            "summary_report.py".into(),
            runs,
            "--out".into(),
            cli_path(&self.runs_dir.read(cx).value().trim(), "report.md"),
        ];
        self.add_step(scenario::Step {
            title: "Сводный отчёт".into(),
            args,
            fixed_params: serde_json::Value::Null,
        });
    }

    fn open_step_editor(&mut self, i: usize, window: &mut Window, cx: &mut Context<Self>) {
        let Some(st) = self
            .scenarios
            .get(self.sel_scenario)
            .and_then(|s| s.steps.get(i))
        else {
            return;
        };
        let val = |k: &str| Job::arg(&st.args, k).unwrap_or_default();
        let name = Job::arg(&st.args, "--output-dir")
            .map(|d| d.trim_start_matches("runs/").to_string())
            .unwrap_or_default();
        let gamma = Some(val("--focal-gamma"))
            .filter(|g| !g.is_empty())
            .unwrap_or_else(|| "2".into());
        let fields = [
            ("--output-dir", name),
            ("--n-trials", val("--n-trials")),
            ("--epochs", val("--epochs")),
            ("--seed", val("--seed")),
            ("--focal-gamma", gamma),
        ];
        let mut inputs = vec![];
        let mut subs = vec![];
        for (key, v) in fields {
            let input = cx.new(|cx| InputState::new(window, cx).default_value(v));
            subs.push(cx.subscribe_in(
                &input,
                window,
                move |this, input, ev: &gpui_kit::component::input::InputEvent, window, cx| {
                    if !matches!(ev, gpui_kit::component::input::InputEvent::Change) {
                        return;
                    }
                    let v = input.read(cx).value().trim().to_string();
                    if key == "--focal-gamma" {
                        return this.set_step_choice(key, v, window, cx);
                    }
                    let Some(i) = this.step_edit.as_ref().map(|e| e.idx) else {
                        return;
                    };
                    this.edit_steps(|steps| {
                        let Some(st) = steps.get_mut(i) else { return };
                        if key == "--output-dir" {
                            set_arg(&mut st.args, key, format!("runs/{v}"));
                            if st.args.iter().any(|a| a == "--fixed-params") {
                                set_arg(
                                    &mut st.args,
                                    "--fixed-params",
                                    format!("runs/{v}.params.json"),
                                );
                            }
                            st.title = v;
                        } else {
                            set_arg(&mut st.args, key, v);
                        }
                    });
                    cx.notify();
                },
            ));
            inputs.push(input);
        }
        let fixed = self.scenarios[self.sel_scenario].steps[i]
            .fixed_params
            .as_object()
            .cloned()
            .unwrap_or_default();
        let mut params = vec![];
        for (k, spec) in SPACE.iter().enumerate() {
            if matches!(spec.kind, Kind::Choice(_)) {
                params.push(None);
                continue;
            }
            let v = fixed
                .get(spec.key)
                .cloned()
                .unwrap_or_else(|| spec.middle());
            let input = cx.new(|cx| InputState::new(window, cx).default_value(spec.show(&v)));
            subs.push(cx.subscribe(
                &input,
                move |this, input, ev: &gpui_kit::component::input::InputEvent, cx| {
                    if !matches!(ev, gpui_kit::component::input::InputEvent::Change) {
                        return;
                    }
                    // Неверное значение не пишется: в файле остаётся последнее верное, ошибка видна под полем.
                    if let Ok(v) = SPACE[k].parse(&input.read(cx).value()) {
                        this.edit_fixed(|m| _ = m.insert(SPACE[k].key.into(), v));
                    }
                    cx.notify();
                },
            ));
            params.push(Some(input));
        }
        self.step_edit = Some(StepEditor {
            idx: i,
            inputs,
            params,
            _subs: subs,
        });
    }

    /// Изменить свою конфигурацию открытого шага (в режиме Optuna её нет — ничего не делает).
    fn edit_fixed(&mut self, f: impl FnOnce(&mut Params)) {
        let Some(i) = self.step_edit.as_ref().map(|e| e.idx) else {
            return;
        };
        self.edit_steps(|steps| {
            if let Some(m) = steps
                .get_mut(i)
                .and_then(|st| st.fixed_params.as_object_mut())
            {
                f(m);
            }
        });
    }

    /// Optuna ↔ своя конфигурация. Своя: файл runs/<имя>.params.json и 1 trial — обучить ровно её.
    fn set_step_custom(&mut self, on: bool, window: &mut Window, cx: &mut Context<Self>) {
        let Some(e) = self.step_edit.as_ref() else {
            return;
        };
        let (i, trials) = (e.idx, e.inputs[1].clone());
        let params: Params = SPACE
            .iter()
            .zip(&e.params)
            .map(|(spec, input)| {
                (
                    spec.key.to_string(),
                    input
                        .as_ref()
                        .and_then(|inp| spec.parse(&inp.read(cx).value()).ok())
                        .unwrap_or_else(|| spec.middle()),
                )
            })
            .collect();
        self.edit_steps(|steps| {
            let Some(st) = steps.get_mut(i) else { return };
            if on {
                let dir = Job::arg(&st.args, "--output-dir").unwrap_or_default();
                set_arg(&mut st.args, "--fixed-params", format!("{dir}.params.json"));
                st.fixed_params = serde_json::Value::Object(params);
            } else {
                st.fixed_params = serde_json::Value::Null;
                if let Some(k) = st.args.iter().position(|a| a == "--fixed-params") {
                    st.args.drain(k..(k + 2).min(st.args.len()));
                }
            }
        });
        Self::set_input(&trials, if on { "1" } else { "25" }.into(), window, cx);
        cx.notify();
    }

    /// Конфигурация лучшего trial прогона — в редактор шага.
    fn step_params_from(&mut self, params: Params, window: &mut Window, cx: &mut Context<Self>) {
        let Some(inputs) = self.step_edit.as_ref().map(|e| e.params.clone()) else {
            return;
        };
        self.edit_fixed(|m| {
            for spec in &SPACE {
                if let Some(v) = params.get(spec.key) {
                    m.insert(spec.key.into(), v.clone());
                }
            }
        });
        for (spec, input) in SPACE.iter().zip(inputs) {
            if let (Some(input), Some(v)) = (input, params.get(spec.key)) {
                Self::set_input(&input, spec.show(v), window, cx);
            }
        }
        cx.notify();
    }

    /// Класс или признаки: имя вида <класс>_<признаки> следует за ними, своё имя не трогаем.
    fn set_step_choice(
        &mut self,
        key: &'static str,
        v: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(e) = self.step_edit.as_ref() else {
            return;
        };
        let (i, name_input) = (e.idx, e.inputs[0].clone());
        let gamma = e.inputs[4].read(cx).value().trim().to_string();
        let mut rename = None;
        self.edit_steps(|steps| {
            let Some(st) = steps.get_mut(i) else { return };
            let auto = auto_name(&st.args);
            if key == "--loss" && v == "focal" {
                set_arg(&mut st.args, "--focal-gamma", gamma);
            }
            set_arg(&mut st.args, key, v);
            let fresh = auto_name(&st.args);
            if st.title == auto && key != "--data-dir" {
                rename = Some(fresh);
            }
        });
        // Change от поля сам перепишет --output-dir и название шага.
        if let Some(n) = rename {
            Self::set_input(&name_input, n, window, cx);
        }
        cx.notify();
    }

    fn render_step_editor(
        &self,
        st: &scenario::Step,
        p: Pal,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let Some(e) = self.step_edit.as_ref() else {
            return div().into_any_element();
        };
        let target = Job::arg(&st.args, "--target").unwrap_or_default();
        let features = Job::arg(&st.args, "--features").unwrap_or_default();
        let data = Job::arg(&st.args, "--data-dir").unwrap_or_default();
        let focal = Job::arg(&st.args, "--loss").as_deref() == Some("focal");
        let custom = st.fixed_params.as_object();
        let label = |t: &str| {
            div()
                .w(px(110.))
                .flex_shrink_0()
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(p.muted)
                .child(t.to_string())
        };
        let row = |t: &str| h_flex().gap_3().items_center().child(label(t));
        let mut datasets: Vec<String> = self.datasets.iter().map(|d| d.name.clone()).collect();
        if !data.is_empty() && !datasets.contains(&data) {
            datasets.push(data.clone());
        }
        let field = |t: &str, input: &Entity<InputState>| {
            v_flex()
                .gap_1()
                .w(px(150.))
                .child(
                    div()
                        .text_xs()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(p.muted)
                        .child(t.to_string()),
                )
                .child(Input::new(input))
        };
        v_flex()
            .mx_8()
            .mb_3()
            .p_4()
            .gap_3()
            .rounded_xl()
            .bg(p.surface)
            .border_1()
            .border_color(p.rule)
            .child(row("Класс").child(h_flex().gap_1().flex_wrap().children(TARGETS.iter().map(|&o| {
                choice(SharedString::from(format!("se-t-{o}")), format!("{} · {o}", class_ru(o)), target == o)
                    .on_click(cx.listener(move |this, _, window, cx| this.set_step_choice("--target", o.into(), window, cx)))
            }))))
            .child(row("Признаки").child(h_flex().gap_1().children(FEATURES.iter().map(|&o| {
                choice(SharedString::from(format!("se-f-{o}")), o, features == o).on_click(cx.listener(move |this, _, window, cx| this.set_step_choice("--features", o.into(), window, cx)))
            }))))
            .child(row("Данные").child(h_flex().gap_1().flex_wrap().children(datasets.into_iter().map(|d| {
                let sel = d == data;
                choice(SharedString::from(format!("se-d-{d}")), d.clone(), sel).on_click(cx.listener(move |this, _, window, cx| this.set_step_choice("--data-dir", d.clone(), window, cx)))
            }))))
            .child(
                h_flex()
                    .gap_3()
                    .flex_wrap()
                    .child(field("Имя прогона", &e.inputs[0]))
                    .child(field("Trials", &e.inputs[1]))
                    .child(field("Эпох", &e.inputs[2]))
                    .child(field("Seed", &e.inputs[3])),
            )
            .child(row("Функция потерь").child(
                h_flex()
                    .gap_1()
                    .child(choice("se-ce", "CrossEntropy", !focal).on_click(cx.listener(|this, _, window, cx| this.set_step_choice("--loss", "ce".into(), window, cx))))
                    .child(choice("se-focal", "Focal Loss", focal).on_click(cx.listener(|this, _, window, cx| this.set_step_choice("--loss", "focal".into(), window, cx))))
                    .when(focal, |d| d.child(div().w(px(80.)).child(Input::new(&e.inputs[4])))),
            ))
            .child(row("Гиперпараметры").child(
                h_flex()
                    .gap_1()
                    .child(choice("se-optuna", "Подбирает Optuna", custom.is_none()).on_click(cx.listener(|this, _, window, cx| this.set_step_custom(false, window, cx))))
                    .child(choice("se-custom", "Своя конфигурация", custom.is_some()).on_click(cx.listener(|this, _, window, cx| this.set_step_custom(true, window, cx)))),
            ))
            .when_some(custom, |d, fixed| {
                let mut best_runs: Vec<(String, f64, Params)> = self
                    .runs
                    .iter()
                    .filter(|r| r.target().as_deref() == Some(target.as_str()))
                    .filter_map(|r| {
                        let b = r.best.as_ref()?;
                        Some((r.name.clone(), b.get("f1")?.as_f64()?, b.get("params")?.as_object()?.clone()))
                    })
                    .collect();
                best_runs.sort_by(|a, b| b.1.total_cmp(&a.1));
                best_runs.truncate(3);
                let cells = SPACE.iter().enumerate().map(|(k, spec)| {
                    let input = e.params[k].as_ref();
                    let err = input.and_then(|inp| spec.parse(&inp.read(cx).value()).err());
                    let control = match (input, &spec.kind) {
                        (Some(inp), _) => Input::new(inp).into_any_element(),
                        (None, Kind::Choice(opts)) => h_flex()
                            .gap_1()
                            .children(opts.iter().map(|&o| {
                                let v: serde_json::Value = serde_json::from_str(o).unwrap();
                                let label = match o {
                                    "true" => "да",
                                    "false" => "нет",
                                    o => o,
                                };
                                let sel = fixed.get(spec.key) == Some(&v);
                                choice(SharedString::from(format!("se-hp-{}-{o}", spec.key)), label, sel).on_click(cx.listener(move |this, _, _, cx| {
                                    this.edit_fixed(|m| _ = m.insert(SPACE[k].key.into(), v.clone()));
                                    cx.notify();
                                }))
                            }))
                            .into_any_element(),
                        _ => div().into_any_element(),
                    };
                    v_flex()
                        .w(px(230.))
                        .gap_1()
                        .child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(p.muted).child(spec.label))
                        .child(control)
                        .child(match err {
                            Some(e) => div().text_xs().text_color(p.loss).child(e),
                            None => div().text_xs().text_color(p.muted).child(spec.range()),
                        })
                });
                let out_dir = Job::arg(&st.args, "--output-dir").unwrap_or_default();
                d.when(!best_runs.is_empty(), |d| {
                    d.child(row("Взять из").child(h_flex().gap_1().flex_wrap().children(best_runs.into_iter().enumerate().map(|(k, (name, f1, params))| {
                        Button::new(SharedString::from(format!("se-from-{k}")))
                            .label(format!("{name} · F1 {f1:.4}"))
                            .outline()
                            .xsmall()
                            .on_click(cx.listener(move |this, _, window, cx| this.step_params_from(params.clone(), window, cx)))
                    }))))
                })
                .child(h_flex().gap_4().flex_wrap().children(cells))
                .child(div().text_xs().text_color(p.muted).child("Trials = 1 — обучить ровно эту конфигурацию; больше — она станет первой точкой подбора Optuna."))
                .when(self.root.join(&out_dir).join("study.db").exists(), |d| {
                    d.child(div().text_xs().font_weight(FontWeight::SEMIBOLD).text_color(p.loss).child(format!(
                        "В {out_dir} уже есть study.db — своя конфигурация применяется только к новому study. Задайте другое имя прогона."
                    )))
                })
            })
            .into_any_element()
    }

    fn render_scenarios(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let active_name = self.active.as_ref().map(|a| a.name.clone());
        let list = v_flex()
            .id("scn-list")
            .w(px(300.))
            .h_full()
            .flex_shrink_0()
            .border_r_1()
            .border_color(p.rule)
            .overflow_y_scrollbar()
            .child(
                h_flex()
                    .px_4()
                    .pt_4()
                    .pb_2()
                    .child(display("Сценарии", 22., p.ink))
                    .child(div().flex_1())
                    .child(Button::new("scn-new").icon(Icon::new(Lucide::Plus)).ghost().xsmall().tooltip("Новый сценарий").on_click(cx.listener(|this, _, _, cx| {
                        this.new_scenario();
                        cx.notify();
                    }))),
            )
            .children(self.scenarios.iter().enumerate().map(|(i, s)| {
                let live = active_name.as_ref() == Some(&s.name);
                h_flex()
                    .id(("scn", i))
                    .mx_2()
                    .px_3()
                    .py_2()
                    .gap_3()
                    .rounded_lg()
                    .cursor_pointer()
                    .hover(|d| d.bg(p.raised))
                    .when(i == self.sel_scenario, |d| d.bg(p.raised))
                    .child(Icon::new(Lucide::ListChecks).size_4().text_color(if live { p.live_ink } else { p.muted }))
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .child(div().font_weight(FontWeight::SEMIBOLD).text_color(p.ink).overflow_hidden().whitespace_nowrap().text_ellipsis().child(s.name.clone()))
                            .child(div().text_xs().text_color(if live { p.live_ink } else { p.muted }).child(match self.active.as_ref().filter(|_| live) {
                                Some(a) => format!("идёт · шаг {} из {}", a.next, a.steps.len()),
                                None => format!("шагов: {}", s.steps.len()),
                            })),
                    )
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.sel_scenario = i;
                        this.scenario_note = None;
                        this.step_edit = None;
                        cx.notify();
                    }))
            }))
            .when(self.scenarios.is_empty(), |d| {
                d.child(div().px_4().py_2().text_color(p.muted).child("Сценариев пока нет. Нажмите «+» или добавьте шаг кнопкой «В сценарий» на вкладке «Запуск»."))
            });

        let note = self.scenario_note.clone().map(|(ok, n)| {
            div()
                .px_8()
                .py_3()
                .border_b_1()
                .border_color(p.rule)
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(if ok { p.win } else { p.loss })
                .child(n)
        });
        let Some(s) = self.scenarios.get(self.sel_scenario) else {
            return h_flex()
                .size_full()
                .child(list)
                .child(v_flex().flex_1().children(note))
                .into_any_element();
        };
        let this_live = active_name.as_ref() == Some(&s.name);
        let busy = self.active.is_some();
        let n = s.steps.len();
        let head = h_flex()
            .px_8()
            .pt_6()
            .pb_4()
            .gap_3()
            .items_end()
            .border_b_1()
            .border_color(p.rule)
            .child(
                v_flex()
                    .flex_1()
                    .gap_1()
                    .child(small_caps("Название · Enter — сохранить", p.muted))
                    .child(Input::new(&self.scenario_name)),
            )
            .child(
                Button::new("scn-add")
                    .icon(Icon::new(Lucide::Plus))
                    .label("Шаг")
                    .outline()
                    .disabled(this_live)
                    .tooltip("Подбор гиперпараметров для класса")
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.add_hpo_step(window, cx);
                        cx.notify();
                    })),
            )
            .child(
                Button::new("scn-report")
                    .icon(Icon::new(Lucide::FileText))
                    .label("Отчёт")
                    .outline()
                    .disabled(this_live)
                    .tooltip("Сводный отчёт по всем прогонам (№7)")
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.add_report_step(cx);
                        cx.notify();
                    })),
            )
            .child(if this_live {
                Button::new("scn-stop")
                    .icon(Icon::new(Lucide::Square))
                    .label("Остановить")
                    .danger()
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.stop_scenario();
                        cx.notify();
                    }))
            } else {
                Button::new("scn-run")
                    .icon(Icon::new(Lucide::Play))
                    .label("Запустить сценарий")
                    .primary()
                    .disabled(busy || n == 0)
                    .tooltip(if busy {
                        "Уже идёт другой сценарий"
                    } else {
                        "Шаги пойдут по очереди; ошибка останавливает сценарий"
                    })
                    .on_click(cx.listener(|this, _, _, cx| this.start_scenario(0, cx)))
            })
            .child(
                Button::new("scn-del")
                    .icon(Icon::new(Lucide::Trash))
                    .outline()
                    .tooltip("Удалить в корзину")
                    .disabled(this_live)
                    .on_click(cx.listener(|this, _, _, cx| {
                        if let Some(s) = this.scenarios.get(this.sel_scenario) {
                            match s.delete(&this.root) {
                                Ok(()) => {
                                    this.scenarios.remove(this.sel_scenario);
                                    this.step_edit = None;
                                    this.sel_scenario = this.sel_scenario.saturating_sub(1);
                                }
                                Err(e) => {
                                    this.scenario_note =
                                        Some((false, format!("Не удалось удалить: {e}")))
                                }
                            }
                        }
                        cx.notify();
                    })),
            );

        let mut steps = v_flex()
            .id("scn-steps")
            .flex_1()
            .overflow_y_scrollbar()
            .pb_8();
        if n == 0 {
            steps = steps.child(div().px_8().py_4().text_color(p.muted).child(
                "Шагов нет. Нажмите «+ Шаг» и выберите класс, признаки и данные. Шаг со своей конфигурацией гиперпараметров добавляется из «Запуска» кнопкой «В сценарий».",
            ));
        }
        for (i, st) in s.steps.iter().enumerate() {
            // Статус шага в идущем сценарии.
            let status = self
                .active
                .as_ref()
                .filter(|_| this_live)
                .and_then(|a| match i {
                    i if i < a.from => None,
                    i if i + 1 < a.next => Some(("ГОТОВО", p.win)),
                    i if i + 1 == a.next => Some(("ИДЁТ", p.live)),
                    _ => Some(("ЖДЁТ", p.muted)),
                });
            let target = Job::arg(&st.args, "--target").unwrap_or_default();
            let edit = |id: &str, icon: Lucide, tip: &'static str, disabled: bool| {
                Button::new(SharedString::from(format!("{id}-{i}")))
                    .icon(Icon::new(icon))
                    .ghost()
                    .xsmall()
                    .tooltip(tip)
                    .disabled(disabled || this_live)
            };
            steps = steps.child(
                h_flex()
                    .px_8()
                    .py_3()
                    .gap_4()
                    .items_start()
                    .border_b_1()
                    .border_color(p.rule)
                    .child(div().w(px(28.)).flex_shrink_0().child(display(&(i + 1).to_string(), 28., p.muted)))
                    .child(
                        v_flex()
                            .flex_1()
                            .min_w_0()
                            .gap_1()
                            .child(
                                h_flex()
                                    .gap_3()
                                    .when(!target.is_empty(), |d| d.child(div().size(px(10.)).rounded_full().bg(class_color(&target))))
                                    .child(div().font_weight(FontWeight::SEMIBOLD).text_color(p.ink).child(st.title.clone()))
                                    .when(!st.fixed_params.is_null(), |d| d.child(div().text_xs().text_color(p.muted).child("своя конфигурация")))
                                    .when_some(status, |d, (label, bg)| d.child(chip(label, bg, rgb(0xFFFFFF).into(), label == "ИДЁТ"))),
                            )
                            .child(div().font_family(MONO).text_xs().text_color(p.muted).child(format!("python {}", st.args.join(" ")))),
                    )
                    .child(
                        h_flex()
                            .gap_1()
                            .flex_shrink_0()
                            .when(st.args.iter().any(|a| a == "hpo_pointnet.py"), |d| {
                                let open = self.step_edit.as_ref().is_some_and(|e| e.idx == i);
                                d.child(edit("scn-edit", Lucide::Pencil, if open { "Свернуть" } else { "Изменить" }, false).when(open, |b| b.selected(true)).on_click(cx.listener(
                                    move |this, _, window, cx| {
                                        if this.step_edit.as_ref().is_some_and(|e| e.idx == i) {
                                            this.step_edit = None;
                                        } else {
                                            this.open_step_editor(i, window, cx);
                                        }
                                        cx.notify();
                                    },
                                )))
                            })
                            .child(edit("scn-up", Lucide::ArrowUp, "Выше", i == 0).on_click(cx.listener(move |this, _, _, cx| {
                                this.step_edit = None;
                                this.edit_steps(|s| s.swap(i - 1, i));
                                cx.notify();
                            })))
                            .child(edit("scn-down", Lucide::ArrowDown, "Ниже", i + 1 == n).on_click(cx.listener(move |this, _, _, cx| {
                                this.step_edit = None;
                                this.edit_steps(|s| s.swap(i, i + 1));
                                cx.notify();
                            })))
                            .child(edit("scn-rm", Lucide::X, "Убрать шаг", false).on_click(cx.listener(move |this, _, _, cx| {
                                this.step_edit = None;
                                this.edit_steps(|s| _ = s.remove(i));
                                cx.notify();
                            })))
                            .child(
                                Button::new(SharedString::from(format!("scn-from-{i}")))
                                    .label("Отсюда")
                                    .outline()
                                    .xsmall()
                                    .tooltip("Запустить сценарий с этого шага (прогон с study.db продолжится)")
                                    .disabled(busy)
                                    .on_click(cx.listener(move |this, _, _, cx| this.start_scenario(i, cx))),
                            ),
                    ),
            );
            if !this_live && self.step_edit.as_ref().is_some_and(|e| e.idx == i) {
                steps = steps.child(self.render_step_editor(st, p, cx));
            }
        }
        h_flex()
            .size_full()
            .child(list)
            .child(
                v_flex()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .child(head)
                    .children(note)
                    .child(steps),
            )
            .into_any_element()
    }

    // ---------- монитор ресурсов ----------

    fn render_usage(&mut self, p: Pal, cx: &mut Context<Self>) -> Div {
        let u = &self.usage;
        let gb = |b: f64| format!("{:.1}", b / 1024f64.powi(3));
        let meter = |label: String, frac: f64, value: String| {
            let frac = frac.clamp(0., 1.);
            v_flex()
                .px_5()
                .gap_1()
                .child(
                    h_flex()
                        .gap_2()
                        .text_xs()
                        .child(
                            div()
                                .flex_1()
                                .min_w_0()
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_ellipsis()
                                .opacity(0.72)
                                .child(label),
                        )
                        .child(div().flex_shrink_0().font_features(tnum()).child(value)),
                )
                .child(
                    div()
                        .h(px(4.))
                        .w_full()
                        .rounded_full()
                        .bg(p.rail_sel)
                        // Под нагрузкой ≥ 90 % полоса загорается: упор в ресурс виден издалека.
                        .child(div().h_full().rounded_full().w(relative(frac as f32)).bg(
                            if frac >= 0.9 {
                                p.live
                            } else {
                                p.rail_ink.opacity(0.55)
                            },
                        )),
                )
        };
        let collapsed = self.resources.collapsed();
        let mut block = v_flex()
            .pb_2()
            .gap_2()
            .text_color(p.rail_ink)
            .child(
                h_flex()
                    .id("resources-toggle")
                    .h(px(28.))
                    .w_full()
                    .px_5()
                    .items_center()
                    .justify_between()
                    .cursor_pointer()
                    .hover(|d| d.bg(p.raised))
                    .child(
                        div()
                            .font_family(BODY)
                            .font_weight(FontWeight::NORMAL)
                            .text_size(px(12.))
                            .child("РЕСУРСЫ"),
                    )
                    .child(
                        Icon::new(if collapsed {
                            Lucide::ChevronRight
                        } else {
                            Lucide::ChevronDown
                        })
                        .size_4()
                        .text_color(p.rail_ink)
                        .opacity(0.7),
                    )
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.resources.toggle();
                        cx.notify();
                    })),
            );
        if !collapsed {
            block = block
                .child(meter(
                    "ЦП".into(),
                    u.cpu as f64 / 100.,
                    format!("{:.0} %", u.cpu),
                ))
                .child(meter(
                    "ОЗУ".into(),
                    if u.ram_total > 0 {
                        u.ram_used as f64 / u.ram_total as f64
                    } else {
                        0.
                    },
                    format!("{} / {} ГБ", gb(u.ram_used as f64), gb(u.ram_total as f64)),
                ));
            for g in &u.gpus {
                let temp = g.temp.map_or(String::new(), |t| format!(" · {t:.0} °C"));
                block = block
                    .child(meter(
                        g.name.clone(),
                        g.util as f64 / 100.,
                        format!("{:.0} %{temp}", g.util),
                    ))
                    .child(meter(
                        (if g.shared { "Общая память ГП" } else { "Видеопамять" }).into(),
                        if g.mem_total_mb > 0. {
                            g.mem_used_mb / g.mem_total_mb
                        } else {
                            0.
                        },
                        format!(
                            "{} / {} ГБ",
                            gb(g.mem_used_mb * 1024. * 1024.),
                            gb(g.mem_total_mb * 1024. * 1024.)
                        ),
                    ));
            }
            if u.seq > 0 && u.gpus.is_empty() {
                block = block.child(div().px_5().text_xs().opacity(0.6).child("ГП: нет данных"));
            }
        }
        block
    }

    // ---------- проекты ----------

    /// Переоткрывает студию на другом корне: всё состояние вкладок пересоздаётся.
    fn open_project(&mut self, root: PathBuf, window: &mut Window, cx: &mut Context<Self>) {
        if !projects::is_project(&root) {
            self.projects_note = Some(format!(
                "В {} нет hpo_pointnet.py — это не папка проекта.",
                show_path(&root)
            ));
            cx.notify();
            return;
        }
        // Процессы привязаны к текущему корню; бросить их — оставить сиротами.
        if self.jobs.iter().any(|j| j.running()) {
            self.projects_note = Some("Сначала остановите запущенные процессы.".into());
            cx.notify();
            return;
        }
        let dark = self.dark;
        *self = Self::new(root, window, cx);
        self.tab = Tab::Live;
        self.set_dark(dark, window, cx);
    }

    /// Копия модели в выбранный файл; ошибка — в шапке вкладки прогонов.
    fn save_model(
        &mut self,
        src: PathBuf,
        name: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let rx = cx.prompt_for_new_path(
            src.parent().unwrap_or(&self.root),
            Some(&format!("{name}.pt")),
        );
        cx.spawn_in(window, async move |this, cx| {
            if let Ok(Ok(Some(dst))) = rx.await {
                let err = std::fs::copy(&src, &dst)
                    .err()
                    .map(|e| format!("Не удалось сохранить модель в {}: {e}", show_path(&dst)));
                _ = this.update_in(cx, |this, _, cx| {
                    this.tab_error = err;
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn pick_project(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("Открыть проект".into()),
        });
        cx.spawn_in(window, async move |this, cx| {
            if let Ok(Ok(Some(mut paths))) = rx.await
                && let Some(p) = paths.pop()
            {
                _ = this.update_in(cx, |this, window, cx| this.open_project(p, window, cx));
            }
        })
        .detach();
    }

    fn render_projects(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let busy = self.jobs.iter().any(|j| j.running());
        let head = h_flex()
            .px_8()
            .pt_6()
            .pb_4()
            .gap_4()
            .border_b_1()
            .border_color(p.rule)
            .child(
                v_flex()
                    .flex_1()
                    .child(display("Проекты", 40., p.ink))
                    .child(div().text_color(p.muted).child("Папка проекта — та, где лежит hpo_pointnet.py. Последний открытый откроется при следующем запуске.")),
            )
            .child(
                Button::new("proj-open")
                    .icon(Icon::new(Lucide::FolderOpen))
                    .label("Открыть папку…")
                    .primary()
                    .on_click(cx.listener(|this, _, window, cx| this.pick_project(window, cx))),
            );
        let root = self.root.clone();
        let row = |key: &str, i: usize, path: &PathBuf, fav: bool, cx: &mut Context<Self>| {
            let current = *path == root;
            let ok = projects::is_project(path);
            let name = path
                .file_name()
                .map_or_else(|| show_path(path), |n| n.to_string_lossy().into_owned());
            let (open, star, forget) = (path.clone(), path.clone(), path.clone());
            h_flex()
                .id(SharedString::from(format!("{key}-{i}")))
                .px_8()
                .py_3()
                .gap_4()
                .border_b_1()
                .border_color(p.rule)
                .when(!current && ok, |d| {
                    d.cursor_pointer().hover(|s| s.bg(p.raised))
                })
                .child(
                    div()
                        .id(SharedString::from(format!("{key}-star-{i}")))
                        .cursor_pointer()
                        .child(Icon::new(Lucide::Star).size_4().text_color(if fav {
                            p.live
                        } else {
                            p.muted
                        }))
                        .tooltip(move |w, cx| {
                            Tooltip::new(if fav {
                                "Убрать из избранного"
                            } else {
                                "В избранное"
                            })
                            .build(w, cx)
                        })
                        .on_click(cx.listener(move |this, _, _, cx| {
                            cx.stop_propagation();
                            this.projects.toggle_favorite(&star);
                            this.projects.save();
                            cx.notify();
                        })),
                )
                .child(
                    v_flex()
                        .flex_1()
                        .min_w_0()
                        .child(
                            h_flex()
                                .gap_3()
                                .child(
                                    div()
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(if ok { p.ink } else { p.muted })
                                        .child(name),
                                )
                                .when(current, |d| {
                                    d.child(chip("ОТКРЫТ", p.win, rgb(0xFFFFFF).into(), false))
                                }),
                        )
                        .child(
                            div()
                                .text_xs()
                                .text_color(p.muted)
                                .overflow_hidden()
                                .whitespace_nowrap()
                                .text_ellipsis()
                                .child(if ok {
                                    show_path(path)
                                } else {
                                    format!(
                                        "{} · папка не найдена или без hpo_pointnet.py",
                                        show_path(path)
                                    )
                                }),
                        ),
                )
                .when(!fav, |d| {
                    d.child(
                        div()
                            .id(SharedString::from(format!("{key}-x-{i}")))
                            .p_2()
                            .rounded_lg()
                            .cursor_pointer()
                            .hover(|s| s.bg(p.raised))
                            .child(Icon::new(Lucide::X).size_4().text_color(p.muted))
                            .tooltip(|w, cx| Tooltip::new("Убрать из недавних").build(w, cx))
                            .on_click(cx.listener(move |this, _, _, cx| {
                                cx.stop_propagation();
                                this.projects.forget(&forget);
                                this.projects.save();
                                cx.notify();
                            })),
                    )
                })
                .when(!current && ok, |d| {
                    d.on_click(cx.listener(move |this, _, window, cx| {
                        this.open_project(open.clone(), window, cx)
                    }))
                })
        };
        let favs = self.projects.favorites.clone();
        let recent: Vec<PathBuf> = self
            .projects
            .recent
            .iter()
            .filter(|r| !self.projects.is_favorite(r))
            .cloned()
            .collect();
        let section = |title: &str| div().px_8().pt_6().pb_2().child(small_caps(title, p.muted));
        let mut list = v_flex().id("proj-list").flex_1().overflow_y_scroll();
        if !favs.is_empty() {
            list = list.child(section("ИЗБРАННЫЕ"));
            for (i, f) in favs.iter().enumerate() {
                list = list.child(row("fav", i, f, true, cx));
            }
        }
        list = list.child(section("НЕДАВНИЕ"));
        if recent.is_empty() {
            list = list.child(
                div()
                    .px_8()
                    .py_2()
                    .text_color(p.muted)
                    .child("Пока пусто — откройте папку проекта."),
            );
        }
        for (i, r) in recent.iter().enumerate() {
            list = list.child(row("rec", i, r, false, cx));
        }
        v_flex()
            .size_full()
            .child(head)
            .when(busy, |d| {
                d.child(
                    div()
                        .px_8()
                        .py_3()
                        .text_color(p.loss)
                        .child("Идут процессы — другой проект можно открыть после их остановки."),
                )
            })
            .when_some(self.projects_note.clone(), |d, n| {
                d.child(div().px_8().py_3().text_color(p.loss).child(n))
            })
            .child(list)
            .into_any_element()
    }

    // ---------- окружение ----------

    fn render_env(&mut self, p: Pal, cx: &mut Context<Self>) -> AnyElement {
        let busy = self.jobs.iter().any(|j| j.setup && j.running());
        let head = h_flex()
            .px_8()
            .pt_6()
            .pb_4()
            .gap_4()
            .border_b_1()
            .border_color(p.rule)
            .child(
                v_flex()
                    .flex_1()
                    .child(display("Окружение Python", 40., p.ink))
                    .child(div().text_color(p.muted).child("То же, что делает run_all.cmd: Python 3.11, .venv311 и torch под вашу видеокарту.")),
            )
            .child(
                Button::new("env-check")
                    .icon(Icon::new(Lucide::RefreshCw))
                    .label(if self.env_checking { "Проверяю…" } else { "Проверить снова" })
                    .outline()
                    .loading(self.env_checking)
                    .on_click(cx.listener(|this, _, _, cx| this.check_env(cx))),
            );
        let Some(info) = &self.env else {
            return v_flex()
                .size_full()
                .child(head)
                .child(
                    div().px_8().py_6().text_color(p.muted).child(
                        "Ищу Python 3.11, видеокарту и пакеты — это занимает несколько секунд.",
                    ),
                )
                .into_any_element();
        };

        // (метка, фон) статуса шага
        let ok = ("ГОТОВО", p.win);
        let bad = ("НЕТ", p.loss);
        let warn = ("ВНИМАНИЕ", p.live);
        let skip = ("НЕ НУЖНО", p.muted);
        let row = |n: &str, status: (&'static str, Hsla), title: &str, detail: String| {
            h_flex()
                .px_8()
                .py_4()
                .gap_4()
                .items_start()
                .border_b_1()
                .border_color(p.rule)
                .child(
                    div()
                        .w(px(28.))
                        .flex_shrink_0()
                        .child(display(n, 28., p.muted)),
                )
                .child(
                    v_flex()
                        .flex_1()
                        .min_w_0()
                        .gap_1()
                        .child(
                            h_flex()
                                .gap_3()
                                .child(
                                    div()
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(p.ink)
                                        .child(title.to_string()),
                                )
                                .child(chip(status.0, status.1, rgb(0xFFFFFF).into(), false)),
                        )
                        .child(div().text_sm().text_color(p.muted).child(detail)),
                )
        };

        // 1. Python 3.11
        let py_status = match (&info.base, &info.venv) {
            (Some(_), _) => ok,
            (None, Some(_)) => skip,
            (None, None) => bad,
        };
        let py_detail = match (&info.base, &info.venv) {
            (Some((path, ver)), _) => format!("Python {ver} · {}", show_path(path)),
            (None, Some(_)) => {
                "Не найден, но .venv311 уже создано — для обучения он не нужен.".into()
            }
            (None, None) => "Не найден. Нужен именно 3.11: для 3.12+ нет колёс pyg-lib.".into(),
        };
        let installer = env::install_python_cmd(info).map(|c| c.0);
        let python_row = row("1", py_status, "Python 3.11", py_detail).when(info.base.is_none() && info.venv.is_none(), |d| {
            d.child(match &installer {
                Some(tool) => Button::new("env-py")
                    .icon(Icon::new(Lucide::ArrowDownToLine))
                    .label(format!("Установить через {tool}"))
                    .primary()
                    .small()
                    .disabled(busy)
                    .on_click(cx.listener(|this, _, _, cx| {
                        this.install_python();
                        cx.notify();
                    }))
                    .into_any_element(),
                None => div()
                    .max_w(px(320.))
                    .text_sm()
                    .text_color(p.ink)
                    .child("Установите Python 3.11 с python.org (с галочкой «py launcher») или uv, затем «Проверить снова».")
                    .into_any_element(),
            })
        });

        // 2. Видеокарта и сборка torch
        let recommended = info.recommended_build();
        let build = self.build.unwrap_or(recommended);
        let gpu_detail = match &info.gpu {
            Some(g) => match g.cuda {
                Some((a, b)) => format!("{} · драйвер поддерживает CUDA до {a}.{b}", g.name),
                None => format!("{} · версию CUDA драйвера определить не удалось", g.name),
            },
            None => "Видеокарта NVIDIA не найдена (нет nvidia-smi) — обучение пойдёт на CPU, примерно вдвое и более медленнее.".into(),
        };
        let gpu_row = row(
            "2",
            if info.gpu.is_some() { ok } else { warn },
            "Видеокарта и сборка torch",
            gpu_detail,
        )
        .child(
            v_flex()
                .items_end()
                .gap_1()
                .child(
                    h_flex()
                        .gap_1()
                        .children(["cpu", "cu126", "cu128"].map(|b| {
                            choice(
                                ElementId::Name(format!("build-{b}").into()),
                                if b == recommended {
                                    format!("{b} ✓")
                                } else {
                                    b.into()
                                },
                                build == b,
                            )
                            .on_click(cx.listener(
                                move |this, _, _, cx| {
                                    this.build = Some(b);
                                    cx.notify();
                                },
                            ))
                        })),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(p.muted)
                        .child(format!("рекомендуется {recommended}")),
                ),
        );

        // 3. .venv311
        let venv_row = row(
            "3",
            if info.venv.is_some() { ok } else { bad },
            "Виртуальное окружение .venv311",
            match &info.venv {
                Some(v) => show_path(v),
                None => "Нет — будет создано из Python 3.11 при установке зависимостей.".into(),
            },
        );

        // 4. Пакеты
        let mismatch = |p: &env::Packages| {
            let torch = p.torch().unwrap_or("");
            (build != "cpu" && !p.cuda_available)
                || (build == "cpu" && torch.contains("+cu"))
                || !torch.ends_with(&format!("+{build}"))
        };
        let (pkg_status, pkg_detail) = match &info.packages {
            None => (bad, "Появятся после создания .venv311.".to_string()),
            Some(Err(e)) => (bad, format!("Проверка не удалась: {e}")),
            Some(Ok(pk)) if !pk.missing().is_empty() => {
                (bad, format!("Не хватает: {}.", pk.missing().join(", ")))
            }
            Some(Ok(pk)) => (
                if mismatch(pk) { warn } else { ok },
                format!(
                    "{} · CUDA {}{}",
                    pk.versions
                        .iter()
                        .map(|(m, v)| format!("{m} {}", v.as_deref().unwrap_or("—")))
                        .collect::<Vec<_>>()
                        .join(", "),
                    if pk.cuda_available {
                        "доступна"
                    } else {
                        "недоступна"
                    },
                    if mismatch(pk) {
                        format!(" · установлена не та сборка — нужна +{build}")
                    } else {
                        String::new()
                    }
                ),
            ),
        };
        let can_install = !busy && (info.venv.is_some() || info.base.is_some());
        let installed = matches!(&info.packages, Some(Ok(pk)) if pk.missing().is_empty());
        let pkg_row = row(
            "4",
            pkg_status,
            "Пакеты: torch, pyg-lib, torch-geometric, torch-pointcloud, optuna",
            pkg_detail,
        )
        .child(
            Button::new("env-install")
                .icon(Icon::new(Lucide::ArrowDownToLine))
                .label(format!(
                    "{} torch {}+{build}",
                    if installed {
                        "Переустановить:"
                    } else {
                        "Установить:"
                    },
                    env::TORCH
                ))
                .primary()
                .small()
                .disabled(!can_install)
                .on_click(cx.listener(|this, _, _, cx| {
                    this.install_packages();
                    cx.notify();
                })),
        );

        // 5. Nevo — для обучения не нужен, только для экспорта отчётов.
        let nevo_row = row(
            "5",
            if info.nevo.is_some() { ok } else { warn },
            "Редактор Nevo",
            match &info.nevo {
                Some(n) => format!("{} · нужен для кнопки «В Nevo» в «Артефактах».", show_path(n)),
                None => "Не найден — экспорт отчётов в Nevo не сработает. Для обучения не нужен.".into(),
            },
        )
        .when(info.nevo.is_none(), |d| {
            d.child(
                Button::new("env-nevo")
                    .icon(Icon::new(Lucide::ArrowDownToLine))
                    .label("Скачать последний релиз")
                    .primary()
                    .small()
                    .disabled(busy)
                    .tooltip("Установщик Nevo для Windows с github.com/eliotBenitez/Nevo")
                    .on_click(cx.listener(|this, _, _, cx| {
                        let a = ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", nevo::INSTALL_PS];
                        this.spawn_setup("powershell".into(), a.map(String::from).to_vec(), "Установка Nevo");
                        cx.notify();
                    })),
            )
        });

        let footer = div().px_8().py_4().text_sm().text_color(p.muted).child(if busy {
            "Идёт установка — лог в «Трансляции». После окончания окружение проверится заново.".to_string()
        } else if let Some(v) = info.venv.as_ref().filter(|_| self.env_ready()) {
            format!("Готово: «Запуск» использует {}.", show_path(v))
        } else {
            "Установка качает около 3 ГБ (torch с CUDA) и идёт 5–15 минут; её можно остановить в «Трансляции».".into()
        });

        v_flex()
            .id("env")
            .size_full()
            .overflow_y_scrollbar()
            .child(head)
            .child(python_row)
            .child(gpu_row)
            .child(venv_row)
            .child(pkg_row)
            .child(nevo_row)
            .child(footer)
            .into_any_element()
    }
}

impl Render for Studio {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.preview.as_ref().is_some_and(|p| p.dirty) {
            self.redraw_preview(Some(window), cx);
        }
        if let Some(py) = self.pending_python.take() {
            Self::set_input(&self.python, py, window, cx);
        }
        if let Some(d) = self.pending_data.take() {
            Self::set_input(&self.data_dir, d, window, cx);
        }
        let cur = self
            .scenarios
            .get(self.sel_scenario)
            .map(|s| s.name.clone());
        if self.tab == Tab::Scenarios && cur != self.scenario_name_for {
            self.scenario_name_for = cur.clone();
            Self::set_input(&self.scenario_name, cur.unwrap_or_default(), window, cx);
        }
        let cur = self.runs.get(self.sel_run).map(|r| r.name.clone());
        if self.tab == Tab::Runs && cur != self.note_for {
            // Сначала сменить note_for: событие Change от set_value должно относиться уже к новому прогону.
            self.note_for = cur;
            let text = self
                .runs
                .get(self.sel_run)
                .and_then(|r| r.studio.as_ref()?.get("note")?.as_str().map(String::from))
                .unwrap_or_default();
            Self::set_input(&self.note, text, window, cx);
        }
        let p = pal(self.dark);
        let live = self.jobs.iter().filter(|j| j.running()).count();
        let nav = |tab: Tab, icon: Lucide, label: String, cx: &mut Context<Self>| {
            let selected = self.tab == tab;
            h_flex()
                .id(label.clone())
                .mx_3()
                .my_0p5()
                .px_4()
                .py_2p5()
                .gap_4()
                .rounded_lg()
                .cursor_pointer()
                .text_color(p.rail_ink)
                .when(!selected, |d| {
                    d.opacity(0.72).hover(|s| s.opacity(1.).bg(p.rail_sel))
                })
                .when(selected, |d| {
                    d.bg(p.rail_sel).font_weight(FontWeight::SEMIBOLD)
                })
                .child(
                    Icon::new(icon)
                        .size_4()
                        .when(selected, |i| i.text_color(p.win)),
                )
                .child(label)
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.tab = tab;
                    this.tab_error = None; // ошибка относилась к прежней вкладке
                    if tab == Tab::Runs {
                        this.reload_runs(cx);
                    }
                    if tab == Tab::Env {
                        this.check_env(cx);
                    }
                    if tab == Tab::Datasets {
                        this.reload_datasets(cx);
                    }
                    if tab == Tab::Testing {
                        this.refresh_tests();
                    }
                    if tab == Tab::Scenarios {
                        this.reload_scenarios();
                    }
                    cx.notify();
                }))
        };
        let jobs =
            v_flex().id("rail-jobs").flex_1().min_h_0().overflow_y_scroll().children(self.jobs.iter().enumerate().rev().map(|(i, j)| {
                let (label, bg, _) = j.status(&p);
                h_flex()
                    .id(("job", i))
                    .mx_3()
                    .px_4()
                    .py_2()
                    .gap_2()
                    .rounded_lg()
                    .cursor_pointer()
                    .text_color(p.rail_ink)
                    .when(i == self.sel_job && self.tab == Tab::Live, |d| {
                        d.bg(p.rail_sel)
                    })
                    .hover(|s| s.bg(p.rail_sel))
                    .child(div().size(px(8.)).flex_shrink_0().rounded_full().bg(bg))
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .overflow_hidden()
                            .whitespace_nowrap()
                            .text_ellipsis()
                            .text_sm()
                            .child(j.run.clone()),
                    )
                    .child(div().text_xs().font_features(tnum()).opacity(0.8).child(
                        if j.running() {
                            hms(j.elapsed())
                        } else {
                            label.to_lowercase()
                        },
                    ))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.sel_job = i;
                        this.tab = Tab::Live;
                        cx.notify();
                    }))
            }));
        let dark = self.dark;
        let rail = v_flex()
            .w(px(220.))
            .h_full()
            .flex_shrink_0()
            .bg(p.rail)
            .border_r_1()
            .border_color(p.rule)
            .child(
                div()
                    .px_5()
                    .pt_5()
                    .pb_4()
                    .child(display("ML Studio", 28., p.rail_ink)),
            )
            .child(nav(Tab::Projects, Lucide::FolderOpen, "Проекты".into(), cx))
            .child(nav(
                Tab::Live,
                Lucide::Radio,
                if live > 0 {
                    format!("Трансляция · {live}")
                } else {
                    "Трансляция".into()
                },
                cx,
            ))
            .child(nav(
                Tab::Runs,
                Lucide::ListOrdered,
                format!("Прогоны · {}", self.runs.len()),
                cx,
            ))
            .child(nav(Tab::Launch, Lucide::Play, "Запуск".into(), cx))
            .child(nav(
                Tab::Scenarios,
                Lucide::ListChecks,
                match &self.active {
                    Some(a) => format!("Сценарии · {}/{}", a.next, a.steps.len()),
                    None => "Сценарии".into(),
                },
                cx,
            ))
            .child(nav(
                Tab::Datasets,
                Lucide::Database,
                format!("Датасеты · {}", self.datasets.len()),
                cx,
            ))
            .child(nav(
                Tab::Testing,
                Lucide::FlaskConical,
                "Тестирование".into(),
                cx,
            ))
            .child(nav(
                Tab::Artifacts,
                Lucide::FileText,
                "Артефакты".into(),
                cx,
            ))
            .child(nav(
                Tab::Env,
                Lucide::Settings2,
                if self.env_ready() {
                    "Окружение".into()
                } else {
                    "Окружение · !".into()
                },
                cx,
            ))
            .when(!self.jobs.is_empty(), |d| {
                d.child(
                    div()
                        .px_4()
                        .pt_6()
                        .pb_1()
                        .text_xs()
                        .text_color(p.rail_ink)
                        .opacity(0.7)
                        .child("ПРОЦЕССЫ"),
                )
                .child(jobs)
            })
            .when(self.jobs.is_empty(), |d| d.child(div().flex_1()))
            .child(self.render_usage(p, cx))
            .child(
                h_flex()
                    .px_4()
                    .py_3()
                    .gap_2()
                    .text_xs()
                    .text_color(p.rail_ink)
                    .child(
                        div()
                            .flex_1()
                            .min_w_0()
                            .overflow_hidden()
                            .whitespace_nowrap()
                            .text_ellipsis()
                            .opacity(0.7)
                            .child(show_path(&self.root)),
                    )
                    .child(
                        div()
                            .id("theme")
                            .cursor_pointer()
                            .px_2()
                            .py_1()
                            .rounded_md()
                            .border_1()
                            .border_color(p.rule)
                            .hover(|s| s.bg(p.rail_sel))
                            .child(if dark {
                                "Тема: тёмная"
                            } else {
                                "Тема: светлая"
                            })
                            .on_click(cx.listener(move |this, _, window, cx| {
                                this.set_dark(!dark, window, cx)
                            })),
                    ),
            );
        let body = match self.tab {
            Tab::Live => self.render_live(p, cx),
            Tab::Runs => self.render_runs(p, cx),
            Tab::Launch => self.render_launch(p, cx),
            Tab::Datasets => self.render_datasets(p, cx),
            Tab::Testing => self.render_testing(p, cx),
            Tab::Artifacts => self.render_artifacts(p, cx),
            Tab::Env => self.render_env(p, cx),
            Tab::Projects => self.render_projects(p, cx),
            Tab::Scenarios => self.render_scenarios(p, cx),
        };
        // Ошибки действий (заметка, повтор, оценка по сценам, копия модели) — над телом вкладки, где их совершили.
        let body = v_flex()
            .size_full()
            .when_some(self.tab_error.clone(), |d, e| {
                d.child(
                    div()
                        .px_8()
                        .py_2()
                        .flex_shrink_0()
                        .border_b_1()
                        .border_color(p.rule)
                        .text_sm()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(p.loss)
                        .child(e),
                )
            })
            .child(div().flex_1().min_h_0().child(body));
        h_flex()
            .size_full()
            .bg(p.bg)
            .text_color(p.ink)
            .font_family(BODY)
            .text_sm()
            .child(rail)
            // Смена вкладки — короткое проявление; id зависит от вкладки, поэтому анимация запускается заново.
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .h_full()
                    .child(body)
                    .with_animation(
                        ElementId::Name(format!("tab-{}", self.tab as u8).into()),
                        Animation::new(Duration::from_millis(180)).with_easing(ease_in_out),
                        |el, t| el.opacity(t),
                    ),
            )
    }
}

// ---------- элементы «полосы» ----------

fn display(text: &str, size: f32, color: Hsla) -> Div {
    div()
        .font_family(DISPLAY)
        .font_weight(FontWeight::BOLD)
        .text_size(px(size))
        .line_height(px(size * 1.05))
        .text_color(color)
        .child(text.to_string())
}

fn small_caps(text: &str, color: Hsla) -> Div {
    div()
        .text_xs()
        .font_weight(FontWeight::SEMIBOLD)
        .text_color(color)
        .child(text.to_uppercase())
}

fn stat_line(label: &str, value: String, color: Hsla, p: Pal) -> Div {
    stat_line_with_label_width(label, value, color, p, 120.)
}

fn compact_stat_line(label: &str, value: String, color: Hsla, p: Pal) -> Div {
    stat_line_with_label_width(label, value, color, p, 48.)
}

fn stat_line_with_label_width(label: &str, value: String, color: Hsla, p: Pal, width: f32) -> Div {
    h_flex()
        .gap_3()
        .child(
            div()
                .w(px(width))
                .flex_shrink_0()
                .text_xs()
                .text_color(p.muted)
                .child(label.to_string()),
        )
        // min_w_0: длинное значение переносится в ширину карточки, а не вылезает за неё.
        .child(
            div()
                .min_w_0()
                .font_features(tnum())
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(color)
                .child(value),
        )
}

/// Модель прогона: best_model.pt (пишет hpo_pointnet.py), у старых прогонов — best.pt лучшего trial.
fn model_file(r: &Run) -> Option<PathBuf> {
    let trial = r.best.as_ref()?.get("best_trial")?.as_u64()?;
    [
        r.path.join("best_model.pt"),
        r.path.join(format!("trial_{trial:04}")).join("best.pt"),
    ]
    .into_iter()
    .find(|p| p.exists())
}

/// Матрица ошибок 2×2 из best.json: строки — истина, столбцы — предсказание; насыщенность — доля строки.
fn confusion_matrix(title: &str, m: &serde_json::Value, color: Hsla, p: Pal) -> Div {
    let n = |i: usize, j: usize| m["confusion"][i][j].as_u64().unwrap_or(0);
    let cell = |i: usize, j: usize| {
        let share = n(i, j) as f64 / (n(i, 0) + n(i, 1)).max(1) as f64;
        v_flex()
            .w(px(104.))
            .h(px(56.))
            .rounded_md()
            .items_center()
            .justify_center()
            .bg(color.opacity(0.06 + 0.44 * share as f32))
            .child(
                div()
                    .font_features(tnum())
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(p.ink)
                    .child(n(i, j).to_string()),
            )
            .child(
                div()
                    .font_features(tnum())
                    .text_xs()
                    .text_color(p.ink)
                    .child(format!("{:.1}%", share * 100.)),
            )
    };
    let label = |t: &str, w: f32| {
        div()
            .w(px(w))
            .text_xs()
            .text_color(p.muted)
            .child(t.to_string())
    };
    v_flex()
        .gap_1()
        .child(small_caps(&format!("Матрица ошибок · {title}"), p.muted))
        .child(
            h_flex()
                .gap_1()
                .child(label("", 76.))
                .child(label("пред. фон", 104.))
                .child(label("пред. цель", 104.)),
        )
        .child(
            h_flex()
                .gap_1()
                .child(label("истина: фон", 76.))
                .child(cell(0, 0))
                .child(cell(0, 1)),
        )
        .child(
            h_flex()
                .gap_1()
                .child(label("истина: цель", 76.))
                .child(cell(1, 0))
                .child(cell(1, 1)),
        )
}

fn stat_big(label: &str, value: String, p: Pal) -> Div {
    v_flex().child(small_caps(label, p.muted)).child(
        div()
            .font_family(DISPLAY)
            .font_weight(FontWeight::BOLD)
            .text_size(px(28.))
            .font_features(tnum())
            .text_color(p.ink)
            .child(value),
    )
}

/// Плашка статуса; «в эфире» мерцает — единственное движение в интерфейсе.
fn chip(label: &'static str, bg: Hsla, fg: Hsla, live: bool) -> AnyElement {
    let base = h_flex()
        .gap_2()
        .px_2p5()
        .py_0p5()
        .rounded_full()
        .bg(bg)
        .text_color(fg)
        .text_xs()
        .font_weight(FontWeight::BOLD)
        .child(label);
    if live {
        base.with_animation(
            "live",
            Animation::new(Duration::from_millis(1600))
                .repeat()
                .with_easing(ease_in_out),
            |el, t| el.opacity(0.65 + 0.35 * (1. - (2. * t - 1.).abs())),
        )
        .into_any_element()
    } else {
        base.into_any_element()
    }
}

/// Строка таблицы сцен инференса: имя сцены тянется, числа 76px справа.
/// Отдельно от `trow`: там первая колонка — номер trial, а здесь нужна широкая колонка имени.
fn scene_row<const N: usize>(cells: [String; N], header: bool, p: Pal) -> Stateful<Div> {
    h_flex()
        .id(ElementId::Name(if header {
            "sh".into()
        } else {
            cells.join("|").into()
        }))
        .w_full()
        .px_4()
        .py_1p5()
        .gap_3()
        .border_b_1()
        .border_color(p.rule)
        .font_features(tnum())
        .when(header, |d| {
            d.text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(p.muted)
        })
        .children(cells.into_iter().enumerate().map(|(i, c)| {
            let cell = div()
                .overflow_hidden()
                .whitespace_nowrap()
                .text_ellipsis()
                .child(if header { c.to_uppercase() } else { c });
            match i {
                0 => cell.w(px(24.)).flex_shrink_0().text_color(p.muted),
                1 => cell.flex_1().min_w_0(),
                _ => cell.w(px(76.)).flex_shrink_0().flex().justify_end(),
            }
        }))
}

/// Строка таблицы: числа справа, табличные цифры, тонкая линейка.
fn trow<const N: usize>(cells: [String; N], header: bool, p: Pal) -> Stateful<Div> {
    h_flex()
        .id(ElementId::Name(if header {
            "th".into()
        } else {
            cells.join("|").into()
        }))
        .w_full()
        .px_4()
        .py_1p5()
        .gap_3()
        .border_b_1()
        .border_color(p.rule)
        .font_features(tnum())
        .when(header, |d| {
            d.text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(p.muted)
        })
        .children(cells.into_iter().enumerate().map(|(i, c)| {
            let cell = div()
                .overflow_hidden()
                .whitespace_nowrap()
                .text_ellipsis()
                .child(if header { c.to_uppercase() } else { c });
            match i {
                0 if N > 3 => cell.w(px(24.)).flex_shrink_0().text_color(p.muted),
                0 => cell.flex_1().min_w_0(),
                _ => cell
                    .w(px(if N > 5 { 76. } else { 70. }))
                    .flex_shrink_0()
                    .flex()
                    .justify_end(),
            }
        }))
        // место под полосу длительности, чтобы заголовок совпадал со строками
        .when(header, |d| d.child(div().w(px(56.)).flex_shrink_0()))
}

/// Карточка: поверхность на ступень выше фона, тонкая рамка, скругление.
fn card(p: Pal) -> Div {
    h_flex()
        .rounded_xl()
        .bg(p.surface)
        .border_1()
        .border_color(p.rule)
}

/// Шапка прогона: карточка с лёгким оттенком цвета класса (вместо сплошной заливки).
fn class_card(color: Hsla, p: Pal) -> Div {
    h_flex()
        .mx_6()
        .mt_4()
        .px_5()
        .py_4()
        .gap_4()
        .rounded_xl()
        .bg(color.opacity(0.14))
        .border_1()
        .border_color(color.opacity(0.4))
        .text_color(p.ink)
}

/// Кнопка на шапке карточки: читается на любом оттенке.
/// Кнопка-иконка с подсказкой — для второстепенных действий, чтобы шапка не переполнялась.
fn icon_button(id: &'static str, icon: Lucide, tip: &'static str, p: Pal) -> Stateful<Div> {
    h_flex()
        .id(id)
        .flex_shrink_0()
        .p_2()
        .rounded_lg()
        .cursor_pointer()
        .bg(p.raised)
        .text_color(p.ink)
        .hover(|s| s.opacity(0.88))
        .child(Icon::new(icon).size_4().text_color(p.ink))
        .tooltip(move |w, cx| Tooltip::new(tip).build(w, cx))
}

fn band_button(id: &'static str, icon: Lucide, label: &str, bg: Hsla, fg: Hsla) -> Stateful<Div> {
    h_flex()
        .id(id)
        .gap_2()
        .flex_shrink_0()
        .whitespace_nowrap()
        .px_3()
        .py_1p5()
        .rounded_lg()
        .cursor_pointer()
        .bg(bg)
        .text_color(fg)
        .text_sm()
        .font_weight(FontWeight::SEMIBOLD)
        .hover(|s| s.opacity(0.88))
        .child(Icon::new(icon).size_4().text_color(fg))
        .child(label.to_string())
}

/// График F1 по эпохам: шкала Y и линия вылета (baseline + 0.05) прямо на графике.
/// `floor` — нижняя граница, общая для нескольких графиков (сравнение прогонов в одном масштабе).
fn f1_chart(
    id: impl Into<ElementId>,
    pts: Vec<(u32, f64)>,
    line: Option<f64>,
    floor: Option<f64>,
    stroke: Hsla,
) -> AnyElement {
    let lo = pts
        .iter()
        .map(|p| p.1)
        .chain(line)
        .chain(floor)
        .fold(1.0f64, f64::min);
    // Круглые деления: шаг 0.1, а при широком диапазоне 0.2; верх ровно 1.0 (без запаса сверху).
    let lo = ((lo - 0.05) * 10.).floor().max(0.) / 10.;
    let (lo, step) = if 1.0 - lo > 0.55 {
        ((lo * 5.).floor() / 5., 0.2)
    } else {
        (lo, 0.1)
    };
    let ticks = ((1.0 - lo) / step).round() as usize + 1;
    let chart = LineChart::new(pts)
        .id(id)
        .x(|(e, _)| e.to_string())
        .y(|(_, f)| *f)
        .stroke(stroke)
        .linear() // ломаная: сплайн рисует значения между эпохами, которых не было
        .dot()
        .y_axis(true)
        .y_domain(lo, 1.0)
        .y_padding(0., 0.)
        .y_tick_count(ticks)
        .y_tick_format(|v| format!("{v:.1}"));
    match line {
        Some(l) => chart.reference_line(l).into_any_element(),
        None => chart.into_any_element(),
    }
}

/// Заголовок графика с легендой: пунктир на графике — линия вылета (цвет пунктира задаёт тема компонента).
fn chart_title(text: &str, line: Option<f64>, p: Pal) -> Div {
    h_flex()
        .gap_3()
        .child(small_caps(text, p.muted))
        .when_some(line, |d, l| {
            d.child(
                h_flex()
                    .gap_1()
                    .text_xs()
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(p.loss)
                    .child(div().w(px(6.)).h(px(2.)).bg(p.loss))
                    .child(div().w(px(6.)).h(px(2.)).bg(p.loss))
                    .child(format!("пунктир — линия вылета {l:.3}")),
            )
        })
}

/// Имя кэша предсказания: сцена + чекпойнт (путь, размер, время изменения), FNV-1a — стабилен между версиями Rust.
/// Имя `preview_v4_<сцена>_<модель>.npy`: префикс до последнего `_` общий для всех версий модели на сцене.
fn prediction_cache_name(scene: &Path, checkpoint: &Path) -> String {
    let (len, mtime) = std::fs::metadata(checkpoint).map_or((0, 0), |m| {
        let secs = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
        (m.len(), secs)
    });
    let fnv = |s: String| s.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    let scene_hash = fnv(scene.display().to_string());
    let model_hash = fnv(format!("{}\0{len}\0{mtime}", checkpoint.display()));
    format!("preview_v4_{scene_hash:016x}_{model_hash:016x}.npy")
}

/// Удалить предсказания той же сцены от прежних версий чекпойнта (имя другое, префикс тот же).
fn remove_stale_predictions(dir: &Path, keep: &str) {
    let Some(i) = keep.rfind('_') else { return };
    let prefix = &keep[..=i];
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name != keep && name.starts_with(prefix) && name.ends_with(".npy") {
            _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Имя прогона по умолчанию: <класс>_<признаки> и суффикс настроек обучения (свой study).
fn auto_name(args: &[String]) -> String {
    let focal = TrainOpts::from_args(args).suffix();
    format!(
        "{}_{}{focal}",
        Job::arg(args, "--target").unwrap_or_default(),
        Job::arg(args, "--features").unwrap_or_default()
    )
}

/// Настройки обучения сверх гиперпараметров Optuna: loss, веса классов, баланс и граница патчей.
/// Study сверяет их (COMPATIBILITY_FIELDS в hpo_core/study.py), поэтому у каждого набора свой
/// каталог прогона — `suffix()`, тот же, что строит run_all.py.
#[derive(Clone, Debug, PartialEq)]
struct TrainOpts {
    /// "ce", "focal" или "focal_lovasz" (--loss).
    loss: String,
    gamma: String,
    lovasz: String,
    power: String,
    /// Some(доля патчей с центром на цели) — --equibatch.
    positive: Option<String>,
    bfrac: String,
    bweight: String,
    bradius: String,
}

impl Default for TrainOpts {
    fn default() -> Self {
        let s = |v: &str| v.to_string();
        Self {
            loss: s("ce"),
            gamma: s("2"),
            lovasz: s("0.5"),
            power: s("1"),
            positive: None,
            bfrac: s("0"),
            bweight: s("1"),
            bradius: s("0.5"),
        }
    }
}

fn num(s: &str) -> Option<f64> {
    s.trim().parse::<f64>().ok().filter(|x| x.is_finite())
}

impl TrainOpts {
    /// По флагам командной строки hpo_pointnet.py.
    fn from_args(args: &[String]) -> Self {
        let d = Self::default();
        let get = |k: &str, d: String| Job::arg(args, k).unwrap_or(d);
        Self {
            positive: args
                .iter()
                .any(|a| a == "--equibatch")
                .then(|| get("--equibatch-positive-fraction", "0.5".into())),
            loss: get("--loss", d.loss),
            gamma: get("--focal-gamma", d.gamma),
            lovasz: get("--lovasz-weight", d.lovasz),
            power: get("--class-weight-power", d.power),
            bfrac: get("--boundary-patch-fraction", d.bfrac),
            bweight: get("--boundary-weight", d.bweight),
            bradius: get("--boundary-radius", d.bradius),
        }
    }

    /// По run_info прогона (best.json/split.json): строка loss и поля, которые пишет hpo_pointnet.py.
    fn from_info(loss: &str, info: impl Fn(&str) -> Option<String>) -> Self {
        let d = Self::default();
        let gamma = loss.split("focal(gamma=").nth(1).and_then(|r| r.split(')').next());
        let lovasz = loss.rsplit_once('+').and_then(|(_, r)| r.strip_suffix("*lovasz"));
        let kind = if lovasz.is_some() {
            "focal_lovasz"
        } else if loss.starts_with("focal") {
            "focal"
        } else {
            "ce"
        };
        Self {
            loss: kind.into(),
            gamma: gamma.map_or(d.gamma, String::from),
            lovasz: lovasz.map_or(d.lovasz, String::from),
            power: info("class_weight_power").unwrap_or(d.power),
            positive: info("positive_fraction"),
            bfrac: info("boundary_fraction").unwrap_or(d.bfrac),
            bweight: info("boundary_weight").unwrap_or(d.bweight),
            bradius: info("boundary_radius").unwrap_or(d.bradius),
        }
    }

    fn differs(v: &str, default: f64) -> bool {
        num(v) != Some(default)
    }

    fn boundary(&self) -> bool {
        Self::differs(&self.bfrac, 0.) || Self::differs(&self.bweight, 1.)
    }

    /// Флаги для hpo_pointnet.py / run_all.py; значения по умолчанию не пишутся.
    fn args(&self) -> Vec<String> {
        let mut a: Vec<String> = vec![];
        let mut push = |k: &str, v: &str| a.extend([k.to_string(), v.trim().to_string()]);
        if self.loss != "ce" {
            push("--loss", &self.loss);
            push("--focal-gamma", &self.gamma);
        }
        if self.loss == "focal_lovasz" {
            push("--lovasz-weight", &self.lovasz);
        }
        if Self::differs(&self.power, 1.) {
            push("--class-weight-power", &self.power);
        }
        if let Some(f) = &self.positive {
            push("--equibatch-positive-fraction", f);
        }
        if Self::differs(&self.bfrac, 0.) {
            push("--boundary-patch-fraction", &self.bfrac);
        }
        if Self::differs(&self.bweight, 1.) {
            push("--boundary-weight", &self.bweight);
        }
        if self.boundary() && Self::differs(&self.bradius, 0.5) {
            push("--boundary-radius", &self.bradius);
        }
        if self.positive.is_some() {
            a.push("--equibatch".into());
        }
        a
    }

    /// Суффикс имени прогона (как в run_all.py): у Focal Loss и каждой gamma свой study,
    /// значения по умолчанию (gamma 2 и т. д.) — без чисел.
    fn suffix(&self) -> String {
        let n = |v: &str| num(v).map_or_else(|| v.trim().to_string(), |x| x.to_string());
        let mut s = String::new();
        match self.loss.as_str() {
            "focal" => s += "_focal",
            "focal_lovasz" => s += "_focal_lovasz",
            _ => {}
        }
        if self.loss != "ce" && Self::differs(&self.gamma, 2.) {
            s += &format!("_g{}", n(&self.gamma));
        }
        if self.loss == "focal_lovasz" && Self::differs(&self.lovasz, 0.5) {
            s += &format!("_lw{}", n(&self.lovasz));
        }
        if Self::differs(&self.power, 1.) {
            s += &format!("_cw{}", n(&self.power));
        }
        if let Some(f) = &self.positive {
            s += "_eq";
            if Self::differs(f, 0.5) {
                s += &n(f);
            }
        }
        if Self::differs(&self.bfrac, 0.) {
            s += &format!("_bf{}", n(&self.bfrac));
        }
        if Self::differs(&self.bweight, 1.) {
            s += &format!("_bw{}", n(&self.bweight));
        }
        if self.boundary() && Self::differs(&self.bradius, 0.5) {
            s += &format!("_br{}", n(&self.bradius));
        }
        s
    }

    /// Текст ошибки для формы (те же ограничения, что проверяет hpo_pointnet.py).
    fn error(&self) -> Option<&'static str> {
        let within = |v: &str, lo: f64, hi: f64| num(v).is_some_and(|x| (lo..=hi).contains(&x));
        if self.loss != "ce" && !num(&self.gamma).is_some_and(|g| g > 0.) {
            Some("Gamma для Focal Loss должна быть положительным числом.")
        } else if self.loss == "focal_lovasz" && !within(&self.lovasz, 0., 1.) {
            Some("Доля Lovász должна быть от 0 до 1.")
        } else if !within(&self.power, 0., 1.) {
            Some("Степень весов классов должна быть от 0 до 1.")
        } else if self
            .positive
            .as_deref()
            .is_some_and(|f| !num(f).is_some_and(|x| x > 0. && x < 1.))
        {
            Some("Доля патчей с центром на цели должна быть строго между 0 и 1.")
        } else if !num(&self.bfrac).is_some_and(|x| (0. ..1.).contains(&x)) {
            Some("Доля граничных патчей должна быть от 0 до 1 (не включая 1).")
        } else if Self::differs(&self.bfrac, 0.) && self.positive.is_none() {
            Some("Граничные патчи работают только с балансом патчей «цель / фон».")
        } else if !num(&self.bweight).is_some_and(|x| x >= 1.) {
            Some("Вес граничных точек должен быть не меньше 1.")
        } else if !num(&self.bradius).is_some_and(|x| x > 0.) {
            Some("Радиус границы должен быть положительным.")
        } else if Self::differs(&self.bweight, 1.)
            && self.loss == "focal_lovasz"
            && !Self::differs(&self.lovasz, 1.)
        {
            Some("Вес граничных точек действует только на Focal-часть: уменьшите долю Lovász.")
        } else {
            None
        }
    }
}

/// Где сцена была при обучении прогона: train (метрики завышены), val, новая; «—» — неизвестно.
fn scene_split(run: Option<&Run>, scene: &str) -> String {
    let has = |k: &str| {
        run.and_then(|r| {
            r.split
                .as_ref()?
                .get(k)?
                .as_array()
                .map(|a| a.iter().any(|v| v.as_str() == Some(scene)))
        })
    };
    match (has("train"), has("val")) {
        (Some(true), _) => "train".into(),
        (_, Some(true)) => "val".into(),
        (Some(false), Some(false)) => "новая".into(),
        _ => "—".into(),
    }
}

/// Заменяет значение `key` в аргументах командной строки или дописывает пару.
fn set_arg(args: &mut Vec<String>, key: &str, value: String) {
    match args.iter().position(|a| a == key) {
        Some(i) if i + 1 < args.len() => args[i + 1] = value,
        _ => args.extend([key.to_string(), value]),
    }
}

/// Путь аргумента CLI `<dir>/<name>`: единые прямые слэши — так пути сравнивает conflict()
/// и показывает лог; обратные слэши Windows нормализуются.
fn cli_path(dir: &str, name: &str) -> String {
    let dir = dir.trim().trim_end_matches(['/', '\\']).replace('\\', "/");
    if dir.is_empty() {
        name.to_string()
    } else {
        format!("{dir}/{name}")
    }
}

/// Путь для сравнения в conflict(): относительно корня проекта с прямыми слэшами
/// (кли_path строит аргументы так же), поэтому абсолютное и относительное написание
/// одного каталога совпадают; вне корня — путь целиком с прямыми слэшами.
fn norm_path(root: &Path, p: &Path) -> String {
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    };
    abs.strip_prefix(root)
        .map(|rel| rel.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| abs.to_string_lossy().replace('\\', "/"))
}

/// Прогон — каталог целиком: в него пишут и предсказания (<прогон>/pred/…), поэтому путь
/// предсказания поднимается до каталога прогона. Работает для прогона любой глубины
/// (runs/x, out/deep/x, вне корня проекта); пути без pred/ не меняются.
fn pred_run_path(root: &Path, p: &Path) -> String {
    let n = norm_path(root, p);
    if let Some((run, _)) = n.rsplit_once("/pred/") {
        run.to_string()
    } else {
        n.strip_suffix("/pred").unwrap_or(&n).to_string()
    }
}

/// Число или строка из JSON-артефакта — в строку аргумента CLI; пусто/не скаляр — умолчание.
/// Строка или число из JSON как текст; иначе None (null у выключенных настроек).
fn json_scalar(v: Option<&serde_json::Value>) -> Option<String> {
    match v? {
        serde_json::Value::String(s) if !s.is_empty() => Some(s.clone()),
        v => v.is_number().then(|| v.to_string()),
    }
}

fn json_scalar_str(v: Option<&serde_json::Value>, default: &str) -> String {
    match v {
        Some(serde_json::Value::String(s)) if !s.is_empty() => s.clone(),
        Some(v) if v.is_number() => v.to_string(),
        _ => default.to_string(),
    }
}

/// Аргументы финального прогона: параметры лучшего trial из best.json, один trial,
/// --fixed-params и каталог final_<имя>. Эпохи/воркеры — из формы.
fn final_run_args(
    best: &serde_json::Value,
    train: &TrainOpts,
    target: &str,
    features: &str,
    final_data: &str,
    output: &str,
    params_path: &str,
    epochs: &str,
    workers: &str,
) -> Vec<String> {
    let field = |k: &str, d: &str| json_scalar_str(best.get(k), d);
    let mut args = vec![
        "-u".into(),
        "hpo_pointnet.py".into(),
        "--data-dir".into(),
        final_data.to_string(),
        "--output-dir".into(),
        output.to_string(),
        "--target".into(),
        target.to_string(),
        "--features".into(),
        features.to_string(),
        "--fixed-params".into(),
        params_path.to_string(),
        "--n-trials".into(),
        "1".into(),
        "--epochs".into(),
        epochs.to_string(),
        "--seed".into(),
        field("seed", "42"),
        "--patch-points".into(),
        field("patch_points", "16384"),
        "--patches-per-scene".into(),
        field("patches_per_scene", "2"),
        "--workers".into(),
        workers.to_string(),
    ];
    args.extend(train.args());
    args
}

/// Путь с разделителями текущей ОС (аргумент запуска мог прийти с «/»).
fn show_path(p: &Path) -> String {
    let s = p.display().to_string();
    if cfg!(windows) {
        s.replace('/', "\\")
    } else {
        s
    }
}

/// 500000 → «500 000».
fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push('\u{202F}');
        }
        out.push(ch);
    }
    out
}

/// Линия вылета: ниже неё trials не обгоняют baseline на 0.05.
fn relegation(p: Pal) -> Div {
    h_flex()
        .px_4()
        .gap_2()
        .items_center()
        .child(div().h(px(2.)).flex_1().bg(p.loss))
        .child(
            div()
                .text_xs()
                .font_weight(FontWeight::BOLD)
                .text_color(p.loss)
                .child("ЛИНИЯ BASELINE + 0.05"),
        )
        .child(div().h(px(2.)).w(px(16.)).bg(p.loss))
}

/// Полоса, длина которой — точная доля от максимума (время trial, точки сцены).
fn duration_bar(value: f64, max: f64, p: Pal) -> Div {
    let frac = if max > 0. {
        (value / max).clamp(0., 1.) as f32
    } else {
        0.
    };
    div()
        .w(px(56.))
        .h(px(6.))
        .flex_shrink_0()
        .rounded_full()
        .bg(p.rule)
        .child(div().h_full().rounded_full().w(relative(frac)).bg(p.muted))
}

fn choice(id: impl Into<ElementId>, label: impl Into<SharedString>, selected: bool) -> Button {
    let b = Button::new(id).label(label).small();
    if selected { b.primary() } else { b.outline() }
}

// ---------- процессы и файлы ----------

/// Читает поток построчно; `\r` (прогресс-бары) — берётся последний сегмент строки.
fn pipe(r: impl Read + Send + 'static, out: Arc<Mutex<Output>>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut r = BufReader::new(r);
        let mut buf = vec![];
        while r
            .read_until(b'\n', &mut buf)
            .map(|n| n > 0)
            .unwrap_or(false)
        {
            let s = String::from_utf8_lossy(&buf);
            let s = s.trim_end_matches(['\n', '\r']);
            let line = s.rsplit('\r').next().unwrap_or(s).to_string();
            // В памяти — последние LOG_CAP строк (Output.push выбрасывает начало); полный лог — в файле.
            out.lock().unwrap().push(line);
            buf.clear();
        }
    })
}

/// Останавливает процесс вместе с дочерними (run_all.py запускает python-подпроцессы).
fn kill(job: &Job) {
    let Some(pid) = job.pid else { return };
    #[cfg(windows)]
    {
        let mut c = Command::new("taskkill");
        c.args(["/PID", &pid.to_string(), "/T", "/F"]);
        std::os::windows::process::CommandExt::creation_flags(&mut c, 0x0800_0000);
        _ = c.stdout(Stdio::null()).stderr(Stdio::null()).status();
    }
    #[cfg(not(windows))]
    {
        _ = Command::new("pkill")
            .args(["-TERM", "-P", &pid.to_string()])
            .status();
        _ = Command::new("kill").arg(pid.to_string()).status();
    }
}

fn open_path(p: &Path) {
    #[cfg(windows)]
    let prog = "explorer";
    #[cfg(target_os = "macos")]
    let prog = "open";
    #[cfg(all(unix, not(target_os = "macos")))]
    let prog = "xdg-open";
    _ = Command::new(prog).arg(p).spawn();
}

fn fmt(v: Option<f64>) -> String {
    v.map_or("—".into(), |v| format!("{v:.4}"))
}

fn hms(d: Duration) -> String {
    let s = d.as_secs();
    format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
}

/// Корень проекта: аргумент, иначе последний открытый (как в IDE), иначе ближайший вверх от cwd/.exe каталог с hpo_pointnet.py.
/// Если ничего не нашлось — cwd, и приложение откроется на экране «Проекты».
fn find_root() -> PathBuf {
    if let Some(a) = std::env::args().nth(1) {
        return PathBuf::from(a);
    }
    if let Some(last) = projects::Projects::load()
        .recent
        .into_iter()
        .find(|p| projects::is_project(p))
    {
        return last;
    }
    let cwd = std::env::current_dir().unwrap_or_default();
    let exe = std::env::current_exe().unwrap_or_default();
    [cwd.as_path(), exe.as_path()]
        .iter()
        .flat_map(|p| p.ancestors())
        .find(|p| projects::is_project(p))
        .map(Path::to_path_buf)
        .unwrap_or(cwd)
}

/// Системное уведомление о конце процесса — прогоны идут часами, окно обычно свёрнуто.
fn notify_end(job: &Job, out: &Output) {
    let ok = matches!(out.exit, Some(Ok(())));
    let summary = format!(
        "{}: {}",
        job.run,
        if ok {
            "завершён"
        } else {
            "ошибка"
        }
    );
    // У инференса нет trials: итог — сколько сцен посчитано и лучшая из них.
    let scenes: Vec<&SceneMetrics> = job
        .scenes
        .iter()
        .map(|(_, m)| m)
        .filter(|m| !m.f1.is_nan())
        .collect();
    let best = if job.predicting {
        scenes.iter().map(|m| m.f1).reduce(f64::max)
    } else {
        job.standings().first().map(|r| r.1)
    };
    let line = if job.predicting {
        scenes.iter().find_map(|m| m.baseline)
    } else {
        job.baseline
    }
    .map(|b| b + MARGIN);
    let body = if ok {
        let units = if job.predicting {
            format!("сцен: {} · ", scenes.len())
        } else {
            String::new()
        };
        match (best, line) {
            (Some(b), Some(l)) => format!(
                "{units}Лучший F1 {b:.4} — {} линии вылета {l:.4}",
                if b >= l { "выше" } else { "ниже" }
            ),
            (Some(b), None) => format!("{units}Лучший F1 {b:.4}"),
            _ => format!("За {}", hms(job.elapsed())),
        }
    } else {
        // последняя содержательная строка лога — обычно причина падения
        out.lines
            .iter()
            .rev()
            .find(|l| !l.starts_with("===") && !l.trim().is_empty())
            .cloned()
            .unwrap_or_default()
    };
    // Вызов ОС может блокировать — не в UI-потоке.
    std::thread::spawn(move || {
        _ = notify_rust::Notification::new()
            .appname("ML Studio")
            .summary(&summary)
            .body(&body)
            .show()
    });
}

fn initial_window_bounds(visible: Bounds<Pixels>) -> Bounds<Pixels> {
    Bounds::centered_at(
        visible.center(),
        size(px(1440.), px(900.)).min(&size(visible.size.width * 0.9, visible.size.height * 0.9)),
    )
}

fn main() {
    let root = find_root();
    // AllAssets: полный каталог Lucide (+~1 МБ), чтобы не перечислять иконки вручную.
    gpui_kit::application()
        .with_assets(gpui_kit::assets::AllAssets)
        .run(move |cx| {
            gpui_kit::init(cx);
            cx.text_system()
                .add_fonts(vec![
                    Cow::Borrowed(include_bytes!("../assets/fonts/PTSansNarrow-Bold.ttf")),
                    Cow::Borrowed(include_bytes!("../assets/fonts/GolosText.ttf")),
                    Cow::Borrowed(include_bytes!("../assets/fonts/PTMono.ttf")),
                ])
                .expect("шрифт PT Sans Narrow");
            cx.on_window_closed(|cx, _| cx.quit()).detach();
            let bounds = cx.primary_display()
                .map(|display| initial_window_bounds(display.visible_bounds()))
                .unwrap_or_else(|| Bounds::centered(None, size(px(1024.), px(768.)), cx));
            let options = WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some("ML Studio".into()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            gpui_kit::open_window(options, cx, |window, cx| {
                cx.new(|cx| Studio::new(root, window, cx))
            })
            .expect("окно не открылось");
            cx.activate(true);
        });
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    #[test]
    fn artifact_views_start_with_documents_and_have_distinct_labels() {
        use super::ArtifactView;
        assert_eq!(ArtifactView::default(), ArtifactView::Documents);
        let labels: Vec<_> = ArtifactView::ALL.iter().map(|view| view.label()).collect();
        assert_eq!(labels, ["Документы", "Модели", "Логи"]);
    }

    #[test]
    fn resources_start_expanded_and_toggle() {
        let mut state = super::ResourcesVisibility::default();
        assert!(!state.collapsed());
        state.toggle();
        assert!(state.collapsed());
        state.toggle();
        assert!(!state.collapsed());
    }

    #[test]
    fn initial_window_fits_visible_display() {
        use super::*;
        for visible in [
            Bounds::new(point(px(0.), px(0.)), size(px(1280.), px(720.))),
            Bounds::new(point(px(100.), px(40.)), size(px(1920.), px(1040.))),
        ] {
            let window = initial_window_bounds(visible);
            assert!(window.origin.x > visible.origin.x);
            assert!(window.origin.y > visible.origin.y);
            assert!(window.origin.x + window.size.width < visible.origin.x + visible.size.width);
            assert!(window.origin.y + window.size.height < visible.origin.y + visible.size.height);
        }
    }

    /// Лог процесса растёт часами: в памяти держится только хвост, на диск пишется всё.
    #[test]
    fn output_keeps_tail_only() {
        let mut out = super::Output::default();
        let total = super::LOG_CAP * 3;
        for i in 0..total {
            out.push(format!("строка {i}"));
        }
        assert!(out.lines.len() <= super::LOG_CAP);
        assert!(out.dropped > 0);
        // Номера сквозные: строка с индексом i в буфере — (dropped + i)-я с начала лога.
        assert_eq!(
            out.lines.first().unwrap(),
            &format!("строка {}", out.dropped)
        );
        assert_eq!(out.lines.last().unwrap(), &format!("строка {}", total - 1));
    }

    #[test]
    fn set_arg() {
        let mut a: Vec<String> = ["x", "--output-dir", "runs/a"].map(String::from).to_vec();
        super::set_arg(&mut a, "--output-dir", "runs/b".into());
        super::set_arg(&mut a, "--fixed-params", "p.json".into());
        assert_eq!(
            a,
            ["x", "--output-dir", "runs/b", "--fixed-params", "p.json"]
        );
    }

    #[test]
    fn prediction_cache_separates_scenes_and_models() {
        use super::prediction_cache_name;
        let scene = Path::new("data/a/scene.npy");
        let model = Path::new("models/a.pt");
        assert_eq!(prediction_cache_name(scene, model), prediction_cache_name(scene, model));
        assert_ne!(prediction_cache_name(scene, model), prediction_cache_name(Path::new("data/b/scene.npy"), model));
        assert_ne!(prediction_cache_name(scene, model), prediction_cache_name(scene, Path::new("other/a.pt")));
    }

    #[test]
    fn auto_name() {
        let mut a: Vec<String> = ["--target", "ground", "--features", "xyz"]
            .map(String::from)
            .to_vec();
        assert_eq!(super::auto_name(&a), "ground_xyz");
        super::set_arg(&mut a, "--loss", "focal".into());
        assert_eq!(super::auto_name(&a), "ground_xyz_focal");
        super::set_arg(&mut a, "--focal-gamma", "2.0".into());
        assert_eq!(super::auto_name(&a), "ground_xyz_focal");
        super::set_arg(&mut a, "--focal-gamma", "0.5".into());
        assert_eq!(super::auto_name(&a), "ground_xyz_focal_g0.5");
        super::set_arg(&mut a, "--loss", "ce".into());
        assert_eq!(super::auto_name(&a), "ground_xyz");
    }

    /// Пути в аргументах CLI — с прямыми слэшами: так их сравнивает conflict() и показывает лог.
    #[test]
    fn cli_path_normalizes_separators() {
        assert_eq!(super::cli_path("runs", "ground_xyz"), "runs/ground_xyz");
        assert_eq!(super::cli_path("runs/", "ground_xyz"), "runs/ground_xyz");
        assert_eq!(super::cli_path("runs\\", "ground_xyz"), "runs/ground_xyz");
        assert_eq!(
            super::cli_path("D:\\TEMP\\smoke", "final_x"),
            "D:/TEMP/smoke/final_x"
        );
        assert_eq!(super::cli_path("", "x"), "x");
    }

    /// Числовые поля best.json читаются как числа, отсутствующие — значение по умолчанию.
    #[test]
    fn json_scalar_str_numbers_and_defaults() {
        let v = serde_json::json!({"seed": 42, "loss": "ce"});
        assert_eq!(
            super::json_scalar_str(v.get("seed"), "7"),
            "42"
        );
        assert_eq!(super::json_scalar_str(v.get("loss"), "x"), "ce");
        assert_eq!(super::json_scalar_str(v.get("patch_points"), "16384"), "16384");
        assert_eq!(super::json_scalar_str(None, "2"), "2");
    }

    /// Команда финального прогона: один trial, --fixed-params, final_<имя>, параметры из best.json.
    #[test]
    fn final_run_args_from_best_json() {
        let best = serde_json::json!({
            "seed": 42, "patch_points": 1024, "patches_per_scene": 3, "params": {"lr": 0.01}
        });
        let args = super::final_run_args(
            &best, &super::TrainOpts::default(), "ground", "xyz", "exports_final",
            "runs/final_ground_xyz", "runs/ground_xyz.params.json", "30", "2",
        );
        let get = |k: &str| {
            args.iter().position(|a| a == k).map(|i| args[i + 1].clone()).unwrap()
        };
        assert_eq!(get("-u"), "hpo_pointnet.py");
        assert_eq!(get("--data-dir"), "exports_final");
        assert_eq!(get("--output-dir"), "runs/final_ground_xyz");
        assert_eq!(get("--target"), "ground");
        assert_eq!(get("--features"), "xyz");
        assert_eq!(get("--fixed-params"), "runs/ground_xyz.params.json");
        assert_eq!(get("--n-trials"), "1");
        assert_eq!(get("--epochs"), "30");
        assert_eq!(get("--seed"), "42");
        assert_eq!(get("--patch-points"), "1024");
        assert_eq!(get("--patches-per-scene"), "3");
        assert_eq!(get("--workers"), "2");
        assert!(!args.iter().any(|a| a == "--loss"));
    }

    /// Пути предсказаний поднимаются к каталогу прогона — с любым числом компонент
    /// и вне корня проекта; --output-dir сравнивается как есть.
    #[test]
    fn conflict_paths_match_at_any_depth() {
        let root = Path::new(if cfg!(windows) { "D:/proj" } else { "/proj" });
        // Обучение: каталог любой глубины, относительное и абсолютное написание.
        assert_eq!(super::norm_path(root, &PathBuf::from("out/deep/x")), "out/deep/x");
        assert_eq!(
            super::norm_path(root, &root.join("out").join("deep").join("x")),
            "out/deep/x"
        );
        // Предсказания того же прогона: файл в <прогон>/pred и каталог <прогон>/pred.
        assert_eq!(
            super::pred_run_path(root, &root.join("out/deep/x/pred/scene1__trial0.npy")),
            "out/deep/x"
        );
        assert_eq!(
            super::pred_run_path(root, &root.join("out/deep/x").join("pred")),
            "out/deep/x"
        );
        // Относительное написание пути предсказания — тот же прогон (разделители Windows).
        #[cfg(windows)]
        assert_eq!(
            super::pred_run_path(root, &PathBuf::from("out/deep/x\\pred\\s.npy")),
            "out/deep/x"
        );
        // Вне корня проекта: путь целиком, предсказание тоже поднимается к прогону.
        let outside = if cfg!(windows) { "D:/TEMP" } else { "/tmp" };
        assert_eq!(
            super::norm_path(root, &PathBuf::from(format!("{outside}/smoke/x"))),
            format!("{outside}/smoke/x")
        );
        assert_eq!(
            super::pred_run_path(root, &PathBuf::from(format!("{outside}/smoke/x/pred/s.npy"))),
            format!("{outside}/smoke/x")
        );
        // Тест моделей пишет вне прогонов — путь не поднимается.
        assert_eq!(
            super::pred_run_path(root, &root.join("tests").join("m__d")),
            "tests/m__d"
        );
    }

    /// Focal-прогон сохраняет gamma из строки loss; отсутствующие поля — по умолчанию.
    #[test]
    fn final_run_args_focal_gamma_and_defaults() {
        let best = serde_json::json!({"params": {}});
        let args = super::final_run_args(
            &best, &super::TrainOpts::from_info("focal(gamma=1.5)", |_| None), "noise", "xyz_hag", "fin",
            "out/final_noise", "out/noise.params.json", "5", "0",
        );
        let get = |k: &str| {
            args.iter().position(|a| a == k).map(|i| args[i + 1].clone()).unwrap()
        };
        assert_eq!(get("--loss"), "focal");
        assert_eq!(get("--focal-gamma"), "1.5");
        assert_eq!(get("--seed"), "42");
        assert_eq!(get("--patch-points"), "16384");
        assert_eq!(get("--patches-per-scene"), "2");
    }

    /// Настройки из run_info лучшего прогона дают те же флаги и тот же каталог, что и форма.
    #[test]
    fn train_opts_from_run_info() {
        let best = serde_json::json!({"class_weight_power": 0.47, "positive_fraction": 0.5,
            "boundary_fraction": 0.5, "boundary_weight": 2.0, "boundary_radius": 0.5,
            "boundary_space": "xyz"});
        let t = super::TrainOpts::from_info("(1-0.5)*focal(gamma=2)+0.5*lovasz", |k| super::json_scalar(best.get(k)));
        assert_eq!(t.loss, "focal_lovasz");
        assert_eq!(t.suffix(), "_focal_lovasz_cw0.47_eq_bf0.5_bw2");
        let args = t.args();
        let get = |k: &str| super::Job::arg(&args, k);
        assert_eq!(get("--loss").as_deref(), Some("focal_lovasz"));
        assert_eq!(get("--lovasz-weight").as_deref(), Some("0.5"));
        assert_eq!(get("--class-weight-power").as_deref(), Some("0.47"));
        assert!(args.iter().any(|a| a == "--equibatch"));
        assert_eq!(get("--boundary-radius"), None); // 0.5 — по умолчанию
        assert_eq!(super::TrainOpts::from_args(&args).suffix(), t.suffix());
        assert_eq!(t.error(), None);
        // Старый прогон без новых полей: прежние CE и пустой суффикс.
        let old = super::TrainOpts::from_info("ce", |_| None);
        assert_eq!((old.args(), old.suffix()), (vec![], String::new()));
        let bad = super::TrainOpts { bfrac: "0.5".into(), ..Default::default() };
        assert!(bad.error().is_some()); // граница без баланса патчей
    }

    #[test]
    fn utc_stamp() {
        assert_eq!(super::utc_stamp(0), "1970-01-01_00-00-00Z");
        assert_eq!(super::utc_stamp(1_700_000_000), "2023-11-14_22-13-20Z");
        assert_eq!(super::utc_stamp(951_782_400), "2000-02-29_00-00-00Z");
    }

    /// Живой лог теста модели (mlstudio-logs) превращается в таблицу сцен: заголовки и метрики парами,
    /// сцены без метрик не показываются как посчитанные.
    #[test]
    fn predicting_log_fills_scenes() {
        let mut job = super::Job {
            run: "Тест".into(),
            program: String::new(),
            args: vec![],
            pid: None,
            out: Default::default(),
            started: std::time::Instant::now(),
            ended: None,
            stopped: false,
            setup: false,
            log_path: None,
            predict: None,
            seen: 0,
            title: "Тест".into(),
            target: "vegetation".into(),
            features: "xyz".into(),
            segment_start: std::time::Instant::now(),
            epochs: vec![],
            baseline: None,
            n_trials: 0,
            n_epochs: 0,
            predicting: true,
            scenes: vec![],
            scene_progress: None,
            patch_progress: None,
        };
        for l in [
            "=== сцена 1/2: scene_001_cleaned.npy",
            "=== 500000 точек, 136 патчей, xyz, cpu",
            "патч 136/136 208s",
            "=== F1=0.7644 precision=0.6187 recall=1.0000 baseline=0.7624 не покрыто=0 → D:\\x.npy",
            "=== сцена 2/2: scene_021_cleaned.npy",
        ] {
            if let Some((k, n, name)) = super::parse_scene_header(l) {
                job.scene_progress = Some((k, n));
                job.patch_progress = None;
                job.scenes.push((
                    name,
                    super::SceneMetrics {
                        f1: f64::NAN,
                        precision: f64::NAN,
                        recall: f64::NAN,
                        baseline: None,
                        uncovered: 0,
                    },
                ));
            } else if let Some(m) = super::parse_scene_line(l) {
                match job.scenes.last_mut() {
                    Some((_, slot)) => *slot = m,
                    None => job.scenes.push((String::new(), m)),
                }
            } else if let Some((i, p)) = super::parse_patch_line(l) {
                job.patch_progress = Some((i, p));
            }
        }
        assert_eq!(job.scene_progress, Some((2, 2)));
        // Началась вторая сцена — прогресс патчей обнулился (у неё своя сетка патчей).
        assert_eq!(job.patch_progress, None);
        assert_eq!(job.scenes.len(), 2);
        assert_eq!(job.scenes[0].0, "scene_001_cleaned.npy");
        assert_eq!(job.scenes[0].1.f1, 0.7644);
        assert_eq!(job.scenes[0].1.baseline, Some(0.7624));
        // Вторая сцена начата, но метрик ещё нет.
        assert!(job.scenes[1].1.f1.is_nan());
    }
}
