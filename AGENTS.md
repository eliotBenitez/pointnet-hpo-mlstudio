# AGENTS.md

PointNet++ hyperparameter search over LiDAR scenes (Python) plus **ML Studio**, a native GPUI desktop app that drives the same scripts. Docs, UI, logs and report text are Russian — keep it that way.

## Model roles

- **Opus 5.5**: plan complex work, assess architectural decisions, and independently review changes and risks.
- **Sonnet 5.5 / DeepSeek V4.1 Flash 1017**: implement changes, write and run tests, and troubleshoot implementation issues.
- **Haiku 5.5**: search for information, explore files, and handle other small, well-defined tasks.

These are preferred roles, not a requirement to involve multiple models in every task. Use delegation when available and worthwhile; if the preferred model or delegation is unavailable, the current agent should continue and report the limitation. Review and verification remain required regardless of the model used. The human retains responsibility for final decisions.

## Entrypoints

| Path | Role |
|:--|:--|
| `hpo_pointnet.py` | The whole engine: Optuna study, patch dataset, training, `best.json`/`report.md`/`trials.csv` writer. CLI is the real spec — read it before touching anything. |
| `run_all.py` | Orchestrates tasks №2–7 + `final` as subprocesses, logging to `runs/run_all.log`. `run_all.cmd`/`.ps1`/`.sh` only bootstrap `.venv311` and call this file. |
| `summary_report.py` | Joins `runs/*/best.json` → `runs/report.md`. Imports `hpo_pointnet`, so it needs the repo root on `sys.path` (running the script directly is fine). |
| `predict_scene.py` | Whole-scene inference → `.npy` predictions + `scenes__<tag>.json`. **Also embedded in the Rust app** (see gotchas). |
| `test_data_summary.py` | The only test suite (stdlib `unittest`, not pytest). Full run ~50 s incl. a real 2-trial training subprocess. |
| `mlstudio/` | Rust app: `src/main.rs` is the ~4.9k-line UI; `data.rs` parses run artifacts, `cloud.rs` CPU point-cloud preview, `env.rs` Python/venv/torch setup, `monitor.rs` CPU/GPU stats, `scenario.rs` + `scenarios/*.json`, `projects.rs` recent projects. |
| `referat/`, `referat_cross/` | Unrelated DOCX/PPTX generators (`python-docx`/`python-pptx`, **not installed** in `.venv311`). Leave alone unless asked. |

## Commands

```bat
.venv311\Scripts\python.exe -m unittest test_data_summary -v          :: all tests
.venv311\Scripts\python.exe -m unittest test_data_summary.MetricsTests :: one class (module.Class)
.venv311\Scripts\python.exe hpo_pointnet.py --data-dir exports_small_mini --output-dir runs/smoke ^
    --target ground --features xyz --n-trials 1 --epochs 1 --patch-points 1024
.venv311\Scripts\python.exe summary_report.py runs --out runs/report.md
cargo test                 :: in mlstudio/
cargo build --release      :: in mlstudio/, then copy target\release\mlstudio.exe to repo root
```

`run_all.cmd --n-trials 1 --epochs 1` / `run_all.ps1 …` / `bash run_all.sh …` for the full task chain. Pass `--tasks 2 7` etc. to limit it.

## Gotchas that cost time

- **`exports/` and `exports_final/` do not exist here.** Only `exports_small`, `exports_large` (200 scenes each) and `exports_small_mini` (10 scenes). `run_all.py` defaults to `exports/`, so a bare `run_all.cmd` fails immediately; pass `--data`/`--final-data`.
- **Python must be 3.11** (`pyg-lib` has no wheels for 3.12+). Current `.venv311` is **CPU-only** `torch 2.8.0+cpu`, so use `exports_small_mini` for smoke runs.
- `SEARCH_SPACE_VERSION` in `hpo_pointnet.py` and `SPACE_VERSION`/`SPACE` in `mlstudio/src/data.rs` mirror each other. Changing search ranges or training order means bumping the Python constant **and** the Rust mirror, and pointing `--output-dir` at a fresh directory — existing studies hard-fail on a version mismatch. Switching `--loss` likewise requires a new output dir (focal runs go to `*_focal*` dirs).
- `mlstudio` **rewrites** `predict_scene.py` from its compile-time copy whenever the file lacks the substring `model=dict(`. Keep that marker when editing, and rebuild the exe after changing the script so the embedded copy stays current.
- The root `mlstudio.exe` is a copy of `mlstudio/target/release/mlstudio.exe` (identical binaries). Release builds set `windows_subsystem = "windows"`, so `println!` produces nothing — debug in `cargo run`.
- The app's project root = the directory containing `hpo_pointnet.py`; it stores `runs/<name>/studio.json` and `hpo_pointnet.snapshot.py` and flags runs whose script changed after the snapshot.
- **Do not change log line shapes.** `mlstudio/src/data.rs` parses `trial=N epoch=E F1=… baseline=…` and `=== HH:MM:SS name (target, features)`. Progress lines deliberately avoid the `trial=` prefix.
- `run_all.cmd` and `run_all.ps1` must stay **ASCII-only** (cmd/PowerShell 5.1 misparse UTF-8); Python sources are UTF-8 and rely on `PYTHONIOENCODING=utf-8` set by the launchers. Both end with a `pause`/`Read-Host`, so they hang non-interactive shells — call `run_all.py` directly in automation.
- `tests/` and `mlstudio-logs/` are app artifacts (inference results, copied stdout), not test/CI config. `runs/` is currently empty.
- Not a git repo — no undo. Never bulk-delete `exports*`, `runs/`, `referat*`.
- Under PowerShell the unittest exit code can look like 1 because tests log to stderr; trust the `OK`/`FAILED` summary (or wrap in `cmd /c`).

## Conventions worth preserving

- Scene arrays are `[N, 9]` float32: col 0 = class, cols 1–3 = xyz, **cols 4–8 are deliberately ignored** (they leak the label). HAG / `is_above_ground` are recomputed from geometry only.
- Metrics are F1/precision/recall of the target class only, versus the all-ones baseline F1; a `< 0.05` margin is a warning, and mIoU is explicitly not used.
- Class weights are always applied (`CrossEntropyLoss` weight, or focal alpha) — mandatory for `noise` and `water`.
- Splits are by whole scene; patches are seeded by `(seed, epoch, index)` so a resumed run reproduces patches.
- UI changes must follow `mlstudio/DESIGN.md` (colors/semantics, tabular figures, no shadows) and `mlstudio/PRODUCT.md`; never display fabricated metrics.
