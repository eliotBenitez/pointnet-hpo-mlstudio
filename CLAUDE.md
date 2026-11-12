# CLAUDE.md

PointNet++ hyperparameter search over LiDAR scenes (Python) + **ML Studio** (Rust/GPUI desktop app). Docs, UI text, and reports are in **Russian** — keep it that way.

See also: [AGENTS.md](AGENTS.md) for full details.

## Commands

```bat
:: Tests (stdlib unittest, not pytest; full run ~50 s including a real 2-trial training subprocess)
.venv311\Scripts\python.exe -m unittest test_data_summary -v
.venv311\Scripts\python.exe -m unittest test_data_summary.MetricsTests   &:: single class

:: Smoke training run
.venv311\Scripts\python.exe hpo_pointnet.py --data-dir exports_small_mini --output-dir runs/smoke ^
    --target ground --features xyz --n-trials 1 --epochs 1 --patch-points 1024

:: Summary report
.venv311\Scripts\python.exe summary_report.py runs --out runs/report.md

:: Rust app
cd mlstudio && cargo test
cd mlstudio && cargo build --release   &:: then copy target\release\mlstudio.exe to repo root
```

Under PowerShell, unittest exit code may appear as 1 (stderr logging); trust the `OK`/`FAILED` summary line.

## Critical gotchas

- **Python 3.11 only** — `pyg-lib` has no wheels for 3.12+. The `.venv311` venv is CPU-only `torch 2.8.0+cpu`.
- **Data directories**: only `exports_small` (200 scenes), `exports_large` (200 scenes), and `exports_small_mini` (10 scenes) exist. `exports/` and `exports_final/` **do not exist**. Use `exports_small_mini` for smoke runs. `run_all.py` defaults to `exports/`, so a bare `run_all.cmd` fails — pass `--data`/`--final-data`.
- **Not a git repo** — no undo. Never bulk-delete `exports*`, `runs/`, `referat*`.
- **`SEARCH_SPACE_VERSION`** (currently 5) in `hpo_pointnet.py` and `SPACE_VERSION`/`SPACE` in `mlstudio/src/data.rs` must stay in sync. Changing search ranges or training order → bump both constants + use a new `--output-dir`. Existing studies hard-fail on version mismatch.
- **Log line format is load-bearing**: `data.rs` parses `trial=N epoch=E F1=… baseline=…` and `=== HH:MM:SS name (target, features)`. Do not change these shapes.
- **`predict_scene.py` marker**: `mlstudio` checks for the substring `model=dict(` and overwrites the script from its embedded copy if missing. Keep that marker. Rebuild the exe after editing.
- **`run_all.cmd` / `run_all.ps1` must be ASCII-only** (cmd.exe/PS 5.1 misparse UTF-8). Python sources are UTF-8; launchers set `PYTHONIOENCODING=utf-8`. Both scripts end with `pause`/`Read-Host` and will hang non-interactive shells — call `run_all.py` directly in automation.
- Release builds set `windows_subsystem = "windows"`, so `println!` produces nothing — use `cargo run` for debugging.

## Architecture

| Path | Role |
|:--|:--|
| `hpo_pointnet.py` | Main engine: Optuna study, patch dataset, training, results writer. **CLI is the real spec.** |
| `run_all.py` | Orchestrates tasks №2–7 + `final` as subprocesses. |
| `summary_report.py` | Joins `runs/*/best.json` → `runs/report.md`. Imports `hpo_pointnet`. |
| `predict_scene.py` | Whole-scene inference. Also embedded in the Rust app. |
| `test_data_summary.py` | The only test suite. |
| `mlstudio/` | Rust (GPUI) app: `main.rs` ~4.9k lines UI; `data.rs` parses run artifacts; `cloud.rs` point-cloud preview; `env.rs` Python/venv setup; `monitor.rs` CPU/GPU stats; `scenario.rs` + `scenarios/*.json`; `projects.rs` recent projects. |
| `referat/`, `referat_cross/` | Unrelated DOCX/PPTX generators. Leave alone unless asked. |
| `tests/`, `mlstudio-logs/` | App artifacts (inference results, copied stdout), not test config. |

## Data conventions

- Scene arrays: `[N, 9]` float32 — col 0 = class, cols 1–3 = xyz. **Cols 4–8 are deliberately ignored** (they leak the label). HAG / `is_above_ground` are recomputed from geometry only.
- Metrics: F1/precision/recall of the target class only, versus all-ones baseline F1. Margin < 0.05 → warning. mIoU is explicitly not used.
- Class weights always applied (`CrossEntropyLoss` weight or focal alpha) — mandatory for `noise` and `water`.
- Splits by whole scene; patches seeded by `(seed, epoch, index)`.

## UI conventions

Changes to mlstudio must follow `mlstudio/DESIGN.md` and `mlstudio/PRODUCT.md`. Never display fabricated metrics.
