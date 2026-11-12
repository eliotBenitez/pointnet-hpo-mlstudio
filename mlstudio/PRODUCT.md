# Product

<!-- impeccable:product-schema 1 -->

## Platform

desktop

Native desktop app on GPUI (gpui-kit 0.9 / gpui-component), not a web page. Windows and Linux. The app has one look of its own on both systems and does not imitate either OS.

## Users

The project author and their teammates (the "purple" and "grey" teams in `../Вводные для команды фиолетовых.md` and `../Гиперпараметры для серых.md`). Each person runs the same tasks from `../tasks.md` on their own machine, usually one with a GPU. Teammates should be able to use the app without anyone walking them through it.

## Product Purpose

A control panel for the PointNet++ hyperparameter search project (`../hpo_pointnet.py`, `../run_all.py`). It covers the full loop in one window: pick a dataset, configure and start runs, watch them live, and review the results to choose parameters for the final runs. The app works when a person can do all of that without a terminal and without opening CSV/JSON files by hand.

## Positioning

The app is built around this project's actual artifacts: `scene_*.npy` datasets, `runs/<target>_<features>/` with `best.json`, `trials.csv` and `trial_XXXX/metrics.csv`, the per-class tasks №2–7 plus `final`, and the `trial=N epoch=E F1=…` log lines. A generic experiment tracker would need setup or a server. This app reads those files in place and drives the existing scripts.

## Operating Context

- Launching, monitoring and reviewing get roughly equal time. No single activity dominates.
- Training runs take hours (25 trials × 10 epochs, final runs at 30 epochs). People check in on them now and then rather than watching constantly.
- Binary segmentation targets: `ground`, `vegetation`, `noise`, `building`, `water`. Features: `xyz`, `xyz_hag`.
- Metrics: F1, precision and recall of the target class, compared against the "all = 1" baseline F1. The warning threshold is a margin under 0.05.
- Data: `exports*/` directories of 200 scenes each, 200k–1M points per scene, `[N, 9]` float32.
- Python runs from `.venv311`. A run resumes from `study.db` when restarted in the same output directory.

## Capabilities and Constraints

- Datasets: list them and inspect per-scene point counts and sizes. Only `.npy` headers are read; the arrays are never loaded.
- Runs: show the best metrics and parameters, a trials table, and F1 per epoch.
- Launch: a single `hpo_pointnet.py` run or a `run_all.py` task script. All CLI arguments are editable, and a preview shows the exact command.
- Processes: several at once, with live stdout/stderr, stop (kills the whole process tree), restart, copy the log, and live F1 charts parsed from the log.
- Prediction jobs (`predict_scene.py`: model testing and per-scene evaluation) get their own live screen: F1 of the last scene, scene and patch progress, and a table of scenes — not the trials-and-epochs match view, which would stay empty forever.
- If the app is closed, it kills its running training processes rather than leaving them orphaned.
- The UI language is Russian, matching the project docs and log output.
- Undecided: a job queue, comparing runs side by side, and reading `report.md` inside the app.

## Evidence on Hand

Real datasets: `../exports_small`, `../exports_large`. `../runs/` is currently empty, so no real run results exist yet. Never present fabricated metrics as real results.

## Product Principles

1. The file system is the source of truth. The app reads and launches; it does not keep its own database.
2. Show the actual command and the actual log. Nothing is hidden behind the controls.
3. Long runs must be easy to read at a glance: it should be clear within seconds whether a run is alive, how far it has got, and whether it beats the baseline.
4. A teammate who has never seen the app should be able to complete the loop.
