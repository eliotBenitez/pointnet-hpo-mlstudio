# ML Studio Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ML Studio support the full reproducible pipeline: HPO selection, final training on a separate dataset, and report generation, without mixing incompatible studies.

**Architecture:** Keep the existing GPUI form, file-based artifacts, `run_all.py`, and scenario system. Add structured inputs and shared command construction in the existing `Studio`; add compatibility metadata checks in Python so CLI and GUI enforce the same rule. Add a dedicated final-run action on the selected run, using `best.json` and `--fixed-params`.

**Tech Stack:** Rust, GPUI/gpui-kit, Python 3.11, Optuna, stdlib `unittest`, Cargo tests.

**Spec:** `docs/superpowers/specs/2026-11-12-mlstudio-stage1-design.md`

## Global Constraints

- Keep Russian UI, docs, logs, and report text.
- Preserve existing log line shapes parsed by `mlstudio/src/data.rs`.
- Keep the filesystem as the source of truth; do not add a project database.
- Do not modify `run_all.cmd` or `run_all.ps1`.
- Preserve old artifacts and make missing new fields explicit.
- `final` requires a non-empty final-data directory in the GUI.
- Incompatible study continuation is blocked; `workers` may change.
- The final run uses `best.json`, `--fixed-params`, one trial, and `final_<name>`.

## Review Focus

- A `run_all.py` command must use the GUI-selected final dataset and runs directory, not defaults; test command arguments in Python/Rust logic.
- Focal Loss settings must reach `run_all.py` and child HPO processes; test the generated command and CLI parsing.
- Patch size/count must be passed, persisted, and restored; test both new and old artifacts.
- An existing study with changed data/seed/epochs/target/features/loss/patch settings must fail before adding trials; test the compatibility guard.
- A final run must never reuse the source study directory; test generated params path and `final_<name>` output.

---

### Task 1: Add structured pipeline options to `run_all.py`

**Files:**
- Modify: `run_all.py:30-107`
- Test: `test_data_summary.py` (new CLI/compatibility test class)

**Interfaces:**
- Consumes: existing `argparse` values and `run()` helper.
- Produces: `run_all.py` options `--patch-points` and `--patches-per-scene`; every HPO child command receives both values, including final fixed-parameter runs.

- [ ] **Step 1: Write failing tests**
  - Add a test that parses `--patch-points 1024 --patches-per-scene 3` through the parser/helper boundary, or executes a minimal mocked `run_all` command and asserts both options are present in child HPO arguments.
  - Add a test that `--loss focal --focal-gamma 1.5` remains present in the generated child command.
- [ ] **Step 2: Run the focused test and verify it fails**
  - Run: `.venv311\Scripts\python.exe -m unittest test_data_summary.RunAllTests -v`
- [ ] **Step 3: Implement the two arguments and forward them**
  - Add positive integer arguments with defaults matching `hpo_pointnet.py` (`16384`, `2`).
  - Extend `run()` command construction with `--patch-points` and `--patches-per-scene` before `*extra`; final runs inherit them.
- [ ] **Step 4: Run the focused test and verify it passes**
- [ ] **Step 5: Commit**

### Task 2: Enforce study compatibility in `hpo_pointnet.py`

**Files:**
- Modify: `hpo_pointnet.py:615-629, 778-800`
- Test: `test_data_summary.py`

**Interfaces:**
- Consumes: `run_info` fields already written to `split.json`/`best.json`.
- Produces: a compatibility guard for existing Optuna studies, comparing all study-defining fields except `workers`.

- [ ] **Step 1: Write failing tests**
  - Add tests for a pure helper, e.g. `study_compatibility_error(study_attrs, run_info)`, asserting no error for equal metadata and an error for each changed `data_dir`, `seed`, `epochs`, `target`, `features`, `patch_points`, `patches_per_scene`, and `loss`.
  - Assert missing compatibility metadata in an existing study is treated as incompatible rather than silently accepted.
- [ ] **Step 2: Run the focused tests and verify they fail**
- [ ] **Step 3: Implement the helper and wire it into study creation**
  - Persist a JSON-safe `compatibility` attribute after the first run.
  - Keep the existing search-space check and loss error wording compatible with current users where practical.
  - Use normalized path strings consistently; do not include `workers` or trial count in compatibility.
  - Reject an existing study with missing or changed metadata before optimization.
- [ ] **Step 4: Run focused tests and the existing Python suite**
  - Run: `.venv311\Scripts\python.exe -m unittest test_data_summary -v`
- [ ] **Step 5: Commit**

### Task 3: Add structured fields and paths to the ML Studio state and command builder

**Files:**
- Modify: `mlstudio/src/main.rs:402-627, 880-915`
- Test: `mlstudio/src/main.rs` unit tests if existing test patterns permit; otherwise verify through `cargo test` and a focused command-builder helper test.

**Interfaces:**
- Consumes: current `InputState` fields and `run_all.py` CLI.
- Produces: state fields for `final_data_dir`, `runs_dir`, `patch_points`, and `patches_per_scene`; `command()` emits the selected paths and values for single and run-all modes.

- [ ] **Step 1: Add/adjust a focused command-construction test**
  - Assert a run-all command contains `--data`, `--final-data`, `--runs`, `--loss`, `--focal-gamma`, `--patch-points`, and `--patches-per-scene` with the entered values.
  - Assert a single-run command uses the selected runs directory in `--output-dir` and includes patch arguments.
- [ ] **Step 2: Run the focused Rust test and verify it fails**
- [ ] **Step 3: Implement state initialization and command construction**
  - Default final data to `exports_final`, runs to `runs`, patch points to `16384`, patches per scene to `2`.
  - Pass `--final-data` and `--runs` in run-all mode.
  - Pass `--patch-points` and `--patches-per-scene` in both modes.
  - Keep loss/gamma handling single-source and ensure run-all receives it.
  - Normalize relative paths against the project root only for validation/conflict checks; preserve clear paths in preview.
- [ ] **Step 4: Run `cargo test`**
- [ ] **Step 5: Commit**

### Task 4: Add GUI validation and restore all run-defining parameters

**Files:**
- Modify: `mlstudio/src/main.rs:1250-1296, 2960-3208`
- Modify: `mlstudio/src/data.rs` only if run metadata accessors are needed.
- Test: Rust unit tests or focused manual smoke test through debug build.

**Interfaces:**
- Consumes: Task 2 compatibility fields and Task 3 inputs.
- Produces: validation that blocks `final` without final data; validation of positive patch values and focal gamma; resume behavior that restores or explicitly reports missing metadata.

- [ ] **Step 1: Write failing tests/checks**
  - Assert final task plus blank final-data produces a blocking Russian error.
  - Assert non-positive patch values block launch.
  - Assert a run containing patch metadata restores both fields.
  - Assert an old run without patch metadata reports unknown values rather than silently treating them as compatible.
- [ ] **Step 2: Run checks and verify they fail**
- [ ] **Step 3: Implement validation and resume logic**
  - Add fields to the form in the existing visual style.
  - In `form_from_run()`, restore `patch_points`, `patches_per_scene`, and all other run-defining values available in artifacts.
  - Track unknown fields separately from default values.
  - Block starting an existing study when a run-defining field differs or is unknown; offer returning to stored values or changing the output name/catalog.
  - Permit workers changes.
  - Ensure the exact blocked reason appears beside the command and disables `Запустить`.
- [ ] **Step 4: Run `cargo test` and a debug GUI smoke check**
- [ ] **Step 5: Commit**

### Task 5: Implement the dedicated final-run action

**Files:**
- Modify: `mlstudio/src/main.rs:2560-2660` and related form/job helpers.
- Test: Rust unit/helper tests for generated final command and params path.

**Interfaces:**
- Consumes: selected `Run`, its `best.json`, final-data and runs-dir inputs, and existing `spawn()`.
- Produces: a `Финальный прогон` action that writes `<runs>/<name>.params.json`, launches `hpo_pointnet.py` with `--fixed-params`, `--n-trials 1`, final epochs, and output `final_<name>`.

- [ ] **Step 1: Write failing tests**
  - Assert the generated command contains the source run's target/features, `--fixed-params`, one trial, final epochs, selected final data, selected runs directory, and output directory `final_<name>`.
  - Assert missing/invalid `best.json` or final data blocks the action.
- [ ] **Step 2: Run focused tests and verify failure**
- [ ] **Step 3: Implement action**
  - Reuse existing JSON parsing and file-writing conventions.
  - Write params atomically enough to avoid a half-written params file on error.
  - Prevent conflicts with active jobs and prevent output collision with the source run.
  - Show the exact command in the resulting job/log view.
- [ ] **Step 4: Run `cargo test` and a one-trial/one-epoch smoke run using `exports_small_mini`**
- [ ] **Step 5: Commit**

### Task 6: Integration verification and documentation consistency

**Files:**
- Modify: `mlstudio/PRODUCT.md` only where the implemented capabilities/undecided list is now stale.
- Modify: `README.md` only if the GUI behavior needs documenting.

- [ ] **Step 1: Run the full Python test suite**
  - `.venv311\Scripts\python.exe -m unittest test_data_summary -v`
- [ ] **Step 2: Run Rust tests**
  - `cargo test` in `mlstudio/`
- [ ] **Step 3: Run a minimal pipeline smoke test**
  - Use `exports_small_mini`, one trial, one epoch, and a temporary runs directory; verify `split.json`, `best.json`, and final command metadata.
- [ ] **Step 4: Check the GUI diff against `DESIGN.md` and log-shape constraints**
- [ ] **Step 5: Update stale product docs and commit**

## Self-review

Coverage checked against the spec: CLI options, structured GUI fields, artifact restoration, Python and GUI compatibility checks, mandatory final-data validation, dedicated final-run action, fixed-parameter final output, old-artifact behavior, and verification commands are all assigned above. No Markdown viewer or process-tree work is included because those belong to later stages.
