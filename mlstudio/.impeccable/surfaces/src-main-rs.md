---
version: 1
slug: "src-main-rs"
primary_target: "src/main.rs"
related_targets: []
---

# ML Studio — главное окно

Scope: the whole app window (src/main.rs): Трансляция (active run), Таблицы (runs), Запуск, Датасеты. Mode: Operate.
Audience/job: the author and teammates run the HPO tasks from tasks.md, monitor them for hours, and compare results. Success: within seconds you can tell whether the run is alive, which trial/epoch it is on, and whether F1 clears baseline + 0.05.
Constraints: GPUI + gpui-component standard controls, Russian UI, light and dark themes, never invent metrics.

## Direction contract

THESIS: Each class is a championship. Trials are rounds, F1 is points, baseline + 0.05 is the relegation line. Refuses the gray SaaS sidebar with metric cards and the W&B panel grid.

OWN-WORLD: The sports page of a newspaper. Light theme: chalk paper #F2F0EA, ink #14171C. Dark theme: ink ground #14171C with chalk text. Pitch green #1F6B4A for the shell rail and "above the line". Scoreboard amber #F2A900 for live. Loss red #C8372D for below the line and errors. Class colours are flat fills on table headers. Tabular figures, condensed bold headings, thin rules, no rounded cards.

STORY: Open the app and the live match is front and centre. Read the score against the line, scan the round table, then stop, restart or start the next run.

FIRST VIEWPORT: A green left rail: Трансляция / Таблицы / Запуск / Датасеты, with a list of processes below. The match header across the top: run name, class-coloured flat strip, huge F1 next to baseline, trial N · epoch E/total, elapsed time and ETA on the right, plus Stop/Restart. Below it, the left two-thirds is the F1-per-epoch chart above the event feed (log). The right third is the trials table with the relegation line and duration bars proportional to seconds.

FORM: Турнирная таблица (sports standings page), grounded candidate 5 of 7, seed key cfe9d4bb. Raises: labanotation (duration bars), zoo map (flat class fill), vertical feed (active run owns the screen), j-card (right-aligned timing and ETA), variable font (live tabular numbers).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
