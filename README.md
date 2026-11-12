# PointNet++: подбор гиперпараметров по классам на локальных патчах

ТЗ и распределение задач — [tasks.md](tasks.md). Каждый прогон — бинарная сегментация: целевой класс = 1, всё остальное = 0.

| `--target` | Задача | Исходные классы → 1 |
|:---|:---|:---|
| `ground` | №2 Грунт | 2 |
| `vegetation` | №3 Растительность | 3, 4, 5 |
| `noise` | №4 Ложноотражённые точки | 7, 18 |
| `building` | №5 Здания | 6 |
| `water` | №6 Вода | 9 |

## Запуск

**Все задачи одной командой** ([run_all.py](run_all.py): №2–6 подбор, финальные прогоны и сводный отчёт
№7; задача `final` требует существующий каталог `--final-data` — по умолчанию `exports_final/` — и без
него останавливается с ошибкой). Live-вывод дублируется в `runs/run_all.log`; прерванный запуск
продолжается повторным запуском.

```bat
run_all.cmd                              :: Windows: сам создаст .venv311 и поставит зависимости
run_all.cmd --n-trials 1 --epochs 1      :: быстрая проверка
```

**Отдельные задачи** — флаг `--tasks` (можно несколько, выполняются в порядке: подбор → final → отчёт):

| `--tasks` | Что делает |
|:---|:---|
| `2` … `6` | подбор для класса (№2 грунт: `ground_xyz` и `ground_xyz_hag`; №3 растительность: оба; №4–6: `xyz`) |
| `7` | сводный отчёт `runs/report.md` по всем готовым прогонам |
| `final` | финальные прогоны на `exports_final/` с лучшими параметрами (`--fixed-params`, 1 trial, `--final-epochs` эпох, каталог `final_<имя>`); для выбранных 2–6, а если их нет — для всех, где есть `best.json`. Каталог данных обязан существовать; params-файл и каталог `final_<имя>` не перезаписываются |

```bat
run_all.cmd --tasks 2                    :: только грунт
run_all.cmd --tasks 3 4                  :: растительность и шум
run_all.cmd --tasks 2 final 7            :: грунт, его финальный прогон и отчёт
run_all.cmd --tasks 7                    :: только пересобрать отчёт
```

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1 --n-trials 1 --epochs 1   # то же из PowerShell
```

```bash
bash run_all.sh --n-trials 1 --epochs 1  # Git Bash / Linux
```

`run_all.cmd` ищет Python 3.11 через `py` (python.org или uv) и ставит `torch 2.8.0` под CUDA, если есть
`nvidia-smi`, иначе CPU-сборку.

**Отдельный прогон:**

```bash
python hpo_pointnet.py --data-dir exports --output-dir runs/ground_xyz --target ground --features xyz --n-trials 25 --epochs 10
python summary_report.py runs --out runs/report.md
```

Окружение: Python 3.11 (`.venv311`; на 3.14 нет колеса `pyg-lib`), `torch==2.8.0`, `pyg-lib` под ту же версию torch,
`torch-geometric`, `torch-pointcloud==0.0.16`, `optuna`, `numpy`. Один прогон = один `--output-dir`; повторный запуск
с тем же каталогом продолжает study (`study.db`). Продолжение возможно только с теми же данными, seed, эпохами,
`--patch-points`, `--patches-per-scene`, target/features и loss — иначе study откажется продолжаться до записи
артефактов (`workers` менять можно).

## Данные

`exports/scene_XXX_cleaned.npy`: 200 сцен по 200 тыс. – 1 млн точек, массив `[N, 9]`. Используются колонки
`class` (0) и `x, y, z` (1–3). Остальные (4–8) игнорируются: в прошлой выгрузке такая колонка вычислялась из метки
класса и давала утечку. HAG и `is_above_ground` считаются заново только по геометрии.

## Пайплайн (задача №1)

- **Разбиение** Train/Val 80/20 по целым сценам до нарезки патчей (`split_scenes`), без фильтрации сцен.
- **Патч** (`PatchDataset`): случайная точка-центр; точки в круге по XY. Начальный радиус рассчитан на 16 384 точки
  при средней плотности сцены; не хватает — радиус ×1.25, много — случайное прореживание до 16 384.
  Координаты центрируются и делятся на радиус патча, поэтому `radius_l0` задан в долях патча.
- **FPS первого SA-слоя** (`FpsCache`) считается один раз на патч (4096 точек) батчем на устройстве модели и
  кешируется: патчи одинаковы во всех trials, а FPS жадный, поэтому для любого `n_points_l0` берётся префикс.
  Шаг обучения на CPU быстрее ≈2×. При `--n-trials 1` (финальные прогоны) порядок той же схемы считается
  без хранения. Смена версии study (`SEARCH_SPACE_VERSION`) требует новой `--output-dir` / `--runs`.
- **Батч** 2–4 патча (подбирает Optuna), ≈32–64 тыс. точек.
- **Признаки** `--features xyz` или `xyz_hag` (+ HAG: z минус минимум z в ячейке 1 м и 8 соседних;
  `is_above_ground` = HAG > 0.5 м).
- **Потери**: `CrossEntropyLoss` с весами классов по Train (всегда; обязательны для `noise` и `water`).
- **LR**: `ReduceLROnPlateau` по F1 на Val (×0.5, patience 1).
- **Pruning**: `MedianPruner` — после 5 полных trials trial с F1 ниже медианы на той же эпохе обрывается,
  начиная с эпохи `max(3, epochs // 3) + 1` (4-я при 8–10 эпохах).
- **Метрики**: F1, Precision, Recall только целевого класса; baseline — F1 предсказания «всё = 1». Если модель
  обгоняет baseline меньше чем на 0.05, в отчёте предупреждение.
- **Seed** (по умолчанию 42) фиксирует разбиение, инициализацию, центры и прореживание патчей, порядок батчей,
  `TPESampler`. Условия запуска (seed, версии, устройство, веса классов) пишутся в `split.json` и `best.json`.

## Формат результатов

```
runs/<target>_<features>/
  best.json        # лучшие параметры и метрики + условия запуска (читает summary_report.py)
  report.md        # топ-5, важность параметров, предупреждения
  trials.csv       # number,state,f1,precision,recall,baseline_f1,seed,target,features,checkpoint,error,params,seconds (длительность trial)
  split.json       # списки Train/Val + условия запуска
  data_summary.md
  study.db
  trial_XXXX/
    config.json    # параметры trial
    metrics.csv    # epoch,f1,precision,recall,lr,seconds (обучение + валидация эпохи)
    metrics.json   # лучшая эпоха: f1, precision, recall, baseline_f1, val_positive_points
    best.pt
runs/report.md     # сводный отчёт (задача №7): таблица на класс, сравнение xyz/xyz_hag, веса классов
```

## Финальные прогоны

`--fixed-params params.json` (например, `params` из `best.json`) обучает ровно эту конфигурацию первым trial:

```bash
python hpo_pointnet.py --data-dir exports_final --output-dir runs/final_ground_xyz --target ground --fixed-params ground.params.json --n-trials 1 --epochs 30
```
