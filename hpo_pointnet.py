#!/usr/bin/env python3
"""Подбор PointNet++ для бинарной сегментации сцен LiDAR на локальных патчах.

Сцена — .npy [N, >=4]: колонка 0 = класс, колонки 1–3 = x, y, z; остальные колонки не используются
(они могут быть выведены из метки класса и дать утечку). Целевой класс = 1, всё остальное = 0.
Проверено по API torch-pointcloud v0.0.16. Нужны Python >=3.10,
PyTorch >=2.8, torch-geometric >=2.8, torch-pointcloud==0.0.16, Optuna, NumPy.
"""
import argparse
import csv
import json
import logging
import math
import platform
import random
import shutil
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

LOG = logging.getLogger("hpo_pointnet")
MODEL_NAME = "pointnet2.s3dis-area1.openpoints"
# Повышать при любом изменении диапазонов в run_trial или обучения, делающем trials несравнимыми
# (5: кеш FPS со своим случайным стартом — другой поток RNG, чем у прежних trials).
SEARCH_SPACE_VERSION = 5
FIELDS = ["number", "state", "f1", "precision", "recall", "baseline_f1", "seed", "target",
          "features", "checkpoint", "error", "params", "seconds"]
EPOCH_FIELDS = ["epoch", "f1", "precision", "recall", "iou", "train_f1", "train_precision",
                "train_recall", "train_iou", "lr", "seconds"]
LABEL_COL = 0  # раскладка exports: class, x, y, z, ...
XYZ = slice(1, 4)
XY = slice(1, 3)
CLASS_NAMES = {
    0: "Не определено", 1: "Не классифицировано", 2: "Поверхность", 3: "Низкая растительность",
    4: "Средняя растительность", 5: "Высокая растительность",
    6: "Здание", 7: "Низкий шум", 9: "Вода", 18: "Высокий шум",
}
CLASS_HEADERS = ["Неопр.", "Не клас.", "Поверх.", "Низк.раст.", "Сред.раст.",
                 "Выс.раст.", "Здание", "Шум", "Вода", "Выс.шум"]
# Цель прогона -> ID исходных классов, которые становятся меткой 1.
TARGETS = {
    "ground": (2,), "vegetation": (3, 4, 5), "noise": (7, 18), "building": (6,), "water": (9,),
}
TARGET_TITLES = {"ground": "Грунт", "vegetation": "Растительность",
                 "noise": "Ложноотражённые точки", "building": "Здания", "water": "Вода"}
FEATURE_CHANNELS = {"xyz": 3, "xyz_hag": 5}
GROUND_CELL = 1.0  # м: при ~100 точках/м² в ячейке хватает точек, чтобы найти землю
ABOVE_GROUND_THRESHOLD = 0.5  # м
BASELINE_MARGIN = 0.05  # ТЗ: модель должна обгонять «всё = 1» хотя бы на 5 п.п. F1
N_POINTS_L0_MAX = 4096  # верх n_points_l0 = длина закешированного порядка FPS


def generate_data_summary(data_dir):
    """Три таблицы по всем исходным сценам."""
    paths = sorted(Path(data_dir).glob("*.npy"))
    if not paths:
        raise ValueError(f"Нет файлов .npy в {data_dir}")
    rows = []
    names = dict(CLASS_NAMES)
    headers = dict(zip(CLASS_NAMES, CLASS_HEADERS))
    totals = {class_id: 0 for class_id in CLASS_NAMES}
    presence = {class_id: 0 for class_id in CLASS_NAMES}
    for path in paths:
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
        except ValueError as exc:
            raise ValueError(f"Файл повреждён или скопирован не полностью: {path} ({exc})") from exc
        if arr.ndim != 2 or arr.shape[1] < 4 or len(arr) == 0:
            raise ValueError(f"Ожидается непустой массив [N,>=4]: {path}")
        if not np.isfinite(arr[:, :4]).all():
            raise ValueError(f"NaN/Inf в {path}")
        labels, frequencies = np.unique(arr[:, LABEL_COL], return_counts=True)
        if np.any(labels != np.round(labels)):
            raise ValueError(f"Нецелый ID класса в {path}")
        counts = dict(zip([int(label) for label in labels], [int(n) for n in frequencies]))
        extent = np.ptp(arr[:, XYZ], axis=0)
        rows.append((path.stem, len(arr), counts, extent))
        for class_id, n in counts.items():
            if class_id not in names:
                LOG.warning("Неизвестный ID класса %s в %s", class_id, path.name)
                names[class_id] = f"Класс {class_id}"
                headers[class_id] = f"ID {class_id}"
                totals[class_id] = presence[class_id] = 0
            totals[class_id] += n
            presence[class_id] += 1
    names = {k: v for k, v in sorted(names.items()) if totals[k]}  # только встречающиеся классы

    lines = [
        "## Таблица 1. Сводка по сценам", "",
        "Размеры вычислены как max−min в единицах исходных координат.", "",
        "| Сцена | Точек | Классов | Размер (x × y × z) |",
        "|:---|---:|---:|---:|",
    ]
    for name, count, classes, extent in rows:
        dims = " × ".join(f"{value:.1f}" for value in extent)
        lines.append(f"| {name} | {count} | {len(classes)} | {dims} |")
    lines += ["", "## Таблица 2. Доля классов по сценам (%)", "",
              "| Сцена | " + " | ".join(headers[class_id] for class_id in names) + " | Итого |",
              "|:---|" + "---:|" * (len(names) + 1)]
    for name, count, classes, _ in rows:
        shares = [f"{100 * classes[class_id] / count:.2f}" if class_id in classes else "—"
                  for class_id in names]
        lines.append(f"| {name} | " + " | ".join(shares) + " | 100.00 |")
    grand_total = sum(totals.values())
    lines += ["", "## Таблица 3. Сводка по классам (все сцены)", "",
              "| Класс | ID | Точек всего | Доля (%) | Сцен с классом |",
              "|:---|---:|---:|---:|---:|"]
    for class_id, name in names.items():
        lines.append(f"| {name} | {class_id} | {totals[class_id]} | "
                     f"{100 * totals[class_id] / grand_total:.3f} | "
                     f"{presence[class_id]} / {len(rows)} |")
    return "\n".join(lines) + "\n"


def load_scenes(data_dir):
    """Все сцены участвуют: сцены без целевого класса — честные отрицательные примеры.

    Формат файлов проверяет generate_data_summary, она вызывается в main раньше.
    """
    paths = sorted(Path(data_dir).glob("*.npy"))
    if len(paths) < 2:
        raise ValueError(f"Для Train/Val нужны хотя бы две сцены .npy в {data_dir}")
    return paths


def split_scenes(paths, seed):
    """Train/Val по целым сценам, до нарезки патчей."""
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    n_val = max(1, math.ceil(0.2 * len(paths)))
    return paths[n_val:], paths[:n_val]


def seed_everything(seed):
    """Единая точка фиксации seed: random, NumPy, PyTorch (CPU/CUDA).

    Хеширование строк текущего процесса не фиксируется: PYTHONHASHSEED читается при старте
    интерпретатора и действует только на дочерние процессы (воркеры DataLoader).
    """
    import os
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # нужно для детерминизма cuBLAS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: часть операций PyG/CUDA не имеет детерминированной версии.
    torch.use_deterministic_algorithms(True, warn_only=True)


def ground_grid(xyz, cell=GROUND_CELL):
    """Сетка высот земли только по геометрии (метки не используются).

    Земля в ячейке = минимум z по ней и 8 соседним. shortcut: грубая оценка,
    на крутых склонах HAG завышается; при необходимости заменить на CSF/PMF.
    """
    origin = xyz[:, :2].min(0)
    ij = np.floor((xyz[:, :2] - origin) / cell).astype(np.int64)
    h, w = ij.max(0) + 1
    grid = np.full((h + 2, w + 2), np.inf, dtype=np.float32)
    np.minimum.at(grid, (ij[:, 0] + 1, ij[:, 1] + 1), xyz[:, 2])
    ground = np.min([grid[di:di + h, dj:dj + w] for di in range(3) for dj in range(3)], axis=0)
    return origin, ground


def hag_at(xyz, origin, ground, cell=GROUND_CELL):
    ij = np.floor((xyz[:, :2] - origin) / cell).astype(np.int64)
    ij = np.clip(ij, 0, np.array(ground.shape) - 1)
    return np.maximum(xyz[:, 2] - ground[ij[:, 0], ij[:, 1]], 0).astype(np.float32)


def compute_hag(xyz, cell=GROUND_CELL):
    return hag_at(xyz, *ground_grid(xyz, cell), cell)


def patch_radius(xy, n_points):
    """Радиус круга на n_points точек при средней плотности сцены (общий для обучения и predict)."""
    area = float(np.prod(np.maximum(np.ptp(xy, axis=0), 1e-3)))
    return max(math.sqrt(n_points * area / (math.pi * len(xy))), 1e-3)


def grow_circle(d2, r, need):
    """Индексы точек в круге (d2 — квадраты расстояний); радиус растёт ×1.25, пока точек < need."""
    idx = np.flatnonzero(d2 <= r * r)
    while len(idx) < need:
        r *= 1.25
        idx = np.flatnonzero(d2 <= r * r)
    return idx, r


class PatchDataset:
    """Патч = точки в круге (по XY) вокруг случайного центра, метки 0/1.

    Начальный радиус рассчитан на n_points при средней плотности сцены; если точек мало,
    радиус растёт, если много — случайное прореживание до n_points. Центр и прореживание
    зависят только от (seed, epoch, index); у val epoch не меняется, и патчи фиксированы.
    Координаты патча центрируются и делятся на радиус, поэтому radius_l0 задан в долях патча.
    """

    def __init__(self, paths, target_ids, n_points=16384, patches_per_scene=2, seed=42,
                 features="xyz"):
        if features not in FEATURE_CHANNELS:
            raise ValueError(f"features должен быть одним из {list(FEATURE_CHANNELS)}")
        self.paths = list(paths)
        self.target_ids = list(target_ids)
        self.n_points = n_points
        self.patches_per_scene = patches_per_scene
        self.seed = seed
        self.features = features
        self.epoch = 0
        # Сетки земли — один раз в основном процессе: воркеры DataLoader получают их готовыми.
        self._ground = [ground_grid(np.asarray(np.load(p, mmap_mode="r", allow_pickle=False)[:, XYZ],
                                               dtype=np.float32))
                        for p in self.paths] if features == "xyz_hag" else None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.paths) * self.patches_per_scene

    def __getitem__(self, index):
        import torch
        scene = index // self.patches_per_scene
        arr = np.load(self.paths[scene], mmap_mode="r", allow_pickle=False)
        rng = np.random.default_rng([self.seed, self.epoch, index])
        xy = np.asarray(arr[:, XY], dtype=np.float32)
        n = len(xy)
        center = xy[rng.integers(n)]
        d2 = ((xy - center) ** 2).sum(1)
        idx, r = grow_circle(d2, patch_radius(xy, self.n_points), min(self.n_points, n))
        if len(idx) > self.n_points:
            idx = np.sort(rng.choice(idx, self.n_points, replace=False))
        xyz = np.asarray(arr[idx, XYZ], dtype=np.float32)
        shift = np.array([center[0], center[1], xyz[:, 2].min()], dtype=np.float32)
        feat = (xyz - shift) / r
        if self.features == "xyz_hag":
            hag = hag_at(xyz, *self._ground[scene])
            feat = np.column_stack((feat, hag / r, hag > ABOVE_GROUND_THRESHOLD))
        y = np.isin(np.asarray(arr[idx, LABEL_COL]), self.target_ids).astype(np.int64)
        return torch.from_numpy(feat.astype(np.float32)), torch.from_numpy(y)  # колонки 0–2 — координаты


class Keyed:
    """Обёртка датасета: к патчу добавляется ключ (epoch, index, точек) для FpsCache.

    Длина патча задана (epoch, index) и в ключе нужна, чтобы границы патчей считались на CPU.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        feat, y = self.dataset[index]
        return feat, y, (self.dataset.epoch, index, len(y))


class FpsCache:
    """Порядок FPS первого SA-слоя на n точек для каждого патча, общий для всех trials.

    FPS жадный: первые k индексов порядка на n точек совпадают с FPS на k точек, а патч зависит
    только от (epoch, index). Считается батчем на устройстве модели, в основном процессе (работает
    и с DataLoader-воркерами). random_start: старт — случайная точка патча от (seed, epoch, index),
    как random_start модели в train; иначе первая точка, как у модели в eval.
    keep=False (один trial — переиспользовать нечего): порядок той же схемы, но ровно на нужное
    число точек и без хранения, поэтому режим обучения одинаков при любом --n-trials.
    """

    def __init__(self, n, random_start, seed, keep=True):
        self.n, self.random_start, self.seed, self.keep = n, random_start, seed, keep
        self.orders = {}  # ключ -> локальные индексы [n], int32 на CPU (не занимают память GPU)

    def __call__(self, pos, keys, n):
        """Глобальные индексы [патчей, n] первых n точек FPS каждого патча батча."""
        import torch
        if self.keep and n > self.n:
            raise ValueError(f"n_points_l0={n} больше длины кеша FPS {self.n}")
        sizes = [k[2] for k in keys]
        starts = np.cumsum([0] + sizes[:-1]).tolist()  # границы патчей на CPU, без синхронизаций GPU
        todo = [b for b, k in enumerate(keys) if not self.keep or k not in self.orders]
        if todo:
            from torch_pointcloud.ops.cluster import fps
            m = self.n if self.keep else n
            # Сдвиг каждого патча по кругу делает стартовой точкой выбранную: FPS стартует с первой.
            perm = torch.cat([
                (torch.arange(sizes[b], device=pos.device) +
                 (int(np.random.default_rng([self.seed, *keys[b][:2], 1]).integers(sizes[b]))
                  if self.random_start else 0)) % sizes[b] + starts[b]
                for b in todo])
            sub_batch = torch.repeat_interleave(
                torch.arange(len(todo), device=pos.device),
                torch.tensor([sizes[b] for b in todo], device=pos.device))
            idx = perm[fps(pos[perm], sub_batch, num_nodes=m, random_start=False)].view(len(todo), m)
            if not self.keep:
                return idx  # todo = весь батч, индексы уже глобальные
            local = (idx - torch.tensor([starts[b] for b in todo], device=pos.device)[:, None]).cpu()
            for b, row in zip(todo, local.to(torch.int32)):
                self.orders[keys[b]] = row.clone()  # clone: не держать весь [патчей, n]
        order = torch.stack([self.orders[k][:n] for k in keys]).to(pos.device).long()
        return order + torch.tensor(starts, device=pos.device)[:, None]


def class_counts(paths, target_ids):
    counts = np.zeros(2, dtype=np.int64)
    for p in paths:
        labels = np.load(p, mmap_mode="r", allow_pickle=False)[:, LABEL_COL]
        count1 = int(np.count_nonzero(np.isin(labels, target_ids)))
        counts += (len(labels) - count1, count1)
    return counts


def class_weights(counts):
    """Веса для CrossEntropyLoss по Train (среднее = 1): редкий целевой класс весит больше."""
    if not np.all(counts):
        raise ValueError("В Train отсутствует один из классов")
    return counts.sum() / (2 * counts.astype(np.float64))


def make_criterion(name, weights, gamma=2.0):
    """CrossEntropyLoss или Focal Loss (Lin et al., 2017) с теми же весами классов как alpha."""
    import torch
    import torch.nn.functional as F
    if name == "ce":
        return torch.nn.CrossEntropyLoss(weight=weights)

    def focal(logits, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        w = weights[y]
        # Нормировка как у взвешенного CE (mean): при gamma=0 совпадает с CrossEntropyLoss.
        return (w * (1 - torch.exp(-ce)) ** gamma * ce).sum() / w.sum()
    return focal


def collate_flat(items):
    import torch
    pos = torch.cat([item[0] for item in items])
    y = torch.cat([item[1] for item in items])
    batch = torch.cat([torch.full((len(item[1]),), i, dtype=torch.long)
                       for i, item in enumerate(items)])
    keys = [item[2] for item in items] if len(items[0]) > 2 else None  # есть только у Keyed
    return pos, batch, y, keys


def forward(model, feat, batch, keys=None, cache=None):
    """Модель на батче; с cache первый SA-слой берёт префикс закешированного FPS."""
    pos = feat[:, :3].contiguous()
    if cache is None or keys is None:
        return model(feat, pos, batch)
    first = model.encoder.sa_blocks[0]
    idx = cache(pos, keys, first.num_points).reshape(-1)
    # Хук живёт один вызов: у модели не остаётся состояния, и прямой model(...) считает FPS сам.
    handle = first.register_forward_pre_hook(
        lambda module, args, kwargs: (args, dict(kwargs, idx=idx)), with_kwargs=True)
    try:
        return model(feat, pos, batch)
    finally:
        handle.remove()


def make_loader(dataset, batch_size, shuffle, seed, workers=0):
    from torch.utils.data import DataLoader
    import torch
    generator = torch.Generator().manual_seed(seed)
    # Воркеры не ломают воспроизводимость: случайность патча задана (seed, epoch, index).
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      collate_fn=collate_flat, num_workers=workers)


def build_model(params, in_channels=3):
    from torch_pointcloud.models import create_model, model_info
    # Registry задаёт 4 блока; переопределяем согласованные encoder/decoder списки.
    base = model_info(MODEL_NAME, task="semantic-segmentation")["hparams"]
    depth = params["n_sa_layers"]
    channels = base["sa_channels"][:depth]
    r = params["radius_l0"]
    k = params["nsample"]
    if params["use_MSG"]:
        # Две шкалы с той же суммарной шириной выходного блока.
        channels = [[[max(8, c // 2) for c in layer]] * 2 for layer in channels]
        radii = [(r * 2 ** i, r * 2 ** (i + 1)) for i in range(depth)]
        neighbors = [(max(8, k // 2), k) for _ in range(depth)]
    else:
        radii = [r * 2 ** i for i in range(depth)]
        neighbors = [k] * depth
    model = create_model(
        MODEL_NAME, task="semantic-segmentation", pretrained=False,
        in_channels=in_channels, num_classes=2, sa_channels=channels,
        fp_channels=base["fp_channels"][-depth:],
        ratios=base["ratios"][:depth], radii=radii,
        num_neighbors=neighbors, dropout=params["dropout"],
        normalize_pos=False,
    )
    # У версии 0.0.16 сегментационный конструктор принимает ratios, но не num_points.
    # Первый SA-блок поддерживает абсолютное num_points; остальные оставляем по ratios.
    first = model.encoder.sa_blocks[0]
    first.ratio = None
    first.num_points = params["n_points_l0"]
    return model


def binary_metrics(tp, fp, fn, n_neg):
    """F1/Precision/Recall только целевого класса и F1 константы «всё = 1» (baseline)."""
    n_pos = tp + fn
    if n_pos == 0:
        LOG.warning("В Val-патчах нет точек целевого класса: F1 не определён, записан 0")
    prevalence = n_pos / max(1, n_pos + n_neg)
    return {"f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, n_pos),
            "iou": tp / max(1, tp + fp + fn),
            "baseline_f1": 2 * prevalence / (1 + prevalence), "val_positive_points": n_pos,
            # Матрица ошибок: строки — истина (фон, цель), столбцы — предсказание.
            "confusion": [[n_neg - fp, fp], [fn, tp]]}


def confusion_counts(pred, y):
    """[tn, fp, fn, tp] на устройстве тензоров — без синхронизации на каждом батче."""
    import torch
    return torch.bincount(y.long() * 2 + pred.long(), minlength=4)


def counts_metrics(counts):
    tn, fp, fn, tp = (int(c) for c in counts)
    return binary_metrics(tp, fp, fn, tn + fp)


def evaluate(model, loader, device, cache=None):
    import torch
    model.eval()
    counts = torch.zeros(4, dtype=torch.long, device=device)
    with torch.no_grad():
        for feat, batch, y, keys in loader:
            feat, batch = feat.to(device), batch.to(device)
            pred = forward(model, feat, batch, keys, cache).argmax(dim=1)
            counts += confusion_counts(pred, y.to(device))
    return counts_metrics(counts)


def best_trial(study):
    import optuna
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and
                t.value is not None and math.isfinite(t.value)]
    return sorted(complete, key=lambda t: t.value, reverse=True)


def generate_report(study, output_path, run_info, data_summary=None, importances=True):
    import optuna
    complete = best_trial(study)
    title = TARGET_TITLES[run_info["target"]]
    lines = [f"# Подбор PointNet++: {title} ({run_info['features']})", "", "## Сводка", "",
             f"- Целевые классы: {run_info['target_ids']} → 1, остальное → 0.",
             f"- Признаки: `{run_info['features']}`; seed {run_info['seed']}; эпох на trial {run_info['epochs']}.",
             f"- Функция потерь: {run_info['loss']}; веса классов (фон, цель): {run_info['class_weights']}.",
             f"- Испытаний: {len(study.trials)} (успешных: {len(complete)})."]
    if not complete:
        lines += ["- Успешных испытаний нет; рекомендации невозможны.", ""]
    else:
        best = complete[0]
        baseline = best.user_attrs.get("baseline_f1", float("nan"))
        lines += [f"- Лучший F1 целевого класса: {best.value:.4f} (trial {best.number}); "
                  f"baseline «всё = 1»: {baseline:.4f}.",
                  f"- Лучшие гиперпараметры: `{json.dumps(best.params, ensure_ascii=False, sort_keys=True)}`.",
                  "", "## Топ-5 конфигураций", "",
                  "| Ранг | Trial | F1 | Precision | Recall | lr | weight_decay | dropout | batch_size (патчи) | n_sa_layers | n_points_l0 | radius_l0 | nsample | use_MSG |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"]
        keys = ["lr", "weight_decay", "dropout", "batch_size", "n_sa_layers",
                "n_points_l0", "radius_l0", "nsample", "use_MSG"]
        for rank, t in enumerate(complete[:5], 1):
            vals = [f"{t.params.get(k, '—'):.3g}" if isinstance(t.params.get(k), float)
                    else str(t.params.get(k, "—")) for k in keys]
            lines.append(f"| {rank} | {t.number} | {t.value:.4f} | "
                         f"{t.user_attrs.get('precision', float('nan')):.4f} | "
                         f"{t.user_attrs.get('recall', float('nan')):.4f} | " + " | ".join(vals) + " |")
        train_m, val_m = best.user_attrs.get("train"), best.user_attrs.get("val")
        if train_m and val_m:
            lines += ["", f"## Метрики лучшего trial (эпоха {best.user_attrs.get('epoch')})", "",
                      "| Выборка | IoU | F1 | Precision | Recall |", "|:---|---:|---:|---:|---:|"]
            lines += [f"| {name} | {m['iou']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | "
                      f"{m['recall']:.4f} |" for name, m in (("Train", train_m), ("Val", val_m))]
            for name, m in (("Train", train_m), ("Val", val_m)):
                (tn, fp), (fn, tp) = m["confusion"]
                lines += ["", f"Матрица ошибок {name} (точки; строки — истина, столбцы — предсказание):", "",
                          "| | Пред. фон | Пред. цель |", "|:---|---:|---:|",
                          f"| Фон | {tn} | {fp} |", f"| Цель | {fn} | {tp} |"]
        lines += ["", "## Важность параметров", ""]
        try:
            if not importances:  # fANOVA дорогая: считаем только в итоговом отчёте
                raise ValueError("считается в конце прогона")
            if len(complete) < 2:
                raise ValueError("Нужно не менее двух успешных испытаний")
            values = optuna.importance.get_param_importances(study)
            lines.extend(f"- `{k}`: {v:.3f}" for k, v in values.items())
        except (ValueError, RuntimeError) as exc:
            lines.append(f"Недоступна: {exc}.")
        lines += ["", "## Предупреждения", ""]
        if best.value - baseline < BASELINE_MARGIN:
            lines.append(f"- F1 модели выше baseline «всё = 1» меньше чем на {BASELINE_MARGIN}: "
                         "проверьте, учится ли сеть.")
        if best.user_attrs.get("val_positive_points", 1) == 0:
            lines.append("- В Val-патчах нет точек целевого класса: метрики не информативны.")
        if best.value < 0.6:
            lines.append("- Лучший F1 < 0.6: модель не достигла приемлемого качества.")
        if best.params["lr"] <= 1.1e-4 or best.params["lr"] >= 9e-3:
            lines.append("- lr близко к границе диапазона; расширьте поиск.")
        if lines[-1] == "":
            lines.append("- Пороговых предупреждений нет.")
        lines += ["", "## Кандидаты для финального прогона", ""]
        chosen = []
        for t in complete:
            signature = (t.params["n_sa_layers"], t.params["n_points_l0"],
                         t.params["radius_l0"], t.params["use_MSG"])
            if not any(s == signature for s in chosen):
                chosen.append(signature)
                lines.append(f"- Trial {t.number} (F1={t.value:.4f}): " +
                             ("лучший F1; проверить на длинном обучении." if len(chosen) == 1 else
                              "другая геометрия SA/MSG; сравнить устойчивость при длинном обучении."))
            if len(chosen) == 3:
                break
        if len(chosen) < 2:
            lines.append("- Различных конфигураций меньше двух; продолжите поиск перед выбором.")
    if data_summary:
        lines += ["", "# Сводка исходного датасета", "", data_summary.rstrip()]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best(study, output_dir, run_info):
    """best.json — лучшие параметры и метрики прогона; его читает summary_report.py."""
    complete = best_trial(study)
    best = complete[0] if complete else None
    out = Path(output_dir)
    model_path = out / "best_model.pt"
    try:
        prev = json.loads((out / "best.json").read_text(encoding="utf-8")).get("best_trial")
    except (OSError, ValueError):
        prev = None
    result = dict(run_info, trials=len(study.trials), complete=len(complete))
    if best:
        result.update(best_trial=best.number, f1=best.value,
                      **{k: best.user_attrs.get(k) for k in
                         ("precision", "recall", "iou", "baseline_f1", "val_positive_points",
                          "epoch", "train", "val")},
                      params=best.params)
        # Итоговая модель прогона — копия best.pt лучшего trial (веса, params, run_info).
        ckpt = out / f"trial_{best.number:04d}" / "best.pt"
        if ckpt.exists():
            if prev != best.number or not model_path.exists():  # копируем только при смене лучшего
                shutil.copy2(ckpt, model_path)
            result["model"] = model_path.name
    if "model" not in result:
        model_path.unlink(missing_ok=True)  # не оставлять веса другого trial рядом с best.json
    (out / "best.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Отдельный каталог на каждый прогон, например runs/ground_xyz")
    parser.add_argument("--target", choices=list(TARGETS), required=True)
    parser.add_argument("--features", choices=list(FEATURE_CHANNELS), default="xyz",
                        help="xyz или xyz_hag (+ HAG и is_above_ground, считаются по геометрии)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=8, help="Эпохи на trial")
    parser.add_argument("--patch-points", type=int, default=16384, help="Точек в патче")
    parser.add_argument("--patches-per-scene", type=int, default=2,
                        help="Патчей на сцену за эпоху (train и val)")
    parser.add_argument("--workers", type=int, default=0, help="Процессов DataLoader")
    parser.add_argument("--fixed-params", type=Path, default=None,
                        help="JSON с гиперпараметрами (например params из best.json): первым "
                             "trial обучается ровно эта конфигурация — для финальных прогонов")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce",
                        help="ce — взвешенная CrossEntropyLoss, focal — Focal Loss с теми же весами")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="gamma для --loss focal")
    parser.add_argument("--wandb-api-key", default=None, help="Не записывается в артефакты")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if min(args.n_trials, args.epochs, args.patch_points, args.patches_per_scene) < 1:
        parser.error("n-trials, epochs, patch-points и patches-per-scene должны быть положительными")
    # Зависимости импортируем после разбора аргументов для понятного --help.
    import optuna
    import torch
    try:
        import torch_pointcloud  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Установите torch-pointcloud==0.0.16 и его PyG-зависимости") from exc
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_ids = TARGETS[args.target]
    data_summary = generate_data_summary(args.data_dir)
    (args.output_dir / "data_summary.md").write_text(
        "# Сводка исходного датасета\n\n" + data_summary, encoding="utf-8")
    train_paths, val_paths = split_scenes(load_scenes(args.data_dir), args.seed)
    train = PatchDataset(train_paths, target_ids, args.patch_points,
                         args.patches_per_scene, args.seed, args.features)
    val = PatchDataset(val_paths, target_ids, args.patch_points,
                       args.patches_per_scene, args.seed, args.features)
    # Длина порядка = верх n_points_l0, чтобы префикс покрывал любой trial. Хранить стоит только
    # между trials; при одном trial (финальный прогон) — та же схема FPS без хранения.
    keep = args.n_trials > 1
    train_cache = FpsCache(N_POINTS_L0_MAX, True, args.seed, keep)
    val_cache = FpsCache(N_POINTS_L0_MAX, False, args.seed, keep)
    counts = class_counts(train_paths, target_ids)
    weights = class_weights(counts)
    run_info = {
        "target": args.target, "target_ids": list(target_ids), "features": args.features,
        "seed": args.seed, "epochs": args.epochs, "n_trials": args.n_trials,
        "patch_points": args.patch_points, "patches_per_scene": args.patches_per_scene,
        "data_dir": str(args.data_dir), "fixed_params": str(args.fixed_params or ""),
        "train_points": {"background": int(counts[0]), "target": int(counts[1])},
        "class_weights": [round(float(w), 4) for w in weights],
        "loss": "ce" if args.loss == "ce" else f"focal(gamma={args.focal_gamma:g})",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "optuna": optuna.__version__},
    }
    (args.output_dir / "split.json").write_text(json.dumps(dict(
        run_info, train=[p.name for p in train_paths], val=[p.name for p in val_paths]),
        indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("Патчей за эпоху: train=%s val=%s; устройство=%s; веса классов=%s",
             len(train), len(val), device, run_info["class_weights"])
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    wandb = None
    if args.wandb_api_key:
        import wandb as wandb_module
        wandb_module.login(key=args.wandb_api_key)
        wandb = wandb_module
    csv_path = args.output_dir / "trials.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def objective(trial):
        # Причина падения идёт в колонку error trials.csv.
        try:
            return run_trial(trial)
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise

    def run_trial(trial):
        params = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "batch_size": trial.suggest_categorical("batch_size", [2, 3, 4]),  # патчей в батче
            "n_sa_layers": trial.suggest_int("n_sa_layers", 2, 4),
            "n_points_l0": trial.suggest_int("n_points_l0", 1024, N_POINTS_L0_MAX),
            "radius_l0": trial.suggest_float("radius_l0", 0.05, 0.5),  # в долях радиуса патча
            "nsample": trial.suggest_int("nsample", 16, 64),
            "use_MSG": trial.suggest_categorical("use_MSG", [False, True]),
        }
        trial_seed = args.seed  # Один и тот же seed для сравнимости trials.
        seed_everything(trial_seed)
        run_dir = args.output_dir / f"trial_{trial.number:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(dict(
            run_info, model=MODEL_NAME, params=params,
            train=[p.name for p in train_paths], val=[p.name for p in val_paths]),
            indent=2, ensure_ascii=False), encoding="utf-8")
        model = build_model(params, FEATURE_CHANNELS[args.features]).to(device)
        criterion = make_criterion(args.loss, weights, args.focal_gamma)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"],
                                     weight_decay=params["weight_decay"])
        # ТЗ: переобучение замечено после 4-й эпохи — снижаем lr, когда F1 на Val не растёт.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                               factor=0.5, patience=1)
        val_loader = make_loader(Keyed(val), params["batch_size"], False, trial_seed, args.workers)
        best, best_m, history = -1.0, None, []
        ckpt_path = run_dir / "best.pt"
        wb_run = wandb.init(project="pointnet-hpo", config=dict(params, target=args.target, loss=run_info["loss"],
                                                                features=args.features),
                            name=f"{args.target}_{args.features}_trial_{trial.number:04d}",
                            reinit=True) if wandb else None
        try:
            for epoch in range(args.epochs):
                train.set_epoch(epoch)
                train_loader = make_loader(Keyed(train), params["batch_size"], True,
                                           trial_seed + epoch, args.workers)
                model.train()
                # Train-метрики — по предсказаниям самих шагов обучения (режим train, веса
                # меняются по ходу эпохи): без второго прохода по Train.
                train_counts = torch.zeros(4, dtype=torch.long, device=device)
                start, n_batches = time.time(), len(train_loader)
                for step, (feat, batch, y, keys) in enumerate(train_loader, 1):
                    feat, batch, y = feat.to(device), batch.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    out = forward(model, feat, batch, keys, train_cache)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    train_counts += confusion_counts(out.detach().argmax(dim=1), y)
                    if step % 20 == 0 or step == n_batches:  # прогресс внутри долгой эпохи
                        LOG.info("trial=%s epoch=%s batch %s/%s loss=%.4f %.0fs", trial.number,
                                 epoch + 1, step, n_batches, loss.item(), time.time() - start)
                train_m = counts_metrics(train_counts)
                m = evaluate(model, val_loader, device, val_cache)
                seconds = round(time.time() - start, 1)  # обучение + валидация
                f1 = m["f1"]
                lr = optimizer.param_groups[0]["lr"]
                history.append({"epoch": epoch + 1, "f1": f1, "precision": m["precision"],
                                "recall": m["recall"], "iou": m["iou"],
                                **{f"train_{k}": train_m[k] for k in ("f1", "precision", "recall", "iou")},
                                "lr": lr, "seconds": seconds})
                scheduler.step(f1)
                if wb_run:
                    wb_run.log({f"{name}/{k}": v for name, d in (("val", m), ("train", train_m))
                                for k, v in d.items() if k != "confusion"} |
                               {"epoch": epoch + 1, "lr": lr})
                if f1 > best:
                    best, best_m = f1, dict(m, epoch=epoch + 1, train=train_m, val=m)
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "params": params, "run_info": run_info, "epoch": epoch + 1,
                                "f1": best}, ckpt_path)
                trial.report(f1, epoch)
                LOG.info("trial=%s epoch=%s F1=%.4f best=%.4f baseline=%.4f lr=%.2g время=%s",
                         trial.number, epoch + 1, f1, best, m["baseline_f1"], lr,
                         timedelta(seconds=round(seconds)))
                # Без «trial=»: mlstudio разбирает строки с «trial=… epoch=… F1=» как точки графика.
                LOG.info("trial %s epoch %s | Train: IoU=%.4f F1=%.4f P=%.4f R=%.4f "
                         "| Val: IoU=%.4f F1=%.4f P=%.4f R=%.4f", trial.number, epoch + 1,
                         *(d[k] for d in (train_m, m) for k in ("iou", "f1", "precision", "recall")))
                LOG.info("trial %s epoch %s | матрица ошибок [[TN, FP], [FN, TP]]: Train=%s Val=%s",
                         trial.number, epoch + 1, train_m["confusion"], m["confusion"])
                if trial.should_prune():
                    raise optuna.TrialPruned()
            trial.set_user_attr("checkpoint", str(ckpt_path))
            return best
        finally:
            if best_m:
                for key in ("precision", "recall", "iou", "baseline_f1", "val_positive_points",
                            "epoch", "train", "val"):
                    trial.set_user_attr(key, best_m[key])
                with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=EPOCH_FIELDS)
                    writer.writeheader()
                    writer.writerows(history)
                (run_dir / "metrics.json").write_text(json.dumps({
                    "seed": trial_seed, "target": args.target, "features": args.features,
                    "class_weights": run_info["class_weights"], "best": best_m,
                }, indent=2), encoding="utf-8")
            if wb_run:
                wb_run.finish()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def log_trial(study, trial):
        seconds = round(trial.duration.total_seconds(), 1) if trial.duration else ""
        if seconds != "":
            LOG.info("trial=%s %s за %s", trial.number, trial.state.name,
                     timedelta(seconds=round(seconds)))
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow({
                "number": trial.number, "state": trial.state.name, "f1": trial.value,
                "precision": trial.user_attrs.get("precision", ""),
                "recall": trial.user_attrs.get("recall", ""),
                "baseline_f1": trial.user_attrs.get("baseline_f1", ""), "seed": args.seed,
                "target": args.target, "features": args.features,
                "checkpoint": trial.user_attrs.get("checkpoint", ""),  # только у COMPLETE
                "error": trial.user_attrs.get("error", ""),
                "params": json.dumps(trial.params, ensure_ascii=False), "seconds": seconds})
        generate_report(study, args.output_dir / "report.md", run_info, data_summary,
                        importances=False)
        write_best(study, args.output_dir, run_info)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed),
        # Медиана по 5 полным trials; обрезаем не раньше 4-й эпохи: к ней ReduceLROnPlateau
        # (patience 1) успевает снизить lr, и медленные конфиги не режутся зря.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=max(3, args.epochs // 3)),
        study_name="pointnet_binary", storage=f"sqlite:///{(args.output_dir / 'study.db').as_posix()}",
        load_if_exists=True)
    # Trials старого пространства поиска несовместимы с новым: Optuna упадёт на первом же trial.
    if study.trials and study.user_attrs.get("search_space") != SEARCH_SPACE_VERSION:
        raise SystemExit(f"{args.output_dir}: study версии {study.user_attrs.get('search_space')}, "
                         f"код — версии {SEARCH_SPACE_VERSION} (изменились диапазоны поиска или "
                         "обучение, trials несравнимы); укажите новую --output-dir "
                         "(для run_all.py — новую --runs) или перенесите старый каталог")
    # Trials с разными функциями потерь несравнимы; studies без атрибута обучались на CE.
    if study.trials and study.user_attrs.get("loss", "ce") != run_info["loss"]:
        raise SystemExit(f"{args.output_dir}: study обучалась с loss={study.user_attrs.get('loss', 'ce')}, "
                         f"запрошено {run_info['loss']}; укажите другую --output-dir")
    study.set_user_attr("search_space", SEARCH_SPACE_VERSION)
    study.set_user_attr("loss", run_info["loss"])
    if args.fixed_params and not study.trials:
        study.enqueue_trial(json.loads(args.fixed_params.read_text(encoding="utf-8")))
    study.optimize(objective, n_trials=args.n_trials, callbacks=[log_trial],
                   catch=(Exception,))  # падение одного trial не обрывает поиск
    generate_report(study, args.output_dir / "report.md", run_info, data_summary)
    write_best(study, args.output_dir, run_info)
    LOG.info("Отчёт: %s", args.output_dir / "report.md")


if __name__ == "__main__":
    main()
