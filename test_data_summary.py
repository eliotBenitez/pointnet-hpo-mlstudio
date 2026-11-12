import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import hpo_pointnet as hpo


class DataSummaryTests(unittest.TestCase):
    def test_all_scenes_and_present_classes_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "01 Лес.npy", np.array([
                [2, 0, 0, 0, 9], [2, 2, 4, 6, 9], [5, 1, 1, 1, 9], [5, 1, 2, 3, 9]
            ], dtype=np.float32))
            np.save(root / "02 Степь.npy", np.array([
                [2, 0, 0, 0, 9], [2, 1, 1, 1, 9]
            ], dtype=np.float32))
            text = hpo.generate_data_summary(root)
            self.assertIn("| 01 Лес | 4 | 2 | 2.0 × 4.0 × 6.0 |", text)
            self.assertIn("| 02 Степь | 100.00 | — | 100.00 |", text)
            self.assertIn("| Поверхность | 2 | 4 | 66.667 | 2 / 2 |", text)
            self.assertIn("| Высокая растительность | 5 | 2 | 33.333 | 1 / 2 |", text)
            self.assertNotIn("| Вода |", text)  # отсутствующие классы не засоряют таблицы

    def test_unknown_class_id_gets_own_column(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "a.npy", np.array([[2, 0, 0, 0], [8, 1, 1, 1]], dtype=np.float32))
            text = hpo.generate_data_summary(root)
            self.assertIn("| ID 8 |", text)
            self.assertIn("| Класс 8 | 8 | 1 | 50.000 | 1 / 1 |", text)


class PatchDatasetTests(unittest.TestCase):
    def scene(self, directory, xyz, labels):
        path = Path(directory) / "a.npy"
        extra = np.full((len(xyz), 1), 7.0)  # лишние колонки, как в exports, игнорируются
        np.save(path, np.column_stack((labels, xyz, extra)).astype(np.float32))
        return [path]

    def test_patch_is_seeded_local_thinned_and_binary(self):
        rng = np.random.default_rng(0)
        xyz = np.column_stack((rng.uniform(0, 100, (5000, 2)), rng.normal(10, 1, 5000)))
        labels = np.where(xyz[:, 2] > 10, 5, 2)
        with tempfile.TemporaryDirectory() as directory:
            paths = self.scene(directory, xyz, labels)

            def patch(seed, epoch=0, i=0, n=500):
                ds = hpo.PatchDataset(paths, [5], n_points=n, patches_per_scene=2, seed=seed)
                ds.set_epoch(epoch)
                return ds[i]

            (pos, y), (pos2, y2) = patch(1), patch(1)
            self.assertEqual(pos.shape, (500, 3))
            self.assertTrue((pos == pos2).all() and (y == y2).all())
            self.assertTrue(set(y.tolist()) <= {0, 1})
            # Патч — круг вокруг центра, нормированный на свой радиус.
            self.assertTrue((pos[:, :2].norm(dim=1) <= 1 + 1e-5).all())
            self.assertFalse(np.array_equal(pos.numpy(), patch(2)[0].numpy()))
            self.assertFalse(np.array_equal(pos.numpy(), patch(1, epoch=1)[0].numpy()))
            self.assertFalse(np.array_equal(pos.numpy(), patch(1, i=1)[0].numpy()))
            # Сцена меньше патча: радиус растёт, берутся все точки без дублей.
            self.assertEqual(len(patch(1, n=20000)[0]), 5000)

    def test_target_group_becomes_one(self):
        xyz = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            paths = self.scene(directory, xyz, [2, 4, 5, 6])
            _, y = hpo.PatchDataset(paths, hpo.TARGETS["vegetation"], n_points=4,
                                    patches_per_scene=1)[0]
            self.assertEqual(y.tolist(), [0, 1, 1, 0])

    def test_hag_ignores_labels_and_marks_canopy(self):
        rng = np.random.default_rng(1)
        ground = np.column_stack((rng.uniform(0, 10, (400, 2)), rng.normal(0, 0.01, 400)))
        canopy = np.column_stack((rng.uniform(0, 10, (100, 2)), np.full(100, 2.0)))
        xyz = np.vstack((ground, canopy)).astype(np.float32)
        hag = hpo.compute_hag(xyz)
        self.assertLess(float(hag[:400].mean()), 0.1)
        self.assertGreater(float(hag[400:].mean()), 1.5)
        with tempfile.TemporaryDirectory() as directory:
            paths = self.scene(directory, xyz, np.full(500, 2))
            feat, _ = hpo.PatchDataset(paths, [2], 500, 1, 0, features="xyz_hag")[0]
            self.assertEqual(feat.shape, (500, 5))
            self.assertEqual(set(feat[:, 4].tolist()), {0.0, 1.0})
            self.assertEqual(int(feat[:, 4].sum()), 100)  # is_above_ground = ровно крона


class MetricsTests(unittest.TestCase):
    def test_baseline_is_f1_of_predicting_all_ones(self):
        m = hpo.binary_metrics(tp=10, fp=5, fn=15, n_neg=75)
        self.assertAlmostEqual(m["baseline_f1"], 2 * 0.25 / 1.25)
        self.assertAlmostEqual(m["f1"], 20 / 40)
        self.assertEqual(m["val_positive_points"], 25)
        self.assertAlmostEqual(m["iou"], 10 / 30)
        self.assertEqual(m["confusion"], [[70, 5], [15, 10]])

    def test_no_positives_gives_zero_f1_and_warns(self):
        with self.assertLogs("hpo_pointnet", "WARNING"):
            m = hpo.binary_metrics(tp=0, fp=3, fn=0, n_neg=100)
        self.assertEqual((m["f1"], m["baseline_f1"]), (0.0, 0.0))


class EndToEndTests(unittest.TestCase):
    def test_runs_fixed_params_and_summary_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            for i in range(4):
                rng = np.random.default_rng(10 + i)
                xyz = np.column_stack((rng.uniform(0, 20, (400, 2)), rng.uniform(0, 3, 400)))
                labels = np.where(xyz[:, 2] > 1, 5, 2)
                np.save(data / f"{i:02d}.npy", np.column_stack((labels, xyz)).astype(np.float32))

            def run(name, *extra):
                cmd = [sys.executable, str(Path(hpo.__file__)), "--data-dir", str(data),
                       "--output-dir", str(root / name), "--target", "ground", "--n-trials", "1",
                       "--epochs", "2", "--patch-points", "128", "--patches-per-scene", "2", *extra]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                self.assertEqual(result.returncode, 0, result.stderr)
                return json.loads((root / name / "best.json").read_text(encoding="utf-8"))

            # Два trials и воркер: кеш FPS хранится и проходит весь цикл обучения/валидации.
            best = run("ground_xyz", "--n-trials", "2", "--workers", "1")
            self.assertEqual(best["complete"], 2)
            self.assertEqual((best["seed"], best["target"], best["features"]), (42, "ground", "xyz"))
            self.assertTrue({"f1", "precision", "recall", "baseline_f1", "params"} <= best.keys())
            self.assertEqual(len(best["class_weights"]), 2)
            for split in ("train", "val"):
                self.assertTrue({"iou", "f1", "precision", "recall"} <= best[split].keys())
                self.assertEqual(len(best[split]["confusion"]), 2)
            self.assertTrue((root / "ground_xyz" / best["model"]).exists())
            self.assertIn("Матрица ошибок Val", (root / "ground_xyz/report.md").read_text(encoding="utf-8"))
            self.assertIn("lr", (root / "ground_xyz/trial_0000/metrics.csv").read_text(encoding="utf-8"))
            split = json.loads((root / "ground_xyz/split.json").read_text(encoding="utf-8"))
            self.assertFalse(set(split["train"]) & set(split["val"]))

            params = root / "params.json"
            params.write_text(json.dumps(best["params"]), encoding="utf-8")
            fixed = run("ground_xyz_hag", "--features", "xyz_hag", "--fixed-params", str(params))
            self.assertEqual(fixed["params"], best["params"])

            sys.path.insert(0, str(Path(hpo.__file__).parent))
            import summary_report
            text = summary_report.build(root)
            self.assertIn("## Грунт", text)
            self.assertIn("Сравнение признаков `xyz` → `xyz_hag`", text)
            self.assertIn("## Вода", text)


class CachedFpsTests(unittest.TestCase):
    def test_cached_fps_prefix_matches_model_fps(self):
        import torch
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, n in (("a", 5000), ("b", 400)):  # вторая сцена меньше патча: патчи разной длины
                paths.append(Path(directory) / f"{name}.npy")
                np.save(paths[-1], np.column_stack((rng.integers(0, 3, n),
                                                    rng.uniform(0, 50, (n, 3)))).astype(np.float32))
            ds = hpo.Keyed(hpo.PatchDataset(paths, [2], n_points=600, patches_per_scene=1))
            feat, batch, _, keys = hpo.collate_flat([ds[0], ds[1]])
            self.assertEqual(batch.bincount().tolist(), [600, 400])
            # Длина порядка 500 больше второго патча: проверяет и повторы индексов.
            val_cache = hpo.FpsCache(500, False, 0)
            for use_msg, n_l0 in ((False, 200), (True, 450)):
                params = dict(dropout=0.0, n_sa_layers=2, n_points_l0=n_l0, radius_l0=0.2,
                              nsample=16, use_MSG=use_msg)
                model = hpo.build_model(params).eval()
                with torch.no_grad():
                    plain = hpo.forward(model, feat, batch)
                    self.assertTrue(torch.equal(hpo.forward(model, feat, batch, keys, val_cache), plain))
                    # Хук снят: прямой вызов модели снова считает FPS сам.
                    self.assertTrue(torch.equal(model(feat, feat[:, :3].contiguous(), batch), plain))
            # Train: случайный старт, префикс длинного порядка = короткий порядок, индексы в своём патче.
            pos = feat[:, :3].contiguous()
            train_cache = hpo.FpsCache(500, True, 0)
            long = train_cache(pos, keys, 500)
            self.assertTrue(torch.equal(train_cache(pos, keys, 200), long[:, :200]))  # из кеша
            # Без хранения (один trial) — тот же порядок: режим не зависит от --n-trials.
            self.assertTrue(torch.equal(hpo.FpsCache(500, True, 0, keep=False)(pos, keys, 200),
                                        long[:, :200]))
            self.assertTrue(bool((batch[long] == torch.arange(2)[:, None]).all()))
            self.assertFalse(torch.equal(long, val_cache(pos, keys, 500)))  # старт не первая точка
            # Частичный промах: считается только новый патч, сохранённые строки не меняются.
            partial = hpo.FpsCache(500, True, 0)
            partial(pos[:600], keys[:1], 500)
            self.assertTrue(torch.equal(partial(pos, keys, 500), long))
            self.assertEqual(train_cache.orders[keys[0]].device.type, "cpu")
            with self.assertRaises(ValueError):  # n_points_l0 длиннее кеша не обрезается молча
                train_cache(pos, keys, 501)

    def test_default_dataset_has_no_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.npy"
            np.save(path, np.random.default_rng(0).uniform(0, 9, (50, 4)).astype(np.float32))
            ds = hpo.PatchDataset([path], [2], n_points=20, patches_per_scene=2)
            self.assertIsNone(hpo.collate_flat([ds[0], ds[1]])[3])


if __name__ == "__main__":
    unittest.main()
