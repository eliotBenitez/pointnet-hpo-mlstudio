# Раздел «Артефакты» Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Добавить в ML Studio отдельный раздел «Артефакты» для встроенного просмотра `report.md`, `data_summary.md` и интерактивного 3D-превью `.pt` моделей.

**Architecture:** Новая вкладка в существующем GPUI-рельсе, без изменения файловых форматов и текущего экрана «Тестирование». Логика обнаружения документов и моделей остаётся файловой; preview переиспользует существующие данные облака, инференс и цветовую схему.

**Tech Stack:** Rust, GPUI/gpui-component, существующие `mlstudio/src/main.rs`, `data.rs`, `cloud.rs`, stdlib unittest/Rust tests.

**Spec:** `docs/superpowers/specs/2026-11-13-artifacts-viewer-design.md`

## Global Constraints

- Все новые подписи и сообщения — на русском.
- Файловая система остаётся источником истины.
- Не показывать фабрикованные метрики.
- Не менять формат логов и артефактов.
- `run_all.cmd`/`run_all.ps1` не изменять.
- Не добавлять webview или тяжёлый 3D-движок.
- Сохранять существующий inference pipeline и лимит/прореживание больших облаков.

## Review Focus

- Пустой `runs/` и отсутствующие документы: безопасное состояние с фактическим путём, без падения.
- Несколько вариантов `.pt`: корректно показывать run models, `trial_XXXX/best.pt` и `models/*.pt` без дубликатов.
- Большие сцены: preview остаётся отзывчивым за счёт существующего ограничения точек.
- Ошибка/отсутствие Python-инференса: явное русское сообщение вместо пустого canvas.
- Переключение между разделами: существующие «Тестирование», запуск и мониторинг не теряют состояние.

## Файлы и границы

- Modify: `mlstudio/src/main.rs` — состояние, маршрут рельса, экран «Артефакты», document/model controls и preview.
- Modify: `mlstudio/src/data.rs` — небольшие чистые helpers для поиска/чтения документов и дедупликации моделей, если текущие helpers недостаточны.
- Modify: `mlstudio/src/cloud.rs` — только если нужен общий интерактивный transform/renderer; не дублировать текущий cloud preparation.
- Test: существующие Rust tests в `data.rs`/`cloud.rs` и новые unit tests рядом с helpers.
- Modify: `mlstudio/PRODUCT.md` — снять статус «undecided» для встроенного report viewer и описать раздел.

### Task 1: File-system artifact helpers

- [ ] Проверить текущие пути `report.md`, `data_summary.md`, model discovery и существующие тесты.
- [ ] Добавить чистые функции для разрешения document paths, чтения UTF-8 Markdown и получения дедуплицированного списка доступных `.pt` с понятной сортировкой.
- [ ] Добавить тесты для отсутствующего файла, пустого файла, дубликатов моделей и порядка `best_model.pt`/trial/imported.
- [ ] Запустить targeted Rust tests в `mlstudio`.

### Task 2: Artifact screen state and rail route

- [ ] Добавить enum/state для выбранного раздела документов/3D, выбранного пути и выбранной модели/сцены, используя существующие паттерны main.rs.
- [ ] Добавить пункт «Артефакты» в рельс и routing/render branch.
- [ ] Убедиться, что выбор раздела не сбрасывает состояния «Тестирование» и активных процессов.
- [ ] Собрать `cargo check`.

### Task 3: Read-only Markdown viewer

- [ ] Реализовать read-only rendering для заголовков, абзацев, списков, таблиц и code blocks; неизвестные строки отображать безопасным текстовым fallback.
- [ ] Показать название файла, фактический путь и русские состояния «файл не найден»/«файл пуст».
- [ ] Добавить переключатели «Сводный отчёт» и «Данные».
- [ ] Проверить `cargo test` и ручной build.

### Task 4: Interactive model preview

- [ ] Переиспользовать существующий model discovery/import/inference pipeline и cloud point preparation.
- [ ] Добавить выбор `.pt`, сцены и кнопки запуска/обновления preview.
- [ ] Добавить интерактивную перспективную проекцию: rotate, zoom, pan; переключатель истинных классов/предсказаний; легенду цветов текущей темы.
- [ ] Показать loading/error/empty states на русском и не фабриковать результаты.
- [ ] Проверить на маленьком наборе данных и убедиться, что существующее preview в «Тестировании» не изменилось.

### Task 5: Documentation and verification

- [ ] Обновить `mlstudio/PRODUCT.md` с новой capability и убрать старое «Undecided».
- [ ] Запустить `cargo test` в `mlstudio`.
- [ ] Запустить `cargo build --release` в `mlstudio`.
- [ ] Проверить diff и убедиться, что изменены только согласованные файлы.
