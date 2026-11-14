//! Превью сцены: облако точек из scene_*.npy ([N, 9] float32: class, x, y, z, …), программная 3D-проекция в картинку.
use std::path::Path;

/// Сколько точек рисовать: больше — медленнее вращение, а глазом разница не видна.
pub const MAX_POINTS: usize = 120_000;

pub struct Cloud {
    /// Координаты, центрированные по XY, z от минимума, в долях наибольшего размаха.
    pub pts: Vec<[f32; 3]>,
    pub cls: Vec<u8>,
    /// Всего точек в файле.
    pub total: usize,
    /// (класс, число точек в выборке) по убыванию.
    pub classes: Vec<(u8, usize)>,
    /// Размах сцены в метрах (x, y, z).
    pub extent: [f32; 3],
    /// Номера строк файла, попавших в выборку (для сопоставления с предсказанием).
    pub rows: Vec<usize>,
    /// Класс каждой точки файла (все N) — для метрик предсказания по всей сцене.
    pub labels: Vec<u8>,
}

/// Форма и данные .npy с нужным dtype (`'<f4'`, `'|u1'`), только порядок C.
fn npy<'a>(bytes: &'a [u8], descr: &str) -> Result<(Vec<u64>, &'a [u8]), String> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("не .npy".into());
    }
    let (len, start) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12)
    };
    let header = String::from_utf8_lossy(bytes.get(start..start + len).ok_or("обрезанный заголовок")?);
    if !header.contains(descr) || header.contains("'fortran_order': True") {
        return Err(format!("ожидался {descr} в порядке C, а в заголовке {}", header.trim()));
    }
    let shape = crate::data::parse_shape(&header).ok_or("нет формы массива")?;
    Ok((shape, &bytes[start + len..]))
}

/// Исходный класс → задача (README: ground 2; vegetation 3,4,5; building 6; noise 7,18; water 9).
pub fn target_of(class: u8) -> &'static str {
    match class {
        2 => "ground",
        3..=5 => "vegetation",
        6 => "building",
        7 | 18 => "noise",
        9 => "water",
        _ => "",
    }
}

pub fn load(path: &Path) -> Result<Cloud, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let (shape, data) = npy(&bytes, "'<f4'")?;
    let (n, cols) = (shape.first().copied().unwrap_or(0) as usize, shape.get(1).copied().unwrap_or(1) as usize);
    if cols < 4 {
        return Err(format!("нужно ≥ 4 колонок (class, x, y, z), а их {cols}"));
    }
    if data.len() < n * cols * 4 {
        return Err("файл короче, чем указано в заголовке".into());
    }
    let at = |row: usize, col: usize| {
        let i = (row * cols + col) * 4;
        f32::from_le_bytes(data[i..i + 4].try_into().unwrap())
    };
    let step = n.div_ceil(MAX_POINTS).max(1);
    let rows: Vec<usize> = (0..n).step_by(step).collect();
    let raw: Vec<[f32; 3]> = rows.iter().map(|&r| [at(r, 1), at(r, 2), at(r, 3)]).collect();
    let cls: Vec<u8> = rows.iter().map(|&r| at(r, 0).round().clamp(0., 255.) as u8).collect();
    if raw.is_empty() {
        return Err("в файле нет точек".into());
    }

    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in &raw {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let extent = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let scale = extent.iter().cloned().fold(1e-6, f32::max);
    let (cx, cy, cz) = ((lo[0] + hi[0]) / 2., (lo[1] + hi[1]) / 2., (lo[2] + hi[2]) / 2.);
    let pts = raw.iter().map(|p| [(p[0] - cx) / scale, (p[1] - cy) / scale, (p[2] - cz) / scale]).collect();

    let mut counts = [0usize; 256];
    cls.iter().for_each(|&c| counts[c as usize] += 1);
    let mut classes: Vec<(u8, usize)> = (0..=255u8).map(|c| (c, counts[c as usize])).filter(|c| c.1 > 0).collect();
    classes.sort_by(|a, b| b.1.cmp(&a.1));
    let labels = (0..n).map(|r| at(r, 0).round().clamp(0., 255.) as u8).collect();
    Ok(Cloud { pts, cls, total: n, classes, extent, rows, labels })
}

/// Исход точки при предсказании; NONE — точка не покрыта патчами.
pub const TN: u8 = 0;
pub const TP: u8 = 1;
pub const FP: u8 = 2;
pub const FN: u8 = 3;
pub const NONE: u8 = 255;

pub struct Outcome {
    /// Исход для точек выборки (как `Cloud::pts`).
    pub sample: Vec<u8>,
    /// TN, TP, FP, FN по всей сцене.
    pub counts: [usize; 4],
    pub uncovered: usize,
}

impl Outcome {
    pub fn f1(&self) -> f64 {
        let [_, tp, fp, fn_] = self.counts;
        2. * tp as f64 / (2 * tp + fp + fn_).max(1) as f64
    }
}

/// Предсказание predict_scene.py (.npy uint8 [N]: 1 — класс, 0 — нет, 255 — не покрыто) против меток сцены.
pub fn load_pred(path: &Path, c: &Cloud, target: &str) -> Result<Outcome, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let (shape, data) = npy(&bytes, "'|u1'")?;
    if shape.first().copied() != Some(c.total as u64) || data.len() < c.total {
        return Err(format!("в предсказании {:?} точек, а в сцене {}", shape.first(), c.total));
    }
    let outcome = |i: usize| match (data[i], target_of(c.labels[i]) == target) {
        (1, true) => TP,
        (1, false) => FP,
        (0, true) => FN,
        (0, false) => TN,
        _ => NONE,
    };
    let (mut counts, mut uncovered) = ([0usize; 4], 0);
    for i in 0..c.total {
        match outcome(i) {
            NONE => uncovered += 1,
            o => counts[o as usize] += 1,
        }
    }
    Ok(Outcome { sample: c.rows.iter().map(|&r| outcome(r)).collect(), counts, uncovered })
}

#[derive(Clone, Copy, PartialEq)]
pub enum ColorBy {
    Class,
    Height,
    /// Исход предсказания: TP зелёный, FP красный, FN жёлтый, TN серый.
    Errors,
}

pub struct View {
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub pan_x: f32,
    pub pan_y: f32,
}

impl Default for View {
    fn default() -> Self {
        Self { yaw: -0.6, pitch: 0.9, zoom: 1.35, pan_x: 0., pan_y: 0. }
    }
}

fn rgb(hex: u32) -> [u8; 3] {
    [(hex >> 16) as u8, (hex >> 8) as u8, hex as u8]
}

/// Высота: синий → бирюзовый → жёлтый → красный.
fn ramp(t: f32) -> [u8; 3] {
    let stops = [[40., 80., 200.], [40., 190., 180.], [240., 210., 60.], [220., 60., 40.]];
    let t = t.clamp(0., 1.) * 3.;
    let (i, f) = ((t as usize).min(2), t - (t as usize).min(2) as f32);
    let (a, b) = (stops[i], stops[i + 1]);
    [0, 1, 2].map(|k| (a[k] + (b[k] - a[k]) * f) as u8)
}

/// Картинка w×h в BGRA (формат RenderImage): ортографическая проекция с z-буфером, точка 2×2 пикселя.
/// `stride` — рисовать каждую n-ю точку (быстрый режим при вращении).
pub fn render(c: &Cloud, v: &View, w: u32, h: u32, stride: usize, by: ColorBy, outcome: Option<&[u8]>, bg: u32, class_hex: impl Fn(&str) -> u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut px = vec![0u8; w * h * 4];
    let bg = rgb(bg);
    for p in px.chunks_exact_mut(4) {
        p.copy_from_slice(&[bg[2], bg[1], bg[0], 255]);
    }
    let mut depth = vec![f32::MAX; w * h];
    let (sy, cy) = v.yaw.sin_cos();
    let (sp, cp) = v.pitch.sin_cos();
    let s = w.min(h) as f32 * 0.85 * v.zoom;
    let palette: Vec<[u8; 3]> = (0..=255u8).map(|k| match target_of(k) {
        "" => rgb(0x8A8F98),
        t => rgb(class_hex(t)),
    }).collect();
    let zmin = c.pts.iter().map(|p| p[2]).fold(f32::MAX, f32::min);
    let zspan = c.pts.iter().map(|p| p[2]).fold(f32::MIN, f32::max) - zmin;
    for (i, (p, &k)) in c.pts.iter().zip(&c.cls).enumerate().step_by(stride.max(1)) {
        // поворот вокруг Z (yaw), затем наклон вокруг X (pitch)
        let x = p[0] * cy - p[1] * sy;
        let y = p[0] * sy + p[1] * cy;
        let (y2, d) = (y * cp - p[2] * sp, y * sp + p[2] * cp);
        let (sx, sy_) = (w as f32 / 2. + v.pan_x * w as f32 + x * s, h as f32 / 2. + v.pan_y * h as f32 + y2 * s);
        if sx < 0. || sy_ < 0. {
            continue;
        }
        let (ix, iy) = (sx as usize, sy_ as usize);
        if ix + 1 >= w || iy + 1 >= h {
            continue;
        }
        let col = match by {
            ColorBy::Class => palette[k as usize],
            ColorBy::Height => ramp(if zspan > 0. { (p[2] - zmin) / zspan } else { 0.5 }),
            ColorBy::Errors => rgb(match outcome.and_then(|o| o.get(i)).copied() {
                Some(TP) => 0x2EAD5B,
                Some(FP) => 0xE8452C,
                Some(FN) => 0xF2C200,
                Some(TN) => 0x5B6068,
                _ => 0x30343B,
            }),
        };
        // ближе к камере — меньше -d (камера смотрит сверху-сбоку)
        let z = -d;
        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let i = (iy + dy) * w + ix + dx;
            if z < depth[i] {
                depth[i] = z;
                px[i * 4..i * 4 + 4].copy_from_slice(&[col[2], col[1], col[0], 255]);
            }
        }
    }
    px
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(rows: &[[f32; 9]]) -> Vec<u8> {
        let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, 9), }}", rows.len());
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend((header.len() as u16).to_le_bytes());
        out.extend(header.as_bytes());
        rows.iter().flatten().for_each(|v| out.extend(v.to_le_bytes()));
        out
    }

    #[test]
    fn load_and_render() {
        let dir = std::env::temp_dir().join("mlstudio_cloud_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scene.npy");
        let mut rows = vec![];
        for i in 0..100 {
            let f = i as f32;
            rows.push([if i % 2 == 0 { 2. } else { 6. }, f, f * 2., f / 10., 0., 0., 0., 0., 0.]);
        }
        std::fs::write(&path, npy(&rows)).unwrap();
        let c = load(&path).unwrap();
        assert_eq!((c.total, c.pts.len()), (100, 100));
        assert_eq!(c.classes, vec![(2, 50), (6, 50)]);
        assert!((c.extent[1] - 198.).abs() < 1e-3);
        let img = render(&c, &View::default(), 64, 48, 1, ColorBy::Class, None, 0x000000, |_| 0xFF0000);
        assert_eq!(img.len(), 64 * 48 * 4);
        // что-то нарисовано красным (BGRA: R в байте 2)
        assert!(img.chunks_exact(4).any(|p| p[2] == 255 && p[0] == 0));
        // сдвиг вида смещает кадр
        let panned = render(&c, &View { pan_x: 0.1, ..View::default() }, 64, 48, 1, ColorBy::Class, None, 0x000000, |_| 0xFF0000);
        assert_ne!(img, panned);
    }

    #[test]
    fn prediction_outcomes() {
        let dir = std::env::temp_dir().join("mlstudio_cloud_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scene2.npy");
        // классы: грунт, грунт, здание, здание
        let rows: Vec<[f32; 9]> = [2., 2., 6., 6.].iter().enumerate().map(|(i, &k)| [k, i as f32, 0., 0., 0., 0., 0., 0., 0.]).collect();
        std::fs::write(&path, npy(&rows)).unwrap();
        let c = load(&path).unwrap();
        // предсказание «грунт»: да, нет, да, не покрыто → TP, FN, FP, —
        let mut header = "{'descr': '|u1', 'fortran_order': False, 'shape': (4,), }".to_string();
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(' ');
        }
        header.push('\n');
        let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
        bytes.extend((header.len() as u16).to_le_bytes());
        bytes.extend(header.as_bytes());
        bytes.extend([1u8, 0, 1, 255]);
        let pred = dir.join("pred.npy");
        std::fs::write(&pred, bytes).unwrap();
        let o = load_pred(&pred, &c, "ground").unwrap();
        assert_eq!(o.sample, vec![TP, FN, FP, NONE]);
        assert_eq!((o.counts, o.uncovered), ([0, 1, 1, 1], 1));
        assert!((o.f1() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn rejects_float64() {
        let dir = std::env::temp_dir().join("mlstudio_cloud_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f8.npy");
        let mut bytes = npy(&[[0.; 9]]);
        let i = bytes.windows(4).position(|w| w == b"<f4'").unwrap();
        bytes[i + 2] = b'8';
        std::fs::write(&path, bytes).unwrap();
        assert!(load(&path).is_err());
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// cargo test --bin mlstudio render_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn render_speed() {
        let n = MAX_POINTS;
        let pts = (0..n).map(|i| { let f = i as f32 / n as f32; [f.sin() * 0.5, (f * 7.).cos() * 0.5, f * 0.2] }).collect();
        let c = Cloud { pts, cls: vec![2; n], total: n, classes: vec![], extent: [100.; 3], rows: (0..n).collect(), labels: vec![2; n] };
        for (w, h, s) in [(1200, 800, 1), (600, 400, 3)] {
            let t = std::time::Instant::now();
            for _ in 0..10 {
                render(&c, &View::default(), w, h, s, ColorBy::Class, None, 0, |_| 0xFFFFFF);
            }
            println!("{w}x{h} stride {s}: {:.1} ms", t.elapsed().as_secs_f64() * 100.);
        }
    }
}
