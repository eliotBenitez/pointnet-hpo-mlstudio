//! Монитор ресурсов: ЦП и ОЗУ (sysinfo); видеокарты — счётчики Windows, как в Диспетчере задач (любой производитель,
//! в т. ч. встроенные), плюс nvidia-smi для NVIDIA (точнее загрузка CUDA и температура). Опрос в фоне раз в 2 с;
//! nvidia-smi — раз в 10 с: он запускает процесс (~0.5 с ЦП), а его показания меняются медленно.
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

#[derive(Clone, Default)]
pub struct Usage {
    /// Номер замера: меняется — пора перерисовать.
    pub seq: u64,
    pub cpu: f32,
    pub ram_used: u64,
    pub ram_total: u64,
    pub gpus: Vec<GpuUsage>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuUsage {
    pub name: String,
    pub util: f32,
    pub mem_used_mb: f64,
    pub mem_total_mb: f64,
    pub temp: Option<f32>,
    /// Память — общая с ОЗУ (встроенная видеокарта без своей), а не видеопамять.
    pub shared: bool,
}

/// Последний замер; поток опроса один на всё приложение (смена проекта его не плодит).
pub fn latest() -> Usage {
    static CELL: OnceLock<&'static Mutex<Usage>> = OnceLock::new();
    let cell = CELL.get_or_init(|| {
        let m: &'static Mutex<Usage> = Box::leak(Box::default());
        std::thread::spawn(move || {
            let mut sys = sysinfo::System::new();
            #[cfg(windows)]
            let pdh = win::Pdh::new();
            // Показания nvidia-smi переиспользуются между замерами: лишний процесс каждые 2 с не нужен.
            let mut nvidia = Vec::new();
            for seq in 1.. {
                sys.refresh_cpu_usage();
                sys.refresh_memory();
                if seq == 1 || seq % 5 == 0 {
                    nvidia = crate::env::run("nvidia-smi", &["--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name", "--format=csv,noheader,nounits"], &std::env::temp_dir())
                        .map(|s| parse_nvidia(&s))
                        .unwrap_or_default();
                }
                #[cfg(windows)]
                let gpus = match &pdh {
                    Some(p) => p.sample(nvidia.clone()),
                    None => nvidia.clone(),
                };
                #[cfg(not(windows))]
                let gpus = nvidia.clone();
                *m.lock().unwrap() = Usage { seq, cpu: sys.global_cpu_usage(), ram_used: sys.used_memory(), ram_total: sys.total_memory(), gpus };
                std::thread::sleep(Duration::from_secs(2));
            }
        });
        m
    });
    cell.lock().unwrap().clone()
}

/// Строки `util, used, total, temp, name`; «[N/A]» у отдельных полей (ноутбучные карты) не ломает строку.
fn parse_nvidia(s: &str) -> Vec<GpuUsage> {
    s.lines()
        .filter_map(|l| {
            let parts: Vec<&str> = l.splitn(5, ',').map(str::trim).collect();
            let f: Vec<Option<f64>> = parts.iter().take(4).map(|x| x.parse().ok()).collect();
            Some(GpuUsage {
                name: parts.get(4).map_or("NVIDIA".into(), |n| n.to_string()),
                util: f.first()?.unwrap_or(0.) as f32,
                mem_used_mb: (*f.get(1)?)?,
                mem_total_mb: (*f.get(2)?)?,
                temp: f.get(3).copied().flatten().map(|t| t as f32),
                shared: false,
            })
        })
        .collect()
}

/// Загрузка адаптера как в Диспетчере задач: сумма по процессам внутри типа движка (3D, Compute, Copy…), максимум по типам.
/// `engines` — (имя экземпляра счётчика GPU Engine, %).
fn adapter_util(luid: &str, engines: &[(String, f64)]) -> f32 {
    let mut by_type: Vec<(&str, f64)> = vec![];
    for (inst, v) in engines.iter().filter(|(i, _)| i.to_lowercase().contains(luid)) {
        let ty = inst.rsplit("engtype_").next().unwrap_or("");
        match by_type.iter_mut().find(|t| t.0 == ty) {
            Some(t) => t.1 += v,
            None => by_type.push((ty, *v)),
        }
    }
    by_type.iter().map(|t| t.1).fold(0., f64::max).min(100.) as f32
}

/// Ключ адаптера в именах экземпляров счётчиков: `luid_0x<High>_0x<Low>`, в нижнем регистре.
fn luid_key(high: i32, low: u32) -> String {
    format!("luid_0x{:08x}_0x{low:08x}", high as u32)
}

#[cfg(windows)]
mod win {
    use super::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Performance::*;
    use windows::core::{PCWSTR, w};

    struct Adapter {
        name: String,
        nvidia: bool,
        luid: String,
        dedicated: f64,
        shared: f64,
    }

    pub struct Pdh {
        query: PDH_HQUERY,
        engine: PDH_HCOUNTER,
        dedicated: PDH_HCOUNTER,
        shared: PDH_HCOUNTER,
        adapters: Vec<Adapter>,
    }

    // Запросом пользуется только поток монитора.
    unsafe impl Send for Pdh {}

    impl Drop for Pdh {
        fn drop(&mut self) {
            unsafe { PdhCloseQuery(self.query) };
        }
    }

    impl Pdh {
        pub fn new() -> Option<Self> {
            let adapters = adapters();
            unsafe {
                let mut query = PDH_HQUERY::default();
                if PdhOpenQueryW(PCWSTR::null(), 0, &mut query) != 0 {
                    return None;
                }
                let add = |path: PCWSTR| {
                    let mut c = PDH_HCOUNTER::default();
                    (PdhAddEnglishCounterW(query, path, 0, &mut c) == 0).then_some(c)
                };
                let (Some(engine), Some(dedicated), Some(shared)) = (
                    add(w!("\\GPU Engine(*)\\Utilization Percentage")),
                    add(w!("\\GPU Adapter Memory(*)\\Dedicated Usage")),
                    add(w!("\\GPU Adapter Memory(*)\\Shared Usage")),
                ) else {
                    PdhCloseQuery(query);
                    return None;
                };
                PdhCollectQueryData(query); // проценты считаются между двумя замерами
                Some(Self { query, engine, dedicated, shared, adapters })
            }
        }

        /// Все адаптеры по счётчикам Windows; у NVIDIA загрузка, память и температура — из nvidia-smi, если он есть.
        pub fn sample(&self, nvidia: Vec<GpuUsage>) -> Vec<GpuUsage> {
            unsafe { PdhCollectQueryData(self.query) };
            let (engine, dedicated, shared) = (values(self.engine), values(self.dedicated), values(self.shared));
            let mem = |vals: &[(String, f64)], luid: &str| vals.iter().filter(|(i, _)| i.to_lowercase().contains(luid)).map(|v| v.1).sum::<f64>();
            let mut nvidia = nvidia.into_iter();
            self.adapters
                .iter()
                .map(|a| {
                    if a.nvidia && let Some(n) = nvidia.next() {
                        return GpuUsage { name: a.name.clone(), ..n };
                    }
                    // Встроенной видеокарте выделяется общая память, а не своя.
                    let own = a.dedicated >= 512. * 1024. * 1024.;
                    let (used, total) = if own { (mem(&dedicated, &a.luid), a.dedicated) } else { (mem(&shared, &a.luid), a.shared) };
                    GpuUsage { name: a.name.clone(), util: adapter_util(&a.luid, &engine), mem_used_mb: used / 1048576., mem_total_mb: total / 1048576., temp: None, shared: !own }
                })
                .collect()
        }
    }

    /// Значения счётчика со звёздочкой по всем экземплярам.
    fn values(counter: PDH_HCOUNTER) -> Vec<(String, f64)> {
        unsafe {
            let (mut size, mut count) = (0u32, 0u32);
            if PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut size, &mut count, None) != PDH_MORE_DATA {
                return vec![];
            }
            // Буфер выровнен под элементы; строки имён лежат в нём же, за массивом.
            let mut buf = vec![0u64; (size as usize).div_ceil(8)];
            let items = buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
            if PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut size, &mut count, Some(items)) != 0 {
                return vec![];
            }
            std::slice::from_raw_parts(items, count as usize)
                .iter()
                .filter(|it| it.FmtValue.CStatus == 0)
                .map(|it| (it.szName.to_string().unwrap_or_default(), it.FmtValue.Anonymous.doubleValue))
                .collect()
        }
    }

    fn adapters() -> Vec<Adapter> {
        let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else { return vec![] };
        (0..)
            .map_while(|i| unsafe { factory.EnumAdapters1(i) }.ok())
            .filter_map(|a| unsafe { a.GetDesc1() }.ok())
            .filter(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0) // Microsoft Basic Render Driver
            .map(|d| {
                let len = d.Description.iter().position(|&c| c == 0).unwrap_or(d.Description.len());
                Adapter {
                    name: String::from_utf16_lossy(&d.Description[..len]).trim().to_string(),
                    nvidia: d.VendorId == 0x10DE,
                    luid: luid_key(d.AdapterLuid.HighPart, d.AdapterLuid.LowPart),
                    dedicated: d.DedicatedVideoMemory as f64,
                    shared: d.SharedSystemMemory as f64,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia() {
        let g = super::parse_nvidia("87, 10240, 12227, 71, NVIDIA GeForce RTX 5070\n[N/A], 512, 4096, [N/A], Quadro, T1000\ngarbage\n");
        assert_eq!(g.len(), 2);
        assert_eq!(g[0], GpuUsage { name: "NVIDIA GeForce RTX 5070".into(), util: 87., mem_used_mb: 10240., mem_total_mb: 12227., temp: Some(71.), shared: false });
        assert_eq!((g[1].util, g[1].temp, g[1].name.as_str()), (0., None, "Quadro, T1000"));
    }

    /// Настоящие счётчики этой машины: cargo test real_gpus -- --ignored --nocapture
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn real_gpus() {
        let pdh = win::Pdh::new().expect("PDH-запрос не открылся");
        std::thread::sleep(Duration::from_secs(1));
        let g = pdh.sample(vec![]);
        println!("{g:#?}");
        assert!(!g.is_empty(), "DXGI не вернул ни одного адаптера");
    }

    #[test]
    fn util_sums_within_engine_type_and_takes_max() {
        let k = luid_key(0, 0xD1B5);
        assert_eq!(k, "luid_0x00000000_0x0000d1b5");
        let e = |s: &str, v: f64| (s.to_string(), v);
        let engines = [
            e("pid_1_luid_0x00000000_0x0000D1B5_phys_0_eng_0_engtype_3D", 20.),
            e("pid_2_luid_0x00000000_0x0000D1B5_phys_0_eng_0_engtype_3D", 15.),
            e("pid_2_luid_0x00000000_0x0000D1B5_phys_0_eng_3_engtype_Compute_0", 30.),
            e("pid_3_luid_0x00000000_0x0000AAAA_phys_0_eng_0_engtype_3D", 99.), // другой адаптер
        ];
        assert_eq!(adapter_util(&k, &engines), 35.);
    }
}
