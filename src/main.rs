#![windows_subsystem = "windows"]

use anyhow::Result;
use clap::Parser;
use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

const DECODE_AHEAD: usize = 8;
const DECODE_BEHIND: usize = 8;
const PREFETCH_AHEAD: usize = 30;
const PREFETCH_BEHIND: usize = 30;
const PREFETCH_WORKERS: usize = 8;
const DECODE_WORKERS: usize = 4;
const PRIORITY_WORKERS: usize = 2;
const MAX_DECODE_RETRIES: usize = 3;
const PREFETCH_CHUNK: usize = 64 * 1024;
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"];

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(default_value = ".")]
    path: PathBuf,
}

struct DecodedImage {
    index: usize,
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    size_bytes: usize,
}

struct SharedState {
    current_idx: AtomicUsize,
    image_count: AtomicUsize,
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| IMAGE_EXTENSIONS.iter().any(|c| ext.eq_ignore_ascii_case(c)))
        .unwrap_or(false)
}

fn scan_directory(path: &Path) -> Vec<PathBuf> {
    let dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(Path::new("."))
    };
    let mut images = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.file_type().map_or(false, |ft| !ft.is_dir()) && is_image_file(&entry.path())
            {
                images.push(entry.path());
            }
        }
    }
    images.sort_by_cached_key(|p| p.to_string_lossy().to_ascii_lowercase());
    images
}

fn circular_dist(idx: usize, cur: usize, count: usize) -> usize {
    if count == 0 {
        return usize::MAX;
    }
    let fwd = (idx + count - cur) % count;
    let bwd = (cur + count - idx) % count;
    fwd.min(bwd)
}

fn ext_of(path: &Path) -> String {
    path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_owned()
}

fn window_indices(cur: usize, count: usize, fwd: usize, bwd: usize) -> Vec<usize> {
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    let max = fwd.max(bwd).min(count.saturating_sub(1));
    for offset in 1..=max {
        if offset <= fwd {
            let idx = (cur + offset) % count;
            if seen.insert(idx) {
                result.push(idx);
            }
        }
        if offset <= bwd {
            let idx = (cur + count - offset) % count;
            if seen.insert(idx) {
                result.push(idx);
            }
        }
    }
    result
}

fn prefetch_file(path: &Path, idx: usize, shared: &SharedState) -> Option<Vec<u8>> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut data = Vec::new();
    let mut buf = [0u8; PREFETCH_CHUNK];
    let max_dist = PREFETCH_AHEAD.max(PREFETCH_BEHIND);
    loop {
        let cur = shared.current_idx.load(Ordering::Relaxed);
        let count = shared.image_count.load(Ordering::Relaxed);
        if circular_dist(idx, cur, count) > max_dist {
            return None;
        }
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => data.extend_from_slice(&buf[..n]),
            Err(_) => return None,
        }
    }
    Some(data)
}

fn decode_from_bytes(data: &[u8], ext: &str) -> Result<(u32, u32, Vec<u8>, usize)> {
    let (w, h, pixels) = if ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg") {
        let options = zune_core::options::DecoderOptions::default()
            .jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGBA);
        let mut decoder =
            zune_jpeg::JpegDecoder::new_with_options(std::io::Cursor::new(data), options);
        let p = decoder.decode()?;
        let dim = decoder
            .dimensions()
            .ok_or_else(|| anyhow::anyhow!("No dimensions"))?;
        (dim.0 as u32, dim.1 as u32, p)
    } else if ext.eq_ignore_ascii_case("png") {
        let mut decoder = zune_png::PngDecoder::new(std::io::Cursor::new(data));
        let pixels_res = decoder.decode()?;
        let (w, h) = decoder
            .dimensions()
            .ok_or_else(|| anyhow::anyhow!("No dimensions"))?;
        let cs = decoder
            .colorspace()
            .unwrap_or(zune_core::colorspace::ColorSpace::Unknown);
        if let zune_core::result::DecodingResult::U8(d) = pixels_res {
            if cs == zune_core::colorspace::ColorSpace::RGBA {
                (w as u32, h as u32, d)
            } else if cs == zune_core::colorspace::ColorSpace::RGB {
                let mut rgba = Vec::with_capacity(w * h * 4);
                for chunk in d.chunks_exact(3) {
                    rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
                }
                (w as u32, h as u32, rgba)
            } else {
                let img = image::load_from_memory(data)?;
                let r = img.into_rgba8();
                (r.width(), r.height(), r.into_raw())
            }
        } else {
            let img = image::load_from_memory(data)?;
            let r = img.into_rgba8();
            (r.width(), r.height(), r.into_raw())
        }
    } else {
        let img = image::load_from_memory(data)?;
        let r = img.into_rgba8();
        (r.width(), r.height(), r.into_raw())
    };
    let size = pixels.len();
    Ok((w, h, pixels, size))
}

fn spawn_prefetch(
    pool: &rayon::ThreadPool,
    idx: usize,
    path: PathBuf,
    shared: Arc<SharedState>,
    done_tx: Sender<(usize, Option<Vec<u8>>)>,
) {
    pool.spawn(move || {
        let data = prefetch_file(&path, idx, &shared);
        let _ = done_tx.send((idx, data));
    });
}

fn spawn_decode(
    pool: &rayon::ThreadPool,
    idx: usize,
    data: Arc<Vec<u8>>,
    ext: String,
    decoded_tx: Sender<DecodedImage>,
    done_tx: Sender<(usize, bool)>,
    ctx: egui::Context,
) {
    pool.spawn(move || {
        let ok = match decode_from_bytes(&data, &ext) {
            Ok((w, h, rgba, size)) => {
                let _ = decoded_tx.send(DecodedImage {
                    index: idx,
                    width: w,
                    height: h,
                    rgba,
                    size_bytes: size,
                });
                true
            }
            Err(_) => false,
        };
        let _ = done_tx.send((idx, ok));
        ctx.request_repaint();
    });
}

fn run_coordinator(
    paths: Arc<Vec<PathBuf>>,
    shared: Arc<SharedState>,
    decoded_tx: Sender<DecodedImage>,
    repaint_ctx: egui::Context,
) {
    let prefetch_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(PREFETCH_WORKERS)
        .thread_name(|i| format!("picdash-prefetch-{i}"))
        .build()
        .unwrap();
    let decode_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(DECODE_WORKERS)
        .thread_name(|i| format!("picdash-decode-{i}"))
        .build()
        .unwrap();
    let priority_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(PRIORITY_WORKERS)
        .thread_name(|i| format!("picdash-priority-{i}"))
        .build()
        .unwrap();

    let (pf_done_tx, pf_done_rx) = crossbeam_channel::unbounded::<(usize, Option<Vec<u8>>)>();
    let (dc_done_tx, dc_done_rx) = crossbeam_channel::unbounded::<(usize, bool)>();

    let count = paths.len();
    if count == 0 {
        return;
    }

    let mut file_cache: HashMap<usize, Arc<Vec<u8>>> = HashMap::new();
    let mut pf_in_flight: HashSet<usize> = HashSet::new();
    let mut dc_in_flight: HashSet<usize> = HashSet::new();
    let mut decoded_sent: HashSet<usize> = HashSet::new();
    let mut failed: HashMap<usize, usize> = HashMap::new();
    loop {
        // Drain all pending events
        loop {
            let mut got = false;
            while let Ok((idx, data)) = pf_done_rx.try_recv() {
                pf_in_flight.remove(&idx);
                if let Some(data) = data {
                    file_cache.insert(idx, Arc::new(data));
                }
                got = true;
            }
            while let Ok((idx, success)) = dc_done_rx.try_recv() {
                dc_in_flight.remove(&idx);
                if success {
                    decoded_sent.insert(idx);
                    failed.remove(&idx);
                } else {
                    *failed.entry(idx).or_insert(0) += 1;
                }
                got = true;
            }
            if !got {
                break;
            }
        }

        let cur = shared.current_idx.load(Ordering::Relaxed);

        // Priority: current image — prefetch then decode on the fast priority pool
        if !decoded_sent.contains(&cur) && !dc_in_flight.contains(&cur) {
            if let Some(data) = file_cache.get(&cur).cloned() {
                dc_in_flight.insert(cur);
                spawn_decode(
                    &priority_pool,
                    cur,
                    data,
                    ext_of(&paths[cur]),
                    decoded_tx.clone(),
                    dc_done_tx.clone(),
                    repaint_ctx.clone(),
                );
            } else if !pf_in_flight.contains(&cur) {
                pf_in_flight.insert(cur);
                spawn_prefetch(
                    &priority_pool,
                    cur,
                    paths[cur].clone(),
                    shared.clone(),
                    pf_done_tx.clone(),
                );
            }
        }

        // Schedule prefetches symmetrically in both directions
        let max_pf = PREFETCH_WORKERS * 2;
        for idx in window_indices(cur, count, PREFETCH_AHEAD, PREFETCH_BEHIND) {
            if pf_in_flight.len() >= max_pf {
                break;
            }
            if file_cache.contains_key(&idx) || pf_in_flight.contains(&idx) {
                continue;
            }
            pf_in_flight.insert(idx);
            spawn_prefetch(
                &prefetch_pool,
                idx,
                paths[idx].clone(),
                shared.clone(),
                pf_done_tx.clone(),
            );
        }

        // Schedule decodes for prefetched files symmetrically in both directions
        let max_dc = DECODE_WORKERS + PRIORITY_WORKERS;
        for idx in window_indices(cur, count, DECODE_AHEAD, DECODE_BEHIND) {
            if dc_in_flight.len() >= max_dc {
                break;
            }
            if decoded_sent.contains(&idx) || dc_in_flight.contains(&idx) {
                continue;
            }
            if *failed.get(&idx).unwrap_or(&0) >= MAX_DECODE_RETRIES {
                continue;
            }
            let Some(data) = file_cache.get(&idx).cloned() else {
                continue;
            };
            dc_in_flight.insert(idx);
            spawn_decode(
                &decode_pool,
                idx,
                data,
                ext_of(&paths[idx]),
                decoded_tx.clone(),
                dc_done_tx.clone(),
                repaint_ctx.clone(),
            );
        }

        // Evict entries outside their windows
        let pf_max = PREFETCH_AHEAD.max(PREFETCH_BEHIND);
        let dc_max = DECODE_AHEAD.max(DECODE_BEHIND);
        file_cache.retain(|&idx, _| circular_dist(idx, cur, count) <= pf_max);
        decoded_sent.retain(|&idx| circular_dist(idx, cur, count) <= dc_max);
        failed.retain(|&idx, _| circular_dist(idx, cur, count) <= dc_max);
        pf_in_flight.retain(|&idx| circular_dist(idx, cur, count) <= pf_max);
        dc_in_flight.retain(|&idx| circular_dist(idx, cur, count) <= dc_max);

        // Tight poll when current image isn't ready, relaxed otherwise
        let timeout = if !decoded_sent.contains(&cur) {
            Duration::from_millis(1)
        } else {
            Duration::from_millis(10)
        };
        crossbeam_channel::select! {
            recv(pf_done_rx) -> msg => {
                if let Ok((idx, data)) = msg {
                    pf_in_flight.remove(&idx);
                    if let Some(data) = data {
                        file_cache.insert(idx, Arc::new(data));
                    }
                }
            }
            recv(dc_done_rx) -> msg => {
                if let Ok((idx, success)) = msg {
                    dc_in_flight.remove(&idx);
                    if success {
                        decoded_sent.insert(idx);
                        failed.remove(&idx);
                    } else {
                        *failed.entry(idx).or_insert(0) += 1;
                    }
                }
            }
            default(timeout) => {}
        }
    }
}

fn paint_image(ui: &mut egui::Ui, texture: &egui::TextureHandle, rect: egui::Rect) {
    let img_size = texture.size_vec2();
    let scale = (rect.width() / img_size.x).min(rect.height() / img_size.y);
    let scaled = img_size * scale;
    let img_rect = egui::Rect::from_center_size(rect.center(), scaled);
    ui.painter().image(
        texture.id(),
        img_rect,
        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
        egui::Color32::WHITE,
    );
}

struct PicDashApp {
    paths: Vec<PathBuf>,
    current_index: usize,
    cache: HashMap<usize, (egui::TextureHandle, usize)>,
    decoded_rx: Receiver<DecodedImage>,
    shared: Arc<SharedState>,
    last_title_index: Option<usize>,
}

impl PicDashApp {
    fn new(cc: &eframe::CreationContext<'_>, args: Args) -> Self {
        let paths = scan_directory(&args.path);
        let mut current_index = 0;

        if args.path.is_file()
            && let Some(idx) = paths.iter().position(|p| p == &args.path)
        {
            current_index = idx;
        }

        let shared = Arc::new(SharedState {
            current_idx: AtomicUsize::new(current_index),
            image_count: AtomicUsize::new(paths.len()),
        });

        let (decoded_tx, decoded_rx) = crossbeam_channel::unbounded();
        let paths_arc = Arc::new(paths.clone());
        let shared2 = shared.clone();
        let ctx = cc.egui_ctx.clone();

        thread::spawn(move || run_coordinator(paths_arc, shared2, decoded_tx, ctx));

        Self {
            paths,
            current_index,
            cache: HashMap::new(),
            decoded_rx,
            shared,
            last_title_index: None,
        }
    }

    fn prune_cache(&mut self) {
        if self.paths.is_empty() {
            return;
        }
        let count = self.paths.len();
        let cur = self.current_index;
        let max = DECODE_AHEAD.max(DECODE_BEHIND) + 2;
        self.cache
            .retain(|&idx, _| circular_dist(idx, cur, count) <= max);
    }

    fn update_title(&mut self, ctx: &egui::Context) {
        if self.last_title_index == Some(self.current_index) {
            return;
        }
        let filename = self.paths[self.current_index]
            .file_name()
            .unwrap()
            .to_string_lossy();
        let title = format!(
            "[{}/{}] {} - picdash",
            self.current_index + 1,
            self.paths.len(),
            filename
        );
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
        self.last_title_index = Some(self.current_index);
    }
}

impl eframe::App for PicDashApp {
    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root_ui.ctx().clone();

        while let Ok(decoded) = self.decoded_rx.try_recv() {
            if !self.cache.contains_key(&decoded.index) {
                let size = [decoded.width as usize, decoded.height as usize];
                let img = egui::ColorImage::from_rgba_unmultiplied(size, &decoded.rgba);
                let tex = ctx.load_texture(
                    format!("picdash_{}", decoded.index),
                    img,
                    egui::TextureOptions::LINEAR,
                );
                self.cache.insert(decoded.index, (tex, decoded.size_bytes));
            }
        }

        egui::CentralPanel::no_frame()
            .frame(egui::Frame::NONE.fill(egui::Color32::BLACK))
            .show_inside(root_ui, |ui| {
                if self.paths.is_empty() {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new("No images found")
                                .color(egui::Color32::WHITE)
                                .size(24.0),
                        );
                    });
                    return;
                }

                let mut changed = false;
                if ui.input(|i| {
                    i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::Space)
                }) {
                    self.current_index = (self.current_index + 1) % self.paths.len();
                    changed = true;
                }
                if ui.input(|i| {
                    i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::Backspace)
                }) {
                    self.current_index = if self.current_index == 0 {
                        self.paths.len() - 1
                    } else {
                        self.current_index - 1
                    };
                    changed = true;
                }

                if changed {
                    self.shared
                        .current_idx
                        .store(self.current_index, Ordering::Relaxed);
                    self.prune_cache();
                }

                self.update_title(ui.ctx());

                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
                if ui.input(|i| i.key_pressed(egui::Key::F)) {
                    let fs = ui.input(|i| i.viewport().fullscreen).unwrap_or(false);
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::Fullscreen(!fs));
                }

                let rect = ui.max_rect();
                if let Some((texture, _)) = self.cache.get(&self.current_index) {
                    paint_image(ui, texture, rect);
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.add(egui::Spinner::new().size(40.0));
                    });
                    ctx.request_repaint_after(Duration::from_millis(16));
                }
            });
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("picdash")
            .with_maximized(true)
            .with_active(true),
        ..Default::default()
    };

    eframe::run_native(
        "picdash",
        options,
        Box::new(|cc| Ok(Box::new(PicDashApp::new(cc, args)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {}", e))?;

    Ok(())
}
