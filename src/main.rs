#![windows_subsystem = "windows"]

use anyhow::Result;
use clap::Parser;
use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

const ESTIMATED_IMAGE_BYTES: usize = 10 * 1024 * 1024;
const LOADING_REPAINT_MS: u64 = 50;
const PRELOAD_WORKER_COUNT: usize = 12;
const PRIORITY_WORKER_COUNT: usize = 2;
const MAX_IN_FLIGHT: usize = PRELOAD_WORKER_COUNT + PRIORITY_WORKER_COUNT;
const PREVIOUS_CACHE_BYTES: usize = 80 * 1024 * 1024;
const NEXT_CACHE_BYTES: usize = 80 * 1024 * 1024;
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff"];

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory to scan for images
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

#[derive(Debug, Clone)]
enum PreloadMsg {
    UpdateStatus {
        current_index: usize,
        cached_sizes: HashMap<usize, usize>,
    },
    RequestImage(usize),
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Backward,
}

struct PreloadState {
    current_idx: usize,
    cached_sizes: HashMap<usize, usize>,
    in_flight: HashSet<usize>,
    direction: Direction,
}

struct CancelSignals {
    current_idx: AtomicUsize,
    image_count: AtomicUsize,
}

struct PreloadDecodeContext<'a> {
    paths: &'a [PathBuf],
    decoded_tx: &'a Sender<DecodedImage>,
    done_tx: &'a Sender<(usize, usize)>,
    repaint_ctx: &'a egui::Context,
    preload_pool: &'a rayon::ThreadPool,
    priority_pool: &'a rayon::ThreadPool,
    cancel: &'a Arc<CancelSignals>,
}

fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            IMAGE_EXTENSIONS
                .iter()
                .any(|candidate| ext.eq_ignore_ascii_case(candidate))
        })
        .unwrap_or(false)
}

fn spawn_decode_task(
    pool: &rayon::ThreadPool,
    cancel: Arc<CancelSignals>,
    decoded_tx: Sender<DecodedImage>,
    done_tx: Sender<(usize, usize)>,
    repaint_ctx: egui::Context,
    idx: usize,
    path: PathBuf,
) {
    pool.spawn(move || {
        let cur = cancel.current_idx.load(Ordering::Relaxed);
        let count = cancel.image_count.load(Ordering::Relaxed);
        if !is_within_window(idx, cur, count) {
            let _ = done_tx.send((idx, 0));
            repaint_ctx.request_repaint();
            return;
        }

        match decode_image(&path) {
            Ok((width, height, rgba, size_bytes)) => {
                let cur = cancel.current_idx.load(Ordering::Relaxed);
                let count = cancel.image_count.load(Ordering::Relaxed);
                if !is_within_window(idx, cur, count) {
                    let _ = done_tx.send((idx, 0));
                    repaint_ctx.request_repaint();
                    return;
                }
                let _ = decoded_tx.send(DecodedImage {
                    index: idx,
                    width,
                    height,
                    rgba,
                    size_bytes,
                });
                let _ = done_tx.send((idx, size_bytes));
                repaint_ctx.request_repaint();
            }
            Err(_) => {
                let _ = done_tx.send((idx, 0));
                repaint_ctx.request_repaint();
            }
        }
    });
}

fn queue_decode_if_needed(
    idx: usize,
    context: &PreloadDecodeContext<'_>,
    state: &mut PreloadState,
    priority: bool,
) -> bool {
    if idx >= context.paths.len()
        || state.cached_sizes.contains_key(&idx)
        || state.in_flight.contains(&idx)
    {
        return false;
    }

    state.in_flight.insert(idx);
    let pool = if priority {
        context.priority_pool
    } else {
        context.preload_pool
    };
    spawn_decode_task(
        pool,
        context.cancel.clone(),
        context.decoded_tx.clone(),
        context.done_tx.clone(),
        context.repaint_ctx.clone(),
        idx,
        context.paths[idx].clone(),
    );
    true
}

fn detect_direction(prev: usize, new: usize, count: usize) -> Direction {
    if count <= 1 || prev == new {
        return Direction::Forward;
    }
    let forward = (new + count - prev) % count;
    let backward = (prev + count - new) % count;
    if forward <= backward {
        Direction::Forward
    } else {
        Direction::Backward
    }
}

fn handle_preload_msg(
    msg: PreloadMsg,
    context: &PreloadDecodeContext<'_>,
    state: &mut PreloadState,
) {
    match msg {
        PreloadMsg::UpdateStatus {
            current_index,
            cached_sizes,
        } => {
            if state.current_idx != current_index {
                let count = context.paths.len();
                state.direction = detect_direction(state.current_idx, current_index, count);
                state
                    .in_flight
                    .retain(|&idx| is_within_window(idx, current_index, count));
                context
                    .cancel
                    .current_idx
                    .store(current_index, Ordering::Relaxed);
            }
            state.current_idx = current_index;
            state.cached_sizes = cached_sizes;
        }
        PreloadMsg::RequestImage(idx) => {
            if idx != state.current_idx {
                return;
            }
            queue_decode_if_needed(idx, context, state, true);
        }
    }
}

#[derive(Clone, Copy)]
enum CacheSide {
    Previous,
    Next,
}

fn side_index(current_idx: usize, image_count: usize, side: CacheSide, offset: usize) -> usize {
    match side {
        CacheSide::Previous => (current_idx + image_count - (offset % image_count)) % image_count,
        CacheSide::Next => (current_idx + offset) % image_count,
    }
}

fn side_budget(side: CacheSide) -> usize {
    match side {
        CacheSide::Previous => PREVIOUS_CACHE_BYTES,
        CacheSide::Next => NEXT_CACHE_BYTES,
    }
}

fn side_preload_usage(state: &PreloadState, image_count: usize, side: CacheSide) -> usize {
    let mut used = 0usize;

    for offset in 1..image_count {
        let idx = side_index(state.current_idx, image_count, side, offset);

        if let Some(&size) = state.cached_sizes.get(&idx) {
            used = used.saturating_add(size);
        } else if state.in_flight.contains(&idx) {
            used = used.saturating_add(ESTIMATED_IMAGE_BYTES);
        } else {
            break;
        }
    }

    used
}

fn priority_score(side: CacheSide, offset: usize, direction: Direction) -> usize {
    let preferred = matches!(
        (side, direction),
        (CacheSide::Next, Direction::Forward) | (CacheSide::Previous, Direction::Backward)
    );
    if preferred {
        offset
    } else {
        offset.saturating_mul(4)
    }
}

fn next_preload_index(state: &PreloadState, image_count: usize) -> Option<usize> {
    if !state.cached_sizes.contains_key(&state.current_idx)
        && !state.in_flight.contains(&state.current_idx)
    {
        return Some(state.current_idx);
    }

    let previous_used = side_preload_usage(state, image_count, CacheSide::Previous);
    let next_used = side_preload_usage(state, image_count, CacheSide::Next);

    let max_next_offset = NEXT_CACHE_BYTES / ESTIMATED_IMAGE_BYTES + 1;
    let max_prev_offset = PREVIOUS_CACHE_BYTES / ESTIMATED_IMAGE_BYTES + 1;
    let max_offset = max_next_offset.max(max_prev_offset).min(image_count);

    let mut best: Option<(usize, usize)> = None;

    for offset in 1..max_offset {
        for side in [CacheSide::Next, CacheSide::Previous] {
            let idx = side_index(state.current_idx, image_count, side, offset);
            let used = match side {
                CacheSide::Previous => previous_used,
                CacheSide::Next => next_used,
            };

            if state.cached_sizes.contains_key(&idx) || state.in_flight.contains(&idx) {
                continue;
            }
            if used.saturating_add(ESTIMATED_IMAGE_BYTES) > side_budget(side) {
                continue;
            }

            let score = priority_score(side, offset, state.direction);
            match best {
                None => best = Some((score, idx)),
                Some((s, _)) if score < s => best = Some((score, idx)),
                _ => {}
            }
        }
        if let Some((s, _)) = best
            && s <= offset + 1
        {
            return best.map(|(_, idx)| idx);
        }
    }

    best.map(|(_, idx)| idx)
}

fn is_within_window(idx: usize, current_idx: usize, image_count: usize) -> bool {
    if image_count == 0 {
        return false;
    }
    if idx == current_idx {
        return true;
    }
    let max_next = NEXT_CACHE_BYTES / ESTIMATED_IMAGE_BYTES;
    let max_prev = PREVIOUS_CACHE_BYTES / ESTIMATED_IMAGE_BYTES;
    let forward_dist = (idx + image_count - current_idx) % image_count;
    let backward_dist = (current_idx + image_count - idx) % image_count;
    forward_dist <= max_next || backward_dist <= max_prev
}

fn mark_decode_done(idx: usize, state: &mut PreloadState) {
    state.in_flight.remove(&idx);
}

fn collect_keep_indices(
    cache: &HashMap<usize, (egui::TextureHandle, usize)>,
    current_index: usize,
    image_count: usize,
) -> HashSet<usize> {
    let mut keep = HashSet::new();
    keep.insert(current_index);

    collect_side_keep(
        cache,
        current_index,
        image_count,
        CacheSide::Previous,
        PREVIOUS_CACHE_BYTES,
        &mut keep,
    );
    collect_side_keep(
        cache,
        current_index,
        image_count,
        CacheSide::Next,
        NEXT_CACHE_BYTES,
        &mut keep,
    );

    keep
}

fn collect_side_keep(
    cache: &HashMap<usize, (egui::TextureHandle, usize)>,
    current_index: usize,
    image_count: usize,
    side: CacheSide,
    budget: usize,
    keep: &mut HashSet<usize>,
) {
    let mut used = 0usize;

    for offset in 1..image_count {
        let idx = side_index(current_index, image_count, side, offset);

        if keep.contains(&idx) {
            continue;
        }

        let Some((_, size)) = cache.get(&idx) else {
            continue;
        };

        if used == 0 || used.saturating_add(*size) <= budget {
            keep.insert(idx);
            used = used.saturating_add(*size);
        } else {
            break;
        }
    }
}

fn scan_directory(path: &Path) -> Vec<PathBuf> {
    let mut images = Vec::new();
    let entries = if path.is_dir() {
        std::fs::read_dir(path)
    } else if let Some(parent) = path.parent() {
        std::fs::read_dir(parent)
    } else {
        std::fs::read_dir(".")
    };

    if let Ok(entries) = entries {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && is_image_file(&path) {
                images.push(path);
            }
        }
    }
    images.sort_by_cached_key(|path| path.to_string_lossy().to_ascii_lowercase());
    images
}

fn decode_image(path: &Path) -> Result<(u32, u32, Vec<u8>, usize)> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let (w, h, pixels) = if ext.eq_ignore_ascii_case("jpg") || ext.eq_ignore_ascii_case("jpeg") {
        let options = zune_core::options::DecoderOptions::default()
            .jpeg_set_out_colorspace(zune_core::colorspace::ColorSpace::RGBA);
        let mut decoder =
            zune_jpeg::JpegDecoder::new_with_options(std::io::Cursor::new(&mmap[..]), options);
        let p = decoder.decode()?;
        let dim = decoder
            .dimensions()
            .ok_or_else(|| anyhow::anyhow!("No dimensions"))?;
        (dim.0 as u32, dim.1 as u32, p)
    } else if ext.eq_ignore_ascii_case("png") {
        let mut decoder = zune_png::PngDecoder::new(std::io::Cursor::new(&mmap[..]));
        let pixels_res = decoder.decode()?;
        let (w, h) = decoder
            .dimensions()
            .ok_or_else(|| anyhow::anyhow!("No dimensions"))?;
        let colorspace = decoder
            .colorspace()
            .unwrap_or(zune_core::colorspace::ColorSpace::Unknown);

        if let zune_core::result::DecodingResult::U8(data) = pixels_res {
            if colorspace == zune_core::colorspace::ColorSpace::RGBA {
                (w as u32, h as u32, data)
            } else if colorspace == zune_core::colorspace::ColorSpace::RGB {
                let mut rgba = Vec::with_capacity(w * h * 4);
                for chunk in data.chunks_exact(3) {
                    rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
                }
                (w as u32, h as u32, rgba)
            } else {
                let img = image::load_from_memory(&mmap)?;
                let rgba8 = img.into_rgba8();
                (rgba8.width(), rgba8.height(), rgba8.into_raw())
            }
        } else {
            let img = image::load_from_memory(&mmap)?;
            let rgba8 = img.into_rgba8();
            (rgba8.width(), rgba8.height(), rgba8.into_raw())
        }
    } else {
        let img = image::load_from_memory(&mmap)?;
        let rgba8 = img.into_rgba8();
        (rgba8.width(), rgba8.height(), rgba8.into_raw())
    };

    let size = pixels.len();
    Ok((w, h, pixels, size))
}

struct PicDashApp {
    image_paths: Vec<PathBuf>,
    current_index: usize,
    cache: HashMap<usize, (egui::TextureHandle, usize)>,
    decoded_rx: Receiver<DecodedImage>,
    msg_tx: Sender<PreloadMsg>,
    last_requested_index: Option<usize>,
    last_title_index: Option<usize>,
}

impl PicDashApp {
    fn new(cc: &eframe::CreationContext<'_>, args: Args) -> Self {
        let image_paths = scan_directory(&args.path);
        let mut current_index = 0;

        if args.path.is_file()
            && let Some(idx) = image_paths.iter().position(|p| p == &args.path)
        {
            current_index = idx;
        }

        let (decoded_tx, decoded_rx) = crossbeam_channel::bounded(MAX_IN_FLIGHT * 2);
        let (msg_tx, msg_rx) = crossbeam_channel::unbounded();

        let paths_clone = image_paths.clone();
        let repaint_ctx = cc.egui_ctx.clone();
        let cancel = Arc::new(CancelSignals {
            current_idx: AtomicUsize::new(current_index),
            image_count: AtomicUsize::new(image_paths.len()),
        });

        thread::spawn(move || {
            let preload_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(PRELOAD_WORKER_COUNT)
                .thread_name(|i| format!("picdash-preload-{i}"))
                .build()
                .unwrap();
            let priority_pool = rayon::ThreadPoolBuilder::new()
                .num_threads(PRIORITY_WORKER_COUNT)
                .thread_name(|i| format!("picdash-priority-{i}"))
                .build()
                .unwrap();

            let (done_tx, done_rx) = crossbeam_channel::unbounded::<(usize, usize)>();
            let context = PreloadDecodeContext {
                paths: &paths_clone,
                decoded_tx: &decoded_tx,
                done_tx: &done_tx,
                repaint_ctx: &repaint_ctx,
                preload_pool: &preload_pool,
                priority_pool: &priority_pool,
                cancel: &cancel,
            };
            let mut state = PreloadState {
                current_idx: current_index,
                cached_sizes: HashMap::new(),
                in_flight: HashSet::new(),
                direction: Direction::Forward,
            };

            loop {
                loop {
                    crossbeam_channel::select! {
                        recv(msg_rx) -> msg => {
                            if let Ok(msg) = msg {
                                handle_preload_msg(msg, &context, &mut state);
                            }
                        }
                        recv(done_rx) -> res => {
                            if let Ok((idx, _size)) = res {
                                mark_decode_done(idx, &mut state);
                            }
                        }
                        default(std::time::Duration::from_millis(1)) => break,
                    }
                }

                let count = context.paths.len();
                if count > 0 {
                    if !state.cached_sizes.contains_key(&state.current_idx)
                        && !state.in_flight.contains(&state.current_idx)
                    {
                        queue_decode_if_needed(state.current_idx, &context, &mut state, true);
                    }

                    while state.in_flight.len() < MAX_IN_FLIGHT {
                        if let Some(idx) = next_preload_index(&state, count) {
                            let priority = idx == state.current_idx;
                            if !queue_decode_if_needed(idx, &context, &mut state, priority) {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }

                crossbeam_channel::select! {
                    recv(msg_rx) -> msg => {
                        if let Ok(msg) = msg {
                            handle_preload_msg(msg, &context, &mut state);
                        }
                    }
                    recv(done_rx) -> res => {
                        if let Ok((idx, _size)) = res {
                            mark_decode_done(idx, &mut state);
                        }
                    }
                }
            }
        });

        let _ = msg_tx.send(PreloadMsg::UpdateStatus {
            current_index,
            cached_sizes: HashMap::new(),
        });

        Self {
            image_paths,
            current_index,
            cache: HashMap::new(),
            decoded_rx,
            msg_tx,
            last_requested_index: None,
            last_title_index: None,
        }
    }

    fn send_status(&self) {
        let cached_sizes = self
            .cache
            .iter()
            .map(|(&idx, (_, size))| (idx, *size))
            .collect();
        let _ = self.msg_tx.send(PreloadMsg::UpdateStatus {
            current_index: self.current_index,
            cached_sizes,
        });
    }

    fn add_to_cache(&mut self, ctx: &egui::Context, decoded: DecodedImage) -> bool {
        if self.cache.contains_key(&decoded.index) {
            return false;
        }

        let size = [decoded.width as usize, decoded.height as usize];
        let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &decoded.rgba);
        let texture = ctx.load_texture(
            format!("picdash_{}", decoded.index),
            color_image,
            egui::TextureOptions::LINEAR,
        );

        self.cache
            .insert(decoded.index, (texture, decoded.size_bytes));
        self.prune_cache_window();
        true
    }

    fn prune_cache_window(&mut self) -> bool {
        if self.image_paths.is_empty() {
            return false;
        }

        let keep = collect_keep_indices(&self.cache, self.current_index, self.image_paths.len());
        let before = self.cache.len();
        self.cache.retain(|idx, _| keep.contains(idx));

        before != self.cache.len()
    }

    fn update_window_title(&mut self, ctx: &egui::Context) {
        if self.last_title_index == Some(self.current_index) {
            return;
        }

        let filename = self.image_paths[self.current_index]
            .file_name()
            .unwrap()
            .to_string_lossy();
        let title = format!(
            "[{}/{}] {} - picdash",
            self.current_index + 1,
            self.image_paths.len(),
            filename
        );
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
        self.last_title_index = Some(self.current_index);
    }
}

impl eframe::App for PicDashApp {
    fn ui(&mut self, root_ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = root_ui.ctx().clone();
        let mut cache_changed = false;
        while let Ok(decoded) = self.decoded_rx.try_recv() {
            cache_changed |= self.add_to_cache(&ctx, decoded);
        }

        if cache_changed {
            self.send_status();
        }

        egui::CentralPanel::no_frame()
            .frame(egui::Frame::NONE.fill(egui::Color32::BLACK))
            .show_inside(root_ui, |ui| {
                if self.image_paths.is_empty() {
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
                    self.current_index = (self.current_index + 1) % self.image_paths.len();
                    changed = true;
                }
                if ui.input(|i| {
                    i.key_pressed(egui::Key::ArrowLeft) || i.key_pressed(egui::Key::Backspace)
                }) {
                    self.current_index = if self.current_index == 0 {
                        self.image_paths.len() - 1
                    } else {
                        self.current_index - 1
                    };
                    changed = true;
                }

                if changed {
                    self.last_requested_index = None;
                    self.prune_cache_window();
                    self.send_status();
                }

                self.update_window_title(ui.ctx());

                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
                if ui.input(|i| i.key_pressed(egui::Key::F)) {
                    let is_fullscreen = ui.input(|i| i.viewport().fullscreen).unwrap_or(false);
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::Fullscreen(!is_fullscreen));
                }

                let rect = ui.max_rect();
                if let Some((texture, _)) = self.cache.get(&self.current_index) {
                    let img_size = texture.size_vec2();
                    let scale = (rect.width() / img_size.x).min(rect.height() / img_size.y);
                    let scaled_size = img_size * scale;
                    let img_rect = egui::Rect::from_center_size(rect.center(), scaled_size);

                    ui.painter().image(
                        texture.id(),
                        img_rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.add(egui::Spinner::new().size(40.0));
                    });
                    if self.last_requested_index != Some(self.current_index) {
                        let _ = self
                            .msg_tx
                            .send(PreloadMsg::RequestImage(self.current_index));
                        self.last_requested_index = Some(self.current_index);
                    }
                    ctx.request_repaint_after(Duration::from_millis(LOADING_REPAINT_MS));
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
