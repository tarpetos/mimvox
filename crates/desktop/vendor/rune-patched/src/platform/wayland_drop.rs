use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::os::fd::{AsFd, AsRawFd};
use std::path::PathBuf;
use std::time::Duration;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use wayland_backend::client::{Backend, ObjectId};
use wayland_client::globals::{GlobalListContents, registry_queue_init};
use wayland_client::protocol::wl_data_device::{self, WlDataDevice};
use wayland_client::protocol::wl_data_device_manager::{DndAction, WlDataDeviceManager};
use wayland_client::protocol::wl_data_offer::{self, WlDataOffer};
use wayland_client::protocol::wl_registry::WlRegistry;
use wayland_client::protocol::wl_seat::WlSeat;
use wayland_client::protocol::wl_surface::WlSurface;
use wayland_client::{Connection, Dispatch, EventQueue, Proxy, QueueHandle};
use winit::window::Window;

const URI_LIST: &str = "text/uri-list";

pub(crate) enum Step {
    Enter(Vec<PathBuf>, (f32, f32)),
    Move((f32, f32)),
    Drop(Vec<PathBuf>, (f32, f32)),
    Leave,
}

struct Current {
    offer: WlDataOffer,
    serial: u32,
    paths: Option<Vec<PathBuf>>,
    reading: Option<std::sync::mpsc::Receiver<Vec<u8>>>,
    at: (f32, f32),
}

struct State {
    surface: WlSurface,
    mimes: HashMap<ObjectId, Vec<String>>,
    current: Option<Current>,
    selection: Option<WlDataOffer>,
    steps: Vec<Step>,
}

struct Binding {
    steps: std::sync::Arc<std::sync::Mutex<Vec<Step>>>,
}

thread_local! {
    static BINDINGS: RefCell<HashMap<usize, Option<Binding>>> = RefCell::new(HashMap::new());
}

pub(crate) fn take(window: &Window) -> Vec<Step> {
    let Ok(handle) = window.window_handle() else { return Vec::new() };
    let RawWindowHandle::Wayland(w) = handle.as_raw() else { return Vec::new() };
    let key = w.surface.as_ptr() as usize;
    BINDINGS.with(|b| {
        let mut b = b.borrow_mut();
        let Some(binding) = b.entry(key).or_insert_with(|| bind(window)) else { return Vec::new() };
        std::mem::take(&mut *binding.steps.lock().unwrap_or_else(|e| e.into_inner()))
    })
}

fn debug() -> bool {
    std::env::var_os("MIMVOX_DEBUG_DROP").is_some()
}

fn bind(window: &Window) -> Option<Binding> {
    let b = bind_inner(window);
    if debug() {
        eprintln!("wayland drop: bound {}", b.is_some());
    }
    b
}

fn bind_inner(window: &Window) -> Option<Binding> {
    let (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(handle)) =
        (window.display_handle().ok()?.as_raw(), window.window_handle().ok()?.as_raw())
    else {
        return None;
    };
    // SAFETY: winit owns this `wl_display` and keeps it alive for the event loop, which outlives every window; a
    // foreign backend never closes it.
    let backend = unsafe { Backend::from_foreign_display(display.display.as_ptr().cast()) };
    let conn = Connection::from_backend(backend);
    let (globals, queue) = registry_queue_init::<State>(&conn).ok()?;
    let qh = queue.handle();
    let seat = globals.bind::<WlSeat, _, _>(&qh, 1..=7, ()).ok()?;
    let manager = globals.bind::<WlDataDeviceManager, _, _>(&qh, 1..=3, ()).ok()?;
    let device = manager.get_data_device(&seat, &qh, ());
    // SAFETY: the pointer is the live `wl_surface` proxy winit created for this window, which outlives the binding's
    // use on the UI thread.
    let id = unsafe { ObjectId::from_ptr(WlSurface::interface(), handle.surface.as_ptr().cast()) }.ok()?;
    let surface = WlSurface::from_id(&conn, id).ok()?;
    let _ = conn.flush();
    let mut state = State { surface, mimes: HashMap::new(), current: None, selection: None, steps: Vec::new() };
    let steps = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let out = steps.clone();
    let mut queue = queue;
    // PATCH: answers the compositor at once; the UI thread only wakes now and then during a drag
    std::thread::Builder::new()
        .name("wayland-drop".into())
        .spawn(move || {
            let _device = device;
            while queue.dispatch_pending(&mut state).is_ok() {
                state.poll_read();
                if !state.steps.is_empty() {
                    out.lock().unwrap_or_else(|e| e.into_inner()).append(&mut state.steps);
                }
                // PATCH: winit destroys the surface before it closes the display; stop reading first
                if !state.surface.is_alive() || queue.flush().is_err() {
                    break;
                }
                let Some(guard) = queue.prepare_read() else { continue };
                let mut fds = [libc::pollfd { fd: guard.connection_fd().as_raw_fd(), events: libc::POLLIN, revents: 0 }];
                // SAFETY: one valid pollfd for the duration of the call.
                let ready = unsafe { libc::poll(fds.as_mut_ptr(), 1, 100) };
                if ready > 0 && state.surface.is_alive() && guard.read().is_err() {
                    break;
                }
            }
            // PATCH: dropping them would destroy proxies on the display winit has closed by now
            std::mem::forget((queue, state, _device));
        })
        .ok()?;
    Some(Binding { steps })
}

impl State {
    fn poll_read(&mut self) {
        let Some(c) = self.current.as_mut() else { return };
        if c.paths.is_some() {
            return;
        }
        let Some(bytes) = c.reading.as_ref().and_then(|r| r.try_recv().ok()) else { return };
        let paths = uri_list(&bytes);
        if debug() {
            eprintln!("wayland drop: paths {paths:?}");
        }
        self.steps.push(Step::Enter(paths.clone(), c.at));
        c.paths = Some(paths);
    }
}

fn start_read(offer: &WlDataOffer, conn: &Connection) -> Option<std::sync::mpsc::Receiver<Vec<u8>>> {
    let (mut reader, writer) = std::io::pipe().ok()?;
    offer.receive(URI_LIST.to_owned(), writer.as_fd());
    let _ = conn.flush();
    drop(writer);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = reader.read_to_end(&mut bytes);
        let _ = tx.send(bytes);
    });
    Some(rx)
}

fn uri_list(bytes: &[u8]) -> Vec<PathBuf> {
    String::from_utf8_lossy(bytes)
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.strip_prefix("file://"))
        .filter_map(|rest| rest.find('/').map(|slash| &rest[slash..]))
        .map(|path| {
            use std::os::unix::ffi::OsStringExt;
            PathBuf::from(std::ffi::OsString::from_vec(percent_decode(path)))
        })
        .collect()
}

fn percent_decode(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16);
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

impl Dispatch<WlDataDevice, ()> for State {
    fn event(
        state: &mut Self,
        _: &WlDataDevice,
        event: wl_data_device::Event,
        _: &(),
        conn: &Connection,
        _: &QueueHandle<Self>,
    ) {
        match event {
            wl_data_device::Event::Enter { serial, surface, x, y, id } => {
                let Some(offer) = id else { return };
                let files = state.mimes.get(&offer.id()).is_some_and(|m| m.iter().any(|m| m == URI_LIST));
                let ours = surface.id().protocol_id() == state.surface.id().protocol_id();
                if debug() {
                    eprintln!("wayland drop: enter ours={ours} v{} mimes={:?}", offer.version(), state.mimes.get(&offer.id()));
                }
                if !ours || !files {
                    offer.accept(serial, None);
                    return;
                }
                offer.accept(serial, Some(URI_LIST.to_owned()));
                if offer.version() >= 3 {
                    offer.set_actions(DndAction::Copy, DndAction::Copy);
                }
                let at = (x as f32, y as f32);
                let reading = start_read(&offer, conn);
                state.current = Some(Current { offer, serial, paths: None, reading, at });
            }
            wl_data_device::Event::Motion { x, y, .. } => {
                if let Some(c) = state.current.as_mut() {
                    c.at = (x as f32, y as f32);
                    c.offer.accept(c.serial, Some(URI_LIST.to_owned()));
                    if c.paths.is_some() {
                        state.steps.push(Step::Move(c.at));
                    }
                }
            }
            wl_data_device::Event::Drop => {
                if debug() {
                    eprintln!("wayland drop: drop {}", state.current.is_some());
                }
                if let Some(c) = state.current.take() {
                    let paths = c.paths.or_else(|| {
                        c.reading.and_then(|r| r.recv_timeout(Duration::from_secs(2)).ok()).map(|b| uri_list(&b))
                    });
                    if debug() {
                        eprintln!("wayland drop: dropped {paths:?}");
                    }
                    if let Some(paths) = paths.filter(|p| !p.is_empty()) {
                        state.steps.push(Step::Drop(paths, c.at));
                    } else {
                        state.steps.push(Step::Leave);
                    }
                    if c.offer.version() >= 3 {
                        c.offer.finish();
                    }
                    state.mimes.remove(&c.offer.id());
                }
            }
            wl_data_device::Event::Leave => {
                if debug() {
                    eprintln!("wayland drop: leave");
                }
                if let Some(c) = state.current.take() {
                    if c.paths.is_some() {
                        state.steps.push(Step::Leave);
                    }
                    state.mimes.remove(&c.offer.id());
                }
            }
            wl_data_device::Event::Selection { id } => {
                if let Some(old) = std::mem::replace(&mut state.selection, id) {
                    state.mimes.remove(&old.id());
                }
            }
            _ => {}
        }
    }

    wayland_client::event_created_child!(State, WlDataDevice, [
        wl_data_device::EVT_DATA_OFFER_OPCODE => (WlDataOffer, ()),
    ]);
}

impl Dispatch<WlDataOffer, ()> for State {
    fn event(state: &mut Self, offer: &WlDataOffer, event: wl_data_offer::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {
        match event {
            wl_data_offer::Event::Offer { mime_type } => state.mimes.entry(offer.id()).or_default().push(mime_type),
            other => {
                if debug() {
                    eprintln!("wayland drop: offer {other:?}");
                }
            }
        }
    }
}

impl Dispatch<WlRegistry, GlobalListContents> for State {
    fn event(_: &mut Self, _: &WlRegistry, _: <WlRegistry as Proxy>::Event, _: &GlobalListContents, _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<WlSeat, ()> for State {
    fn event(_: &mut Self, _: &WlSeat, _: <WlSeat as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}

impl Dispatch<WlDataDeviceManager, ()> for State {
    fn event(_: &mut Self, _: &WlDataDeviceManager, _: <WlDataDeviceManager as Proxy>::Event, _: &(), _: &Connection, _: &QueueHandle<Self>) {}
}
