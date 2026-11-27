# Vendored crates

## rune-patched

A copy of `crates/rune` from `github.com/anomalyco/rune` at the pinned rev
`a7cea1692a8ce68b1b63c46688c493bb94a8456d` (only `src/`, `assets/` and the two `.dxbc` shaders), wired in with
`[patch."https://github.com/anomalyco/rune"]` in the root `Cargo.toml`. Its `Cargo.toml` is the upstream one with the
workspace inheritance inlined, the sibling crates (`rune-motion`, `rune-staging`, `rune-taffy`, `procwatch`) taken
from the same git rev, and the examples, tests and dev-dependencies dropped. Nothing else differs except the patches
below. To go back to plain upstream, delete the `[patch]` section and this directory.

### Patch 1: the app can draw the Windows caption buttons itself

Upstream draws the three Windows caption buttons itself (flat 46 px Windows 11 look) and offers no way to restyle or
replace them. The snap layouts flyout over maximize needs a real `HTMAXBUTTON` hit test, so the buttons cannot simply
be hidden and redrawn as ordinary elements. The patch keeps all the native behaviour and only makes the drawing
optional:

- `Titlebar::custom_caption(cell_width)` (+ `custom_caption_width()`): Rune still hit-tests `cell_width` x
  `Titlebar::height` cells flush with the trailing edge as `HTMINBUTTON` / `HTMAXBUTTON` / `HTCLOSE`, tracks hover and
  press and carries out the actions, but draws nothing in them. `ChromeInsets::trailing` is the cells' total width.
- `ChromeInsets::caption: CaptionView { hovered, pressed, maximized }` and the now public `CaptionButton`: where the
  pointer is with the buttons (the system takes it as non-client input, so the view never sees it otherwise). It is part
  of the memo hash like the other insets, and a change re-runs the view.

Touched: `src/window/titlebar.rs` (the types), `src/window/titlebar/caption.rs` (skip drawing; width parameter),
`src/window/chrome.rs`, `src/window/create.rs`, `src/platform/win32/frame.rs` (the frame keeps the cell width and
the "drawn by Rune" flag), `src/runtime/input/titlebar.rs`, `src/runtime/commit/measure.rs` + `src/memo.rs` (hash),
`src/lib.rs` (exports). Lines are marked `PATCH` where they add something new.

### Patch 2: frames keep coming during the Windows move / size loop

Dragging the header of a maximized window restores it inside the system's modal move loop (`WM_ENTERSIZEMOVE` ..
`WM_EXITSIZEMOVE`), which runs inside the window procedure and never returns to winit's event loop. Rune draws the
frame a resize produces right away (`in_size_move`), but everything it normally asks for "at the end of the turn" (the
next frame of a running animation, timers) waits for the loop to end. The restore changes the view (the maximize
button's state), so layout moves go through Rune's layout springs; with only the first frame drawn, the content stayed
laid out at the maximized size, cropped, until the mouse was released. Edge resizes are affected the same way for
anything animating.

`src/platform/win32/dpi_drag.rs`: a 16 ms `SetTimer` runs from `WM_ENTERSIZEMOVE` to `WM_EXITSIZEMOVE`; each
`WM_TIMER` asks for a paint (`RedrawWindow(RDW_INTERNALPAINT)`), which winit turns into the normal redraw.

### Patch 3: no Windows "steady run"

Upstream keeps a focused Windows window presenting every refresh for 1 s after each change (`STEADY` in
`src/window/redraw.rs`, against refresh-rate swings with G-SYNC / FreeSync). A frame the run alone asked for should end
it when it draws nothing new, but without a compositor that check (`Presented::same`) failed every time, so the run
renewed itself on every frame: the idle window presented ~490 times a second (measured: 4932 frames in 10 s, every one
`idle=true` with `steady=Hold`), took most of the GPU and made speech generation 20-40x slower. Even when it ends, it
would keep the GPU busy for 1 s after every progress event of a running job. `STEADY` is zero, so a window rests as soon
as its frames stop changing.

### Patch 4: no white flash when the window closes (Windows)

Hiding a window that presents through a flip-model swap chain lets DWM play its hide animation on a stale or white
copy of the window, so closing the app flashed white with artefacts. `Cloak::hide_quietly`
(`src/platform/win32/cloak.rs`) cloaks the window, hides it and uncloaks it, so nothing of it is on screen while it
hides; `Win::hide` (`src/window/win.rs`) uses it on Windows. A close hides the window that way before Rune releases the GPU and destroys the window (`Windows::hide`, which used to destroy it while still visible), and a quit hides it before the event loop ends
(`src/window/lifecycle.rs`). Lines are marked `PATCH (local)`.
Every window is also hidden that way just before it is dropped at exit (`AnyHost::hide_for_exit`, called from
`exiting` in `src/window.rs`). The app itself never blocks the UI thread on quit: it only asks the engine to exit and
waits for it after the loop has ended (`ui::train::quit_now`, `main`).

### Patch 5: a pointer move does not draw a frame that changes nothing

Upstream asks for a redraw on every `CursorMoved` and draws and presents it, even when the frame is identical to the
one on screen (the pointer moved inside one element). On a 165 Hz display that was 165 whole-window frames a second
for as long as the pointer moved over the window (measured: GPU 12 W idle, 36 W while moving the pointer). The patch
marks the frame a pointer move asks for as one that may reuse what is on screen (`TimerFrame::pointer_moved`), the
path upstream already has for timer wakes: the frame is built, compared with the presented one byte for byte
(`Presented::unchanged`), and drawn only when it differs (a hover change, an animation, pending input).

Touched: `src/window/pacing.rs`, `src/window/events.rs`. Lines are marked `PATCH`.

### Patch 6: motion on a fast display draws about 60 frames a second (Windows, Linux)

Upstream asks for the next frame as soon as one presents while anything animates, so motion runs at the display's
refresh rate: on a 165 Hz display a tab switch (a 0.72 s fade) took about 118 whole-window frames (0.72 s x 165), a short burst
of GPU load on every click that the owner heard as coil whine from the graphics card. With the patch, a frame that
only moves things waits for the next step of `motion_interval` (`src/window/pacing.rs`): a whole number of
refreshes about 1/60 s long (55 Hz at 165 Hz, 72 Hz at 144 Hz, 60 Hz at 120 or 240 Hz; displays below 105 Hz keep
vsync pacing; an unknown rate takes 60 Hz). The window sleeps until then (`Win::motion_at`, a `wake_deadline`);
input, state changes, live values and window moves still draw at once, and a pointer move rides on the frame already
on its way unless a press drags. The same tab switch is about 40 frames at 165 Hz.

Touched: `src/window/pacing.rs`, `src/window/draw.rs`, `src/window/any_host.rs`, `src/window/events.rs`,
`src/window/win.rs`, `src/window/create.rs`. Lines are marked `PATCH`.

### Patch 7: no system frame on Linux for an app that draws its caption buttons

On GNOME's Wayland session there are no server-side decorations, so winit drew its own Adwaita title bar above the
app's header: two title bars, one of them out of place. A window whose `Titlebar` has `custom_caption` now opens
without decorations on Linux (`chrome::frameless`), as on Windows: `ChromeInsets` report the titlebar's height and
the caption cells' width, and the app's buttons are ordinary elements under the pointer (`CaptionView` carries only
`maximized`). A press within 6 pt of an edge (16 pt along a corner) resizes the window through
`Window::drag_resize_window`, with the matching cursor, unless it is maximized or full screen (`chrome::resize_edge`).

Touched: `src/window/chrome.rs`, `src/window/create.rs`, `src/window/events.rs`, `src/window/native.rs`,
`src/input/cursor/os.rs` (`OsCursor::forget`). Lines are marked `PATCH`.

### Patch 8: an element without a transition of its own leaves at once

Upstream gives every element a 0.2 s fade in and a 0.15 s fade out. When the app swaps one element for another in the
same place (Turn off / Remove for a delete confirmation, one version's notes for another's), the old one kept fading
out over the new one, and a card that lost children shrank while they were still drawn, so they stuck out of it. The
default now fades in as before and removes at once (`default_transition`: the exit spring is `Spring::NONE`).
Elements that set `El::transition` (menus, popovers, toasts) keep theirs.

Touched: `src/runtime/commit/apply.rs`. Lines are marked `PATCH`.

### Patch 9: file drops on Wayland

winit 0.30 reports file drags only on X11, Windows and macOS: on a Wayland session nothing could be dropped on the
window. `platform::wayland_drop` binds `wl_data_device` on winit's own display connection, in a private queue that
a thread of its own dispatches (the window's loop read the queue too late: Mutter had cancelled the drop because the
offer's copy action arrived after the pointer left), accepts an offer with `text/uri-list` over the window's surface, reads the list through a
pipe on a thread from the moment the drag enters (or at the drop, waiting at most 2 s), and hands `Host::flush_files`
the same enter, move, drop and leave steps, in window points, as Windows' OLE target (`platform::win32::drop_target`).
Offers are never destroyed and the queue and bindings are forgotten when the thread ends: an offer destroyed on leave
could still get events queued for it, and at exit the bindings outlived the display winit had closed; both crashed in
libwayland.
`MIMVOX_DEBUG_DROP` prints each step to stderr.

Touched: `src/platform/wayland_drop.rs` (new), `src/platform.rs`, `src/window/events.rs`. Lines are marked `PATCH`.

### Patch 10: one running copy on Linux and macOS

`single_instance` claimed the app only on Windows, so every launch on Linux, and `open -n` or the bare binary on
macOS, opened another window with its own engine. `open_url::instance` now has a Unix side as well: the first process
listens on a local socket per user (an abstract name on Linux, a file in the temp folder on macOS, removed and claimed
again when nobody answers it); a later launch sends its `XDG_ACTIVATION_TOKEN` or `DESKTOP_STARTUP_ID` and exits. The
first process reopens as on Windows and passes the token on, so Wayland lets it take focus; a window that already shows
is brought forward too.

Touched: `src/open_url/instance.rs`, `src/window.rs`, `src/window/app_events.rs`, `src/window/any_host.rs`. Lines are
marked `PATCH`.
