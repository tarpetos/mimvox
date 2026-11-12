# Vendored crates

## rune-patched

A copy of `crates/rune` from `github.com/anomalyco/rune` at the pinned rev
`a7cea1692a8ce68b1b63c46688c493bb94a8456d` (only `src/`, `assets/` and the two `.dxbc` shaders), wired in with
`[patch."https://github.com/anomalyco/rune"]` in `app/Cargo.toml`. Its `Cargo.toml` is the upstream one with the
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
