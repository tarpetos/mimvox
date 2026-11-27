//! What the whole app hears, not one window: a quit (`crate::quit`) and a reopen.

use super::*;

impl Windows {
    /// The app starts to quit (`crate::quit`): every window still open runs `App::on_quit`, once
    /// per quit. A session ending now (`ending`) then closes the windows as their close buttons
    /// would, before the loop ends, as a macOS logout does: the OS may end the process next.
    pub(super) fn quit(&mut self, event_loop: &ActiveEventLoop, ending: bool) {
        use crate::quit::{After, Effect, Event};
        for effect in crate::quit::send(Event::Start { ending, after: After::Exit }) {
            match effect {
                Effect::RunHooks => {
                    for host in self.hosts.iter_mut().flatten() { host.quit_hook(); }
                    crate::marks::mark("quit.hooks");
                    crate::quit::send(Event::HooksRan);
                }
                Effect::CloseAndExit => {
                    if !event_loop.exiting() {
                        for host in self.hosts.iter_mut().flatten() { host.terminate(); }
                        event_loop.exit();
                    }
                    crate::quit::send(Event::WindowsClosed);
                }
                // Only the loop's end relaunches.
                Effect::Relaunch => {}
            }
        }
    }

    /// The app was reopened: every `on_reopen` hook runs, then, unless one handled it or the
    /// app is an accessory, the default brings back a window when none shows (`reopen::target`).
    pub(super) fn reopen(&mut self, event_loop: &ActiveEventLoop) {
        let candidates = |hosts: &[Option<Box<dyn AnyHost>>]| -> Vec<_> {
            hosts.iter().map(|h| h.as_ref().map(|h| h.reopen_candidate()).unwrap_or_default()).collect()
        };
        let has_visible = crate::platform::reopen::has_visible(&candidates(&self.hosts));
        let mut handled = false;
        for host in self.hosts.iter_mut().flatten() {
            handled |= host.reopen_hook(event_loop, has_visible);
        }
        if handled || event_loop.exiting() { return; }
        #[cfg(target_os = "macos")]
        if crate::platform::reopen::accessory() { return; }
        // The default sees what the hooks did, to other windows too.
        for host in self.hosts.iter_mut().flatten() { host.apply_pending(event_loop); }
        if event_loop.exiting() { return; }
        #[cfg(target_os = "macos")]
        crate::platform::reopen::unhide();
        #[cfg(all(unix, not(target_os = "ios"), not(target_os = "android")))] // PATCH
        let token = crate::open_url::instance::take_token(); // PATCH
        #[cfg(not(all(unix, not(target_os = "ios"), not(target_os = "android"))))] // PATCH
        let token = None; // PATCH
        let target = crate::platform::reopen::target(&candidates(&self.hosts)); // PATCH
        #[cfg(all(unix, not(target_os = "ios"), not(target_os = "android")))] // PATCH
        let target = target.or_else(|| candidates(&self.hosts).iter().position(|c| c.main && c.reopenable && c.visible)); // PATCH
        if let Some(Some(host)) = target.and_then(|i| self.hosts.get_mut(i)) { // PATCH
            host.reopen(event_loop, token); // PATCH
        }
        // Windows: a launch while the app runs brings its front window forward, restored, as
        // AppKit's activation does on macOS; the launch let this process take the foreground.
        #[cfg(target_os = "windows")]
        crate::open_url::instance::raise_front();
    }
}
