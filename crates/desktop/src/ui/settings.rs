use super::api::{self, Status};
use super::components::{Button, Tone, icon_button, icon_el, icons, segmented, switch};
use super::theme::{Palette, radius, ty};
use super::voices::kit;
use crate::state::AppState;
use mimvox_api::token::TokenSource;
use rune::{Color, El, ScrollSpec, Tx, col, el, row, text};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Modules,
    Data,
    Api,
}

impl Section {
    fn key(self) -> &'static str {
        match self {
            Section::Modules => "settings-modules",
            Section::Data => "settings-data",
            Section::Api => "settings-api",
        }
    }
}

#[derive(Default)]
pub struct SettingsState {
    pub open: bool,
    pub fold: Option<Section>,
}

pub fn open(s: &mut AppState) {
    open_at(s, Section::Api);
}

pub fn open_modules(s: &mut AppState, tx: &mut Tx) {
    open_at(s, Section::Modules);
    tx.scroll_key_to(Section::Modules.key(), 0.0);
}

fn open_at(s: &mut AppState, which: Section) {
    s.settings.open = true;
    s.settings.fold = Some(which);
    super::modules::reset(s);
    super::relocate::measure_free(s);
}

fn close(s: &mut AppState) {
    s.settings.open = false;
    super::modules::reset(s);
    s.api.confirm_new_token = false;
    s.api.token_shown = false;
}

pub fn overlay(s: &AppState, p: &Palette) -> Option<El> {
    if !s.settings.open {
        return None;
    }
    let p = *p;
    let head =
        row()
            .w_full()
            .gap(12.0)
            .center_y()
            .child(
                col().grow(1.0).min_w(0.0).gap(2.0).child(text("Settings").font(ty::TITLE).color(p.text)).child(
                    text("Saved on this computer and applied at once.").font(ty::CAPTION).color(p.text_dim).wrap(),
                ),
            )
            .child(icon_button(p, "settings-close", &icons::CLOSE, "Close", close));
    let fold = |which: Section, title: &str, summary: String, content: fn(&AppState, Palette) -> El| {
        let open = s.settings.fold == Some(which);
        col()
            .key(which.key())
            .w_full()
            .gap(14.0)
            .padding(12.0)
            .radius(radius::CONTROL)
            .fill(p.well)
            .border(1.0, p.well_rim)
            .child(fold_row(p, title, &summary, open, which))
            .children(open.then(|| content(s, p)))
    };
    let body = col()
        .key("settings-list")
        .w_full()
        .shrink()
        .min_h(0.0)
        .gap(12.0)
        .scroll(ScrollSpec { stick_bottom: false })
        .scroll_edge_fade(12.0)
        .child(fold(Section::Modules, "Modules", super::modules::summary(s), super::modules::section))
        .child(fold(Section::Data, "Data folder", super::relocate::summary(s), |s, p| super::relocate::section(s, &p)))
        .child(fold(Section::Api, "API", api_summary(s), api_section));
    let card = col()
        .key("settings-dialog")
        .w(720.0)
        .shrink()
        .min_h(0.0)
        .padding(22.0)
        .gap(12.0)
        .radius(radius::PANEL)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 1.0))
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(18.0, 50.0, p.shadow)
        .on_press(|_: &mut AppState| {})
        .child(head)
        .child(body);
    let scrim = el().key("settings-scrim").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).on_press(close);
    let mut layer = col()
        .key("settings-layer")
        .pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0))
        .padding(24.0)
        .center()
        .fill(Color::BLACK.alpha(0.55))
        .child(scrim)
        .child(card);
    if s.api.notice_open {
        layer = layer.child(notice(p));
    }
    Some(layer)
}

fn fold_row(p: Palette, title: &str, summary: &str, open: bool, which: Section) -> El {
    row()
        .key(format!("{}-fold", which.key()))
        .role("button")
        .label(title.to_owned())
        .w_full()
        .gap(10.0)
        .center_y()
        .px(6.0)
        .py(6.0)
        .radius(radius::CONTROL)
        .hover(p.glass_hover)
        .focusable(true)
        .on_press(move |s: &mut AppState| {
            s.settings.fold = if s.settings.fold == Some(which) { None } else { Some(which) }
        })
        .child(text(title.to_owned()).font(ty::LABEL).color(p.text).a11y_hidden())
        .child(
            text(summary.to_owned()).font(ty::CAPTION).color(p.text_dim).grow(1.0).min_w(0.0).truncate().a11y_hidden(),
        )
        .child(icon_el(&icons::CHEVRON_DOWN, 12.0, p.text_dim).rotate(if open { std::f32::consts::PI } else { 0.0 }))
}

fn api_summary(s: &AppState) -> String {
    if !s.api.prefs.on {
        return "Off".to_owned();
    }
    match &s.api.status {
        _ if api::blocked(s).is_some() => "On, cannot start".to_owned(),
        Status::Listening(addr) => format!("On · {}", api::url(s, *addr)),
        Status::Failed(_) => "On, not listening".to_owned(),
        Status::Starting | Status::Off => "On, starting…".to_owned(),
    }
}

fn setting_row(p: Palette, title: &str, detail: &str, control: El) -> El {
    row()
        .w_full()
        .gap(16.0)
        .center_y()
        .child(
            col().grow(1.0).min_w(0.0).gap(2.0).child(text(title.to_owned()).font(ty::LABEL).color(p.text)).children(
                (!detail.is_empty()).then(|| text(detail.to_owned()).font(ty::CAPTION).color(p.text_dim).wrap()),
            ),
        )
        .child(control)
}

fn status_line(s: &AppState, p: Palette) -> Option<El> {
    let prefs = s.api.prefs;
    if !prefs.on {
        return None;
    }
    let (tone, msg) = if let Some(why) = api::blocked(s) {
        (Tone::Warning, why.to_owned())
    } else if let Some(e) = s.api.token_error.as_ref().filter(|_| prefs.needs_token()) {
        (Tone::Danger, e.clone())
    } else {
        match &s.api.status {
            Status::Listening(addr) => (Tone::Success, format!("Listening on {}", api::url(s, *addr))),
            Status::Failed(e) => (Tone::Danger, e.clone()),
            Status::Starting | Status::Off => (Tone::Neutral, "Starting…".to_owned()),
        }
    };
    Some(kit::notice(&p, tone, &msg).key("api-status"))
}

fn api_section(s: &AppState, p: Palette) -> El {
    let prefs = s.api.prefs;
    let mut c = col().key("api-section").w_full().gap(16.0).px(6.0).pb(6.0);
    c = c.child(setting_row(
        p,
        "Turn on API",
        "Lets scripts and other programs use Mimvox, as long as the app is open.",
        switch(p, "api-on", "Turn on API", prefs.on, true, move |s: &mut AppState| api::set_on(s, !prefs.on)),
    ));
    c = c.children(status_line(s, p));

    let address = segmented(
        p,
        "api-address",
        &["This computer only (127.0.0.1)", "Local network"],
        usize::from(prefs.lan),
        |s: &mut AppState, i| api::set_lan(s, i == 1),
    );
    c =
        c.child(
            col().w_full().gap(8.0).child(text("Address").font(ty::LABEL).color(p.text)).child(
                row().w_full().gap(10.0).center_y().child(address).child(el().grow(1.0)).child(port_field(s, p)),
            ),
        );
    if let Some(e) = &s.api.port_error {
        c = c.child(text(e.clone()).key("api-port-error").font(ty::CAPTION).color(p.bad));
    }
    c = c.child(token_block(s, p));

    let (require_detail, require_on) = if prefs.lan {
        ("Always on for the local network.", true)
    } else {
        if prefs.require_token {
            ("On: every program needs it.", true)
        } else {
            ("Off: programs on this computer get in without it. Web pages never do.", false)
        }
    };
    c = c.child(setting_row(
        p,
        "Require token",
        require_detail,
        switch(p, "api-require-token", "Require token", require_on, !prefs.lan, move |s: &mut AppState| {
            api::set_require_token(s, !require_on)
        }),
    ));
    c.child(
        text(format!(
            "OpenAI-compatible: base URL {}/v1, the token as the API key (Authorization: Bearer <token>).",
            hint_url(s)
        ))
        .key("api-hint")
        .font(ty::CAPTION)
        .color(p.text_faint)
        .wrap(),
    )
}

fn hint_url(s: &AppState) -> String {
    match &s.api.status {
        Status::Listening(addr) => api::url(s, *addr),
        _ => api::url(s, s.api.prefs.addr()),
    }
}

fn port_field(s: &AppState, p: Palette) -> El {
    let edit = &s.api.port;
    let invalid = s.api.port_error.is_some();
    let well = row()
        .key("api-port-well")
        .w(96.0)
        .h(34.0)
        .px(12.0)
        .center_y()
        .radius(radius::CONTROL)
        .fill(p.well)
        .border(
            1.0,
            if invalid {
                p.bad.alpha(0.8)
            } else if edit.focused {
                p.accent.alpha(0.8)
            } else {
                p.well_rim
            },
        )
        .child(
            rune::text_input(edit, ty::BODY)
                .key("api-port")
                .label("Port")
                .single_line()
                .w_full()
                .color(p.text)
                .placeholder(api::DEFAULT_PORT.to_string(), p.text_faint)
                .caret(p.accent, p.accent.alpha(0.35))
                .on_edit(|s: &mut AppState| &mut s.api.port)
                .on_submit(|s: &mut AppState, _| api::apply_port(s)),
        );
    let mut r = row().gap(8.0).center_y().child(text("Port").font(ty::LABEL).color(p.text_dim)).child(well);
    if api::port_dirty(s) {
        r = r.child(Button::new("Apply").small().key("api-port-apply").on_press(api::apply_port).build(p));
    }
    r
}

fn token_block(s: &AppState, p: Palette) -> El {
    let from_env = api::token_from_env(s);
    let token = s.api.token.as_ref().map(|(t, _)| t.as_str());
    let shown = match token {
        Some(t) if s.api.token_shown => t.to_owned(),
        Some(t) => "•".repeat(t.chars().count().min(32)),
        None => s.api.token_error.clone().unwrap_or_else(|| "No token yet".to_owned()),
    };
    let value = row()
        .key("api-token-value")
        .grow(1.0)
        .min_w(0.0)
        .h(34.0)
        .px(12.0)
        .center_y()
        .radius(radius::CONTROL)
        .fill(p.well)
        .border(1.0, p.well_rim)
        .child(text(shown).font(ty::NUM).color(if token.is_some() { p.text } else { p.text_faint }).truncate());
    let has = token.is_some();
    let buttons = row()
        .gap(6.0)
        .center_y()
        .child(
            Button::new(if s.api.token_shown { "Hide" } else { "Show" })
                .small()
                .key("api-token-show")
                .disabled(!has)
                .on_press(|s| s.api.token_shown = !s.api.token_shown)
                .build(p),
        )
        .child(Button::new("Copy").small().key("api-token-copy").disabled(!has).on_press_tx(api::copy_token).build(p))
        .child(
            Button::new("New token")
                .small()
                .icon(&icons::REFRESH)
                .key("api-token-new")
                .disabled(from_env)
                .on_press(|s| s.api.confirm_new_token = true)
                .build(p),
        );
    let source = match &s.api.token {
        Some((_, TokenSource::Env)) => {
            format!("From {}: New token is off while it is set.", mimvox_api::token::TOKEN_ENV)
        }
        Some((_, TokenSource::File(path))) => format!("Saved in {}", path.display()),
        None => String::new(),
    };
    let mut c = col()
        .key("api-token")
        .w_full()
        .gap(8.0)
        .child(text("Token").font(ty::LABEL).color(p.text))
        .child(row().w_full().gap(10.0).center_y().child(value).child(buttons));
    if s.api.confirm_new_token {
        c = c.child(
            row()
                .key("api-token-confirm")
                .w_full()
                .gap(8.0)
                .center_y()
                .padding(10.0)
                .radius(radius::CONTROL)
                .fill(p.warn.alpha(0.1))
                .border(1.0, p.warn.alpha(0.35))
                .child(
                    text("Make a new token? Programs that use the current one stop working.")
                        .font(ty::BODY)
                        .color(p.text)
                        .wrap()
                        .grow(1.0)
                        .min_w(0.0),
                )
                .child(
                    Button::new("Keep")
                        .small()
                        .key("api-token-keep")
                        .on_press(|s| s.api.confirm_new_token = false)
                        .build(p),
                )
                .child(
                    Button::new("New token")
                        .small()
                        .danger()
                        .key("api-token-new-yes")
                        .on_press(api::new_token)
                        .build(p),
                ),
        );
    }
    if !source.is_empty() {
        c = c.child(text(source).key("api-token-source").font(ty::CAPTION).color(p.text_faint).wrap());
    }
    c
}

fn notice(p: Palette) -> El {
    let point = |t: &str| {
        row()
            .w_full()
            .gap(10.0)
            .items_start()
            .child(el().size(5.0, 5.0).radius(2.5).fill(p.text_dim).mt(7.0))
            .child(text(t.to_owned()).font(ty::BODY).color(p.text).wrap().grow(1.0).min_w(0.0))
    };
    let card = col()
        .key("api-notice")
        .role("dialog")
        .label("Before you turn on the API")
        .w(520.0)
        .padding(22.0)
        .gap(12.0)
        .radius(radius::PANEL)
        .fill(rune::oklcha(0.21, 0.03, 255.0, 1.0))
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(18.0, 50.0, p.shadow)
        .on_press(|_: &mut AppState| {})
        .child(text("Before you turn on the API").font(ty::TITLE).color(p.text))
        .child(point("Clone only your own voice, or one whose owner gave you permission."))
        .child(point("The speech model's weights are licensed CC BY-NC 4.0: no commercial use."))
        .child(point("Anyone who can reach the address can use your voices and read and write files through the API."))
        .child(
            row()
                .w_full()
                .gap(8.0)
                .justify_end()
                .child(Button::new("Cancel").key("api-notice-cancel").on_press(api::decline_notice).build(p))
                .child(
                    Button::new("Turn on")
                        .primary()
                        .icon(&icons::POWER)
                        .key("api-notice-ok")
                        .on_press(api::accept_notice)
                        .build(p),
                ),
        );
    let scrim =
        el().key("api-notice-scrim").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).on_press(api::decline_notice);
    el().key("api-notice-layer")
        .pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0))
        .center()
        .fill(Color::BLACK.alpha(0.4))
        .child(scrim)
        .child(card)
}

fn demo(s: &mut AppState) {
    super::speak::scene_idle(s);
    super::relocate::demo(s);
    super::modules::demo(s);
    s.api.prefs_read = true;
    s.api.token = Some((
        "q3Vt0bWm9xR2kLs7YdFh4NcJ8uPzA1eGiOwT6yHrBvE".into(),
        TokenSource::File(s.shell.data_dir.join("api").join("token")),
    ));
    open(s);
}

fn listening(s: &mut AppState, lan: bool) {
    demo(s);
    s.api.prefs.on = true;
    s.api.prefs.notice_seen = true;
    s.api.prefs.lan = lan;
    s.api.lan_ip = Some(std::net::IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 1, 23)));
    s.api.status = Status::Listening(s.api.prefs.addr());
}

fn scene_off(s: &mut AppState) {
    demo(s);
}

fn scene_on(s: &mut AppState) {
    listening(s, false);
}

fn scene_lan(s: &mut AppState) {
    listening(s, true);
    s.api.token_shown = true;
}

fn scene_port_error(s: &mut AppState) {
    listening(s, false);
    s.api.status = Status::Failed(mimvox_api::host::bind_error(
        s.api.prefs.addr(),
        &std::io::Error::from(std::io::ErrorKind::AddrInUse),
    ));
    s.api.port = rune::TextEdit::new("99999");
    s.api.port_error = api::parse_port("99999").err();
}

fn scene_notice(s: &mut AppState) {
    demo(s);
    s.api.notice_open = true;
}

fn scene_new_token(s: &mut AppState) {
    listening(s, false);
    s.api.prefs.require_token = true;
    s.api.confirm_new_token = true;
}

fn scene_other_engine(s: &mut AppState) {
    listening(s, false);
    s.api.status = Status::Off;
    s.shell.set_engine(super::shell::EngineStatus::Stopped {
        code: None,
        detail: "another Mimvox engine (the app or mimvox-server) is already running on /data".into(),
    });
}

fn scene_folded(s: &mut AppState) {
    listening(s, false);
    s.settings.fold = None;
}

fn modules_demo(s: &mut AppState) {
    demo(s);
    s.settings.fold = Some(Section::Modules);
}

fn scene_modules(s: &mut AppState) {
    modules_demo(s);
}

fn scene_modules_confirm_remove(s: &mut AppState) {
    modules_demo(s);
    super::modules::demo_confirm_remove(s, "ukrainian");
}

fn scene_modules_installing(s: &mut AppState) {
    modules_demo(s);
    super::setuprun::scene_running(s);
    s.setup.adding = Some("ukrainian".into());
}

pub fn scene_data(s: &mut AppState) {
    demo(s);
    s.settings.fold = Some(Section::Data);
}

pub fn scenes() -> &'static [super::Scene] {
    &[
        ("settings_api_off", scene_off),
        ("settings_api_on", scene_on),
        ("settings_api_lan", scene_lan),
        ("settings_api_port_error", scene_port_error),
        ("settings_api_notice", scene_notice),
        ("settings_api_new_token", scene_new_token),
        ("settings_api_other_engine", scene_other_engine),
        ("settings_folded", scene_folded),
        ("settings_modules", scene_modules),
        ("settings_modules_confirm_remove", scene_modules_confirm_remove),
        ("settings_modules_installing", scene_modules_installing),
        ("settings_data", scene_data),
    ]
}
