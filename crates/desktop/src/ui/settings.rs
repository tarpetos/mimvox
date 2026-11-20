use super::api::{self, Status};
use super::components::{Button, Tone, icon_button, icon_el, icons, segmented, status_dot, switch};
use super::shell::{self, EngineStatus};
use super::theme::{Palette, radius, ty};
use super::voices::kit;
use crate::engine::protocol::RedubMode;
use crate::state::{AppState, Screen};
use mimvox_api::token::TokenSource;
use rune::{Color, El, ScrollSpec, Span, Tx, col, el, row, text};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Modules,
    Data,
    Api,
}

const ALL: [Section; 3] = [Section::Modules, Section::Data, Section::Api];

/// settings.json key of the section left open.
const SAVED_KEY: &str = "settings_section";

impl Section {
    fn key(self) -> &'static str {
        match self {
            Section::Modules => "settings-modules",
            Section::Data => "settings-data",
            Section::Api => "settings-api",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Section::Modules => "modules",
            Section::Data => "data",
            Section::Api => "api",
        }
    }
}

/// `None` is all folded; never saved (or unknown) is API.
fn from_saved(v: Option<&Value>) -> Option<Section> {
    match v.and_then(Value::as_str) {
        Some("none") => None,
        Some(name) => ALL.into_iter().find(|s| s.name() == name).or(Some(Section::Api)),
        None => Some(Section::Api),
    }
}

fn to_saved(fold: Option<Section>) -> Value {
    json!(fold.map_or("none", Section::name))
}

pub struct SettingsState {
    pub open: bool,
    pub fold: Option<Section>,
    /// The section the person left open; Settings opens on it.
    pub last: Option<Section>,
    last_read: bool,
    last_dirty: bool,
}

impl Default for SettingsState {
    fn default() -> Self {
        SettingsState { open: false, fold: None, last: Some(Section::Api), last_read: false, last_dirty: false }
    }
}

fn screen_module(s: &AppState) -> Option<&'static str> {
    match s.screen {
        Screen::Train => Some("training"),
        Screen::Redub => match s.redub.mode {
            RedubMode::Dub => Some("translation"),
            RedubMode::Convert => Some("training"),
            _ => None,
        },
        _ => None,
    }
}

fn needs_attention(s: &AppState) -> bool {
    shell::models_missing(s)
        || screen_module(s).is_some_and(|id| super::modules::status(s, id) == super::modules::Status::Off)
}

/// On Modules when something needs attention (without forgetting the remembered section), else on the remembered one.
pub fn open(s: &mut AppState) {
    let which = if needs_attention(s) { Some(Section::Modules) } else { s.settings.last };
    show(s, which);
}

pub fn open_modules(s: &mut AppState, tx: &mut Tx) {
    show(s, Some(Section::Modules));
    tx.scroll_key_to(Section::Modules.key(), 0.0);
}

fn show(s: &mut AppState, which: Option<Section>) {
    s.settings.open = true;
    s.settings.fold = which;
    super::modules::reset(s);
    super::relocate::measure_free(s);
}

fn toggle(s: &mut AppState, which: Section) {
    let fold = if s.settings.fold == Some(which) { None } else { Some(which) };
    s.settings.fold = fold;
    s.settings.last = fold;
    s.settings.last_dirty = true;
}

pub fn settle(s: &mut AppState) -> bool {
    if s.shell.scripted() {
        return false;
    }
    if !s.settings.last_read {
        s.settings.last_read = true;
        s.settings.last = from_saved(super::onboarding::setting(SAVED_KEY).as_ref());
    }
    if s.settings.last_dirty {
        s.settings.last_dirty = false;
        super::onboarding::save_setting(SAVED_KEY, to_saved(s.settings.last));
    }
    false
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
        .child(engine_line(s, p))
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
        .on_press(move |s: &mut AppState| toggle(s, which))
        .child(text(title.to_owned()).font(ty::LABEL).color(p.text).a11y_hidden())
        .child(
            text(summary.to_owned()).font(ty::CAPTION).color(p.text_dim).grow(1.0).min_w(0.0).truncate().a11y_hidden(),
        )
        .child(icon_el(&icons::CHEVRON_DOWN, 12.0, p.text_dim).rotate(if open { std::f32::consts::PI } else { 0.0 }))
}

fn engine_line(s: &AppState, p: Palette) -> El {
    let look = shell::engine_look(s);
    let (summary, facts) = shell::engine_details(s);
    let info = col()
        .grow(1.0)
        .min_w(0.0)
        .gap(2.0)
        .child(text(look.label).font(ty::LABEL).color(p.text))
        .child(text(summary).font(ty::CAPTION).color(p.text_dim).wrap())
        .children((!facts.is_empty()).then(|| text(facts.join(" · ")).font(ty::CAPTION).color(p.text_faint).wrap()));
    row()
        .key("settings-engine")
        .role("status")
        .label(format!("Engine: {}", look.label))
        .w_full()
        .gap(12.0)
        .padding(12.0)
        .radius(radius::CONTROL)
        .fill(p.well)
        .border(1.0, p.well_rim)
        .child(el().pt(5.0).child(status_dot(p, look.tone, look.pulse)))
        .child(info)
        .children(look.retry.then(|| {
            Button::new("Retry")
                .small()
                .icon(&icons::REFRESH)
                .key("settings-engine-retry")
                .on_press(shell::restart_engine)
                .build(p)
        }))
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
    c = c.child(docs_row(s, p));
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

fn docs_row(s: &AppState, p: Palette) -> El {
    let prefs = s.api.prefs;
    let listening = match &s.api.status {
        Status::Listening(addr) if prefs.on => Some(*addr),
        _ => None,
    };
    let url = api::docs_url(s, listening.unwrap_or(prefs.addr()));
    let detail = match listening {
        Some(_) => "Every endpoint, with try-it, in your browser.",
        None if !prefs.on => "Turn on the API to open the docs.",
        None => "Opens once the API is listening.",
    };
    let ink = if listening.is_some() { p.accent } else { p.text_faint };
    let link = row()
        .key("api-docs-link")
        .gap(6.0)
        .center_y()
        .child(
            text(url.clone())
                .font(ty::LABEL)
                .color(ink)
                .spans(vec![Span::new(0, url.len()).underline(ink.alpha(0.7))])
                .a11y_hidden(),
        )
        .child(icon_el(&icons::OPEN, 14.0, ink));
    let link = match listening {
        Some(_) => link
            .href(url.clone())
            .label(format!("Open API docs, {url}"))
            .help("Open in your browser")
            .focusable(true)
            .cursor(rune::Cursor::Pointer),
        None => link.role("link").label(format!("API docs, {url}")).disabled(true),
    };
    setting_row(p, "API docs", detail, link)
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

fn demo_state(s: &mut AppState) {
    super::speak::scene_idle(s);
    super::relocate::demo(s);
    super::modules::demo(s);
    s.api.prefs_read = true;
    s.api.token = Some((
        "q3Vt0bWm9xR2kLs7YdFh4NcJ8uPzA1eGiOwT6yHrBvE".into(),
        TokenSource::File(s.shell.data_dir.join("api").join("token")),
    ));
}

fn demo(s: &mut AppState) {
    demo_state(s);
    open(s);
}

fn with_engine(s: &mut AppState, engine: EngineStatus) {
    demo_state(s);
    s.shell.set_engine(engine);
    open(s);
}

fn scene_engine_ready(s: &mut AppState) {
    demo(s);
}

fn scene_engine_starting(s: &mut AppState) {
    with_engine(s, EngineStatus::Starting { since: 0.0 });
}

fn scene_engine_stopped(s: &mut AppState) {
    with_engine(
        s,
        EngineStatus::Stopped { code: Some(1), detail: "CUDA out of memory while loading the speech model.".into() },
    );
}

fn scene_engine_no_memory(s: &mut AppState) {
    super::speak::scene_engine_no_memory(s);
    let engine = s.shell.engine.clone();
    let why = s.shell.tts_message.clone();
    with_engine(s, engine);
    s.shell.tts_message = why;
}

fn scene_engine_models_missing(s: &mut AppState) {
    with_engine(s, super::speak::models_missing());
}

fn scene_remembered_data(s: &mut AppState) {
    demo_state(s);
    s.settings.last = Some(Section::Data);
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
        ("settings_engine_ready", scene_engine_ready),
        ("settings_engine_starting", scene_engine_starting),
        ("settings_engine_stopped", scene_engine_stopped),
        ("settings_engine_no_memory", scene_engine_no_memory),
        ("settings_engine_models_missing", scene_engine_models_missing),
        ("settings_remembered_data", scene_remembered_data),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_reopen_on_the_section_left_open_unless_something_needs_attention() {
        let mut s = AppState::default();
        super::super::speak::scene_idle(&mut s);
        open(&mut s);
        assert_eq!(s.settings.fold, Some(Section::Api), "never chosen: API");
        toggle(&mut s, Section::Data);
        close(&mut s);
        open(&mut s);
        assert_eq!(s.settings.fold, Some(Section::Data));
        toggle(&mut s, Section::Data);
        close(&mut s);
        open(&mut s);
        assert_eq!(s.settings.fold, None, "all folded stays folded");
        toggle(&mut s, Section::Data);
        close(&mut s);

        s.shell.set_engine(super::super::speak::models_missing());
        open(&mut s);
        assert_eq!(s.settings.fold, Some(Section::Modules));
        assert_eq!(s.settings.last, Some(Section::Data), "attention does not overwrite the choice");
        close(&mut s);
        super::super::speak::scene_idle(&mut s);
        s.screen = Screen::Train;
        s.shell.components_off = vec!["training".into()];
        open(&mut s);
        assert_eq!(s.settings.fold, Some(Section::Modules), "Train needs voice training, which is off");
        close(&mut s);
        s.screen = Screen::Speak;
        open(&mut s);
        assert_eq!(s.settings.fold, Some(Section::Data));
    }

    #[test]
    fn the_remembered_section_survives_a_restart_through_settings_json() {
        for fold in [Some(Section::Modules), Some(Section::Data), Some(Section::Api), None] {
            let saved = json!({ SAVED_KEY: to_saved(fold) });
            let text = serde_json::to_string(&saved).unwrap();
            let read: Value = serde_json::from_str(&text).unwrap();
            assert_eq!(from_saved(read.get(SAVED_KEY)), fold);
        }
        assert_eq!(from_saved(None), Some(Section::Api));
        assert_eq!(from_saved(Some(&json!("gone"))), Some(Section::Api));
        assert_eq!(from_saved(Some(&json!(3))), Some(Section::Api));
    }
}
