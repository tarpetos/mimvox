//! Speech languages: names for the screen and the shared language picker (a compact chip that opens a searchable
//! glass list, "Common" first, then "All languages").
//!
//! The list comes from the engine (`hello` -> `languages`: every language the speech model speaks, with English and
//! native names); until the engine has answered (and in headless scenes) the built-in copy below is used. UI strings stay
//! English; a native name is shown only when the app font can draw it (Latin, Greek and Cyrillic scripts), otherwise the
//! English name.

use rune::{Align, Color, El, ScrollSpec, Side, TextEdit, col, el, row, text};

use crate::engine::protocol::SpeechLanguage;
use crate::state::AppState;
use crate::ui::components::{dropdown_chip, glass_edge, icon_el, icons};
use crate::ui::theme::{Palette, motion, radius, ty};
use crate::ui::voices::kit;

/// The engine's list (OmniVoice languages with ~100 h of training data or more), used until `hello` arrives:
/// (code, English name, native name, common).
const BUILTIN: &[(&str, &str, &str, bool)] = &[
    ("en", "English", "English", true), ("uk", "Ukrainian", "Українська", true), ("de", "German", "Deutsch", true),
    ("fr", "French", "Français", true), ("es", "Spanish", "Español", true), ("pl", "Polish", "Polski", true),
    ("ro", "Romanian", "Română", true), ("it", "Italian", "Italiano", true), ("pt", "Portuguese", "Português", true), ("cs", "Czech", "Čeština", true),
    ("ru", "Russian", "Русский", true), ("ar", "Arabic", "العربية", false), ("as", "Assamese", "অসমীয়া", false),
    ("ba", "Bashkir", "Башҡортса", false), ("eu", "Basque", "Euskara", false), ("be", "Belarusian", "Беларуская", false),
    ("bn", "Bengali", "বাংলা", false), ("bs", "Bosnian", "Bosanski", false), ("bg", "Bulgarian", "Български", false),
    ("yue", "Cantonese", "粵語", false), ("ca", "Catalan", "Català", false), ("zh", "Chinese", "中文", false),
    ("hr", "Croatian", "Hrvatski", false), ("da", "Danish", "Dansk", false), ("nl", "Dutch", "Nederlands", false),
    ("et", "Estonian", "Eesti", false), ("fi", "Finnish", "Suomi", false), ("gl", "Galician", "Galego", false),
    ("ka", "Georgian", "ქართული", false), ("el", "Greek", "Ελληνικά", false), ("hi", "Hindi", "हिन्दी", false),
    ("hu", "Hungarian", "Magyar", false), ("is", "Icelandic", "Íslenska", false), ("id", "Indonesian", "Bahasa Indonesia", false),
    ("ja", "Japanese", "日本語", false), ("kn", "Kannada", "ಕನ್ನಡ", false), ("kk", "Kazakh", "Қазақ тілі", false),
    ("ko", "Korean", "한국어", false), ("lv", "Latvian", "Latviešu", false), ("lt", "Lithuanian", "Lietuvių", false),
    ("ml", "Malayalam", "മലയാളം", false), ("mt", "Maltese", "Malti", false), ("mr", "Marathi", "मराठी", false),
    ("mn", "Mongolian", "Монгол", false), ("ne", "Nepali", "नेपाली", false), ("no", "Norwegian", "Norsk", false),
    ("fa", "Persian", "فارسی", false), ("pa", "Punjabi", "ਪੰਜਾਬੀ", false), ("sr", "Serbian", "Српски", false),
    ("sk", "Slovak", "Slovenčina", false), ("sl", "Slovenian", "Slovenščina", false), ("sw", "Swahili", "Kiswahili", false),
    ("sv", "Swedish", "Svenska", false), ("ta", "Tamil", "தமிழ்", false), ("te", "Telugu", "తెలుగు", false),
    ("th", "Thai", "ไทย", false), ("tr", "Turkish", "Türkçe", false), ("ur", "Urdu", "اردو", false),
    ("uz", "Uzbek", "Oʻzbek", false), ("vi", "Vietnamese", "Tiếng Việt", false), ("cy", "Welsh", "Cymraeg", false),
];

/// Names of languages Whisper may report that the speech model does not speak (shown in Redub badges).
const EXTRA_NAMES: &[(&str, &str)] = &[("he", "Hebrew"), ("ro", "Romanian"), ("ms", "Malay"), ("mk", "Macedonian"), ("hy", "Armenian"), ("az", "Azerbaijani"), ("sq", "Albanian"), ("af", "Afrikaans"), ("tl", "Tagalog")];

/// The base code of `code` ("uk-UA" -> "uk").
pub fn base(code: &str) -> String {
    code.trim().split(['-', '_']).next().unwrap_or("").to_ascii_lowercase()
}

fn builtin() -> Vec<SpeechLanguage> {
    BUILTIN.iter().map(|(c, n, nat, common)| SpeechLanguage { code: (*c).into(), name: (*n).into(), native: (*nat).into(), common: *common }).collect()
}

/// Every language the speech model speaks, common ones first (the engine's list, else the built-in one).
pub fn all(s: &AppState) -> Vec<SpeechLanguage> {
    if s.shell.languages.is_empty() { builtin() } else { s.shell.languages.clone() }
}

/// Whether the voice can speak `code`.
pub fn is_spoken(s: &AppState, code: &str) -> bool {
    let b = base(code);
    if s.shell.languages.is_empty() { BUILTIN.iter().any(|(c, ..)| *c == b) } else { s.shell.languages.iter().any(|l| l.code == b) }
}

/// English name of a code ("uk" -> "Ukrainian"); an unknown code upper-case, `""` for an empty one.
pub fn name(code: &str) -> String {
    let b = base(code);
    if b.is_empty() {
        return String::new();
    }
    if let Some((_, n, ..)) = BUILTIN.iter().find(|(c, ..)| *c == b) {
        return (*n).to_owned();
    }
    EXTRA_NAMES.iter().find(|(c, _)| *c == b).map_or_else(|| b.to_uppercase(), |(_, n)| (*n).to_owned())
}

/// Native name of a code when the app font can draw it, else the English name ("uk" -> "Українська", "ja" -> "Japanese").
pub fn label(code: &str) -> String {
    let b = base(code);
    match BUILTIN.iter().find(|(c, ..)| *c == b) {
        Some((_, _, nat, _)) if drawable(nat) => (*nat).to_owned(),
        Some((_, n, ..)) => (*n).to_owned(),
        None => name(code),
    }
}

/// Whether Noto Sans (the app font: Latin, Greek, Cyrillic, Vietnamese) has every character of `t`.
fn drawable(t: &str) -> bool {
    t.chars().all(|c| (c as u32) < 0x0530 || (0x1E00..=0x1EFF).contains(&(c as u32)))
}

fn display(l: &SpeechLanguage) -> String {
    if !l.native.is_empty() && drawable(&l.native) { l.native.clone() } else { l.name.clone() }
}

// -------------------------------------------------------------------------------------------------
// Picker
// -------------------------------------------------------------------------------------------------

/// What a picked language is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Speak: the output language (remembered per voice).
    Speak,
    /// Redub: the language of the speech (offers "Auto-detect").
    RedubFrom,
    /// Redub: the language to translate to and speak.
    RedubTo,
    /// Voices: add a reference clip for this language.
    Reference,
    /// Voices: the language of the reference made from media files.
    MediaRef,
}

#[derive(Default)]
pub struct PickerState {
    /// The open picker and the key of the chip it hangs from.
    pub open: Option<(Target, String)>,
    pub search: TextEdit,
}

fn open(s: &mut AppState, target: Target, anchor: &str) {
    if s.langpick.open.as_ref().is_some_and(|(t, _)| *t == target) {
        s.langpick.open = None;
        return;
    }
    let mut search = TextEdit::new("");
    search.focused = true;
    s.langpick = PickerState { open: Some((target, anchor.to_owned())), search };
}

fn close(s: &mut AppState) {
    s.langpick.open = None;
}

fn pick(s: &mut AppState, target: Target, code: &str) {
    close(s);
    match target {
        Target::Speak => super::speak::set_language(s, code),
        Target::RedubFrom => super::redub::set_from(s, code),
        Target::RedubTo => super::redub::set_to(s, code),
        Target::Reference => super::refclip::start_language(s, code),
        Target::MediaRef => super::refmedia::set_language(s, code),
    }
}

/// The code currently chosen for `target` (shown with a check mark).
fn current(s: &AppState, target: Target) -> String {
    match target {
        Target::Speak => super::speak::language(s),
        Target::RedubFrom => s.redub.from.clone(),
        Target::RedubTo => s.redub.to.clone(),
        Target::Reference => String::new(),
        Target::MediaRef => super::refmedia::language(s),
    }
}

/// The languages `target` offers.
fn options(s: &AppState, target: Target) -> Vec<SpeechLanguage> {
    let mut list = all(s);
    match target {
        Target::RedubFrom => list.insert(0, SpeechLanguage { code: "auto".into(), name: "Auto-detect".into(), native: "Auto-detect".into(), common: true }),
        Target::Reference => {
            // The voice's own language already has `reference.wav`.
            let own = super::shell::active_voice(s).map(|v| v.lang().to_owned()).unwrap_or_default();
            list.retain(|l| l.code != own);
        }
        _ => {}
    }
    list
}

/// A compact chip showing `label` that opens the picker for `target`. `key` anchors the list.
pub fn chip(p: &Palette, key: &str, label: &str, target: Target, enabled: bool) -> El {
    let k = key.to_owned();
    dropdown_chip(*p, key, &GLOBE, label, format!("Language: {label}"), enabled, move |s: &mut AppState| open(s, target, &k))
}

fn matches(l: &SpeechLanguage, q: &str) -> bool {
    q.is_empty() || l.code == q || l.name.to_lowercase().contains(q) || l.native.to_lowercase().contains(q)
}

fn row_el(p: &Palette, target: Target, l: &SpeechLanguage, on: bool) -> El {
    let code = l.code.clone();
    let main = display(l);
    let sub = if main == l.name || l.code == "auto" { String::new() } else { l.name.clone() };
    row().key(format!("lang-{}", l.code)).role("button").label(l.name.clone()).checked(on).w_full().h(32.0).px(12.0).gap(10.0).center_y().radius(16.0)
        .fill(if on { p.accent.alpha(0.16) } else { Color::TRANSPARENT })
        .hover(p.glass_hover).focusable(true).motion(motion::UI)
        .on_press(move |s: &mut AppState| pick(s, target, &code))
        .child(text(main).font(ty::LABEL).color(p.text).truncate().a11y_hidden())
        .child(text(sub).font(ty::CAPTION).color(p.text_faint).grow(1.0).min_w(0.0).truncate().a11y_hidden())
        .child(el().w(16.0).children(on.then(|| icon_el(&icons::CHECK, 14.0, p.accent))))
}

/// The open picker, added at the root above the screens. `None` when closed.
pub fn overlay(s: &AppState, p: &Palette) -> Option<El> {
    let (target, anchor) = s.langpick.open.clone()?;
    let scrim = el().key("lang-scrim").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).on_press(close);
    let q = s.langpick.search.text.trim().to_lowercase();
    let cur = current(s, target);
    let opts = options(s, target);
    let mut list = col().w_full().gap(1.0);
    if q.is_empty() {
        list = list.child(text("Common").font(ty::OVERLINE).color(p.text_faint).px(12.0).pt(4.0).pb(2.0));
        for l in opts.iter().filter(|l| l.common) {
            list = list.child(row_el(p, target, l, l.code == cur));
        }
        list = list.child(text("All languages").font(ty::OVERLINE).color(p.text_faint).px(12.0).pt(10.0).pb(2.0));
        for l in opts.iter().filter(|l| !l.common) {
            list = list.child(row_el(p, target, l, l.code == cur));
        }
    } else {
        let found: Vec<&SpeechLanguage> = opts.iter().filter(|l| matches(l, &q)).collect();
        if found.is_empty() {
            list = list.child(text("No language matches.").font(ty::CAPTION).color(p.text_faint).px(12.0).py(8.0));
        }
        for l in found {
            list = list.child(row_el(p, target, l, l.code == cur));
        }
    }
    let title = match target {
        Target::Speak => "Speak in",
        Target::RedubFrom => "Spoken language",
        Target::RedubTo => "Translate to",
        Target::Reference => "Add a reference for",
        Target::MediaRef => "Media reference for",
    };
    let search = kit::field(p, &s.langpick.search, "Search languages", |s: &mut AppState| &mut s.langpick.search).on_submit(move |s: &mut AppState, _| {
        let q = s.langpick.search.text.trim().to_lowercase();
        if let Some(l) = options(s, target).into_iter().find(|l| matches(l, &q)) {
            pick(s, target, &l.code);
        }
    });
    let panel = col().key("lang-menu").w(330.0).padding(10.0).gap(8.0).radius(radius::PANEL + 4.0)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 0.97))
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(18.0, 50.0, p.shadow)
        .backdrop_blur(28.0)
        .anchored(anchor, Side::Bottom, Align::End, 8.0)
        .transition(rune::Transition::scale(0.98))
        .child(text(title).font(ty::TITLE).color(p.text).px(4.0))
        .child(row().w_full().child(search))
        .child(col().key("lang-list").w_full().h(330.0).scroll(ScrollSpec { stick_bottom: false }).scroll_edge_fade(12.0).child(list));
    Some(el().key("lang-menu-layer").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).child(scrim).child(glass_edge(*p, panel)))
}

/// A small globe (outline, like the other icons).
pub static GLOBE: rune::Icon = rune::Icon {
    name: "globe",
    view: 16.0,
    body: concat!(
        r#"<g fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">"#,
        r#"<circle cx="8" cy="8" r="6"/><path d="M2 8H14"/><path d="M8 2C9.8 4 9.8 12 8 14C6.2 12 6.2 4 8 2Z"/>"#,
        "</g>"
    ),
};
