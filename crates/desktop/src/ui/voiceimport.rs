use std::path::{Path, PathBuf};

use rune::{Color, El, ScrollSpec, TextEdit, col, el, row, text};

use super::components::{Button, Tone};
use super::lang;
use super::shell::{self, VoiceInfo};
use super::theme::{Palette, radius, ty};
use super::voices::kit;
use crate::engine::EngineMsg;
use crate::state::AppState;

#[derive(Default)]
pub struct ImportState {
    open: bool,
    name: TextEdit,
    model: TextEdit,
    index: TextEdit,
    reference: TextEdit,
    language: String,
    consent: bool,
    error: Option<String>,
    pending: Option<u64>,
    picking: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    Model,
    Index,
    Reference,
}

struct Form<'a> {
    name: &'a str,
    model: &'a str,
    index: &'a str,
    reference: &'a str,
    consent: bool,
}

impl ImportState {
    fn form(&self) -> Form<'_> {
        Form {
            name: self.name.text.trim(),
            model: self.model.text.trim(),
            index: self.index.text.trim(),
            reference: self.reference.text.trim(),
            consent: self.consent,
        }
    }

    fn edit(&mut self, field: Field) -> &mut TextEdit {
        match field {
            Field::Model => &mut self.model,
            Field::Index => &mut self.index,
            Field::Reference => &mut self.reference,
        }
    }
}

fn has_ext(path: &str, ext: &str) -> bool {
    Path::new(path).extension().is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

fn problem(f: &Form) -> Option<&'static str> {
    if !f.model.is_empty() && !has_ext(f.model, "pth") {
        return Some("The model must be a .pth file.");
    }
    if !f.index.is_empty() && !has_ext(f.index, "index") {
        return Some("The index must be an .index file.");
    }
    None
}

fn can_import(f: &Form) -> bool {
    !f.name.is_empty() && !f.model.is_empty() && !f.reference.is_empty() && f.consent && problem(f).is_none()
}

/// `anna_300e_9000s.pth` -> `anna`.
fn name_from_model(path: &str) -> String {
    let stem = Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or_default();
    let parts: Vec<&str> = stem.split('_').collect();
    let trained = |p: &[&str]| {
        let [e, s] = p else { return false };
        let num = |t: &str, unit: char| {
            t.strip_suffix(unit).is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        };
        num(e, 'e') && num(s, 's')
    };
    if parts.len() > 2 && trained(&parts[parts.len() - 2..]) {
        parts[..parts.len() - 2].join("_")
    } else {
        stem.to_owned()
    }
}

pub fn open(s: &mut AppState) {
    let mut name = TextEdit::new("");
    name.focused = true;
    s.voiceimport = ImportState { open: true, name, language: s.record.lang().to_owned(), ..ImportState::default() };
}

fn close(s: &mut AppState) {
    if s.voiceimport.pending.is_none() {
        s.voiceimport = ImportState::default();
    }
}

pub fn set_language(s: &mut AppState, code: &str) {
    s.voiceimport.language = code.to_owned();
}

pub fn language(s: &AppState) -> String {
    s.voiceimport.language.clone()
}

fn set_path(s: &mut AppState, field: Field, path: PathBuf) {
    let path = path.display().to_string();
    let f = &mut s.voiceimport;
    if field == Field::Model && f.name.text.trim().is_empty() {
        f.name = TextEdit::new(name_from_model(&path));
    }
    *f.edit(field) = TextEdit::new(path);
    f.error = None;
}

fn choose(s: &mut AppState, field: Field) {
    if s.voiceimport.picking || s.voiceimport.pending.is_some() || s.shell.scripted() {
        return;
    }
    s.voiceimport.picking = true;
    let (title, kind, exts) = match field {
        Field::Model => ("Choose the RVC model", "RVC model", vec!["pth".to_owned()]),
        Field::Index => ("Choose the model's index", "Index", vec!["index".to_owned()]),
        Field::Reference => {
            ("Choose a short clip of the same voice", "Audio and video", super::formats::import_extensions(s))
        }
    };
    std::thread::spawn(move || {
        let mut dialog = rfd::FileDialog::new().set_title(title);
        if !exts.is_empty() {
            dialog = dialog.add_filter(kind, &exts);
        }
        let choice = dialog.pick_file();
        super::events::post(move |s| {
            s.voiceimport.picking = false;
            if let Some(path) = choice {
                set_path(s, field, path);
            }
        });
    });
}

fn submit(s: &mut AppState) {
    let f = s.voiceimport.form();
    if s.voiceimport.pending.is_some() || !can_import(&f) {
        return;
    }
    let mut params = serde_json::json!({
        "name": f.name,
        "rvc_model": f.model,
        "reference_path": f.reference,
        "language": s.voiceimport.language,
    });
    if !f.index.is_empty() {
        params["rvc_index"] = f.index.into();
    }
    match shell::request(s, "voices.import", params) {
        Some(id) => {
            s.voiceimport.pending = Some(id);
            s.voiceimport.error = None;
        }
        None => {
            s.voiceimport.error =
                Some("The voice engine is not running: press the status at the top right, then Retry.".into())
        }
    }
}

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    match msg {
        EngineMsg::Exited { .. } if s.voiceimport.pending.take().is_some() => {
            s.voiceimport.error = Some("The voice engine stopped. Try again once it is running.".into());
        }
        EngineMsg::Response { id, result } if s.voiceimport.pending == Some(*id) => {
            s.voiceimport.pending = None;
            match result.as_ref().map(|v| v.get("voice").and_then(VoiceInfo::from_json)) {
                Ok(Some(info)) => {
                    let id = info.id.clone();
                    s.shell.voices.retain(|v| v.id != id);
                    s.shell.voices.push(info);
                    super::voices::select(s, &id);
                    s.voiceimport = ImportState::default();
                    shell::refresh_voices(s);
                    shell::toast(s, Tone::Success, "Voice imported.");
                }
                Ok(None) => s.voiceimport.error = Some("The engine did not return the new voice.".into()),
                Err(e) => s.voiceimport.error = Some(e.message.clone()),
            }
        }
        _ => {}
    }
}

const LABEL_W: f32 = 128.0;

fn line(p: &Palette, label: &str, control: El) -> El {
    row()
        .w_full()
        .gap(12.0)
        .center_y()
        .child(text(label.to_owned()).font(ty::LABEL).color(p.text).w(LABEL_W))
        .child(el().grow(1.0).min_w(0.0).child(control))
}

fn path_line(s: &AppState, p: &Palette, label: &str, field: Field, placeholder: &str) -> El {
    let f = &s.voiceimport;
    let edit = match field {
        Field::Model => &f.model,
        Field::Index => &f.index,
        Field::Reference => &f.reference,
    };
    let key = match field {
        Field::Model => "import-model",
        Field::Index => "import-index",
        Field::Reference => "import-reference",
    };
    let input = kit::field(p, edit, placeholder, move |s: &mut AppState| s.voiceimport.edit(field)).key(key);
    let pick = Button::new("Choose…")
        .small()
        .secondary()
        .key(format!("{key}-choose"))
        .disabled(f.picking || f.pending.is_some())
        .on_press(move |s: &mut AppState| choose(s, field))
        .build(*p);
    line(p, label, row().w_full().gap(8.0).center_y().child(input).child(pick))
}

pub fn overlay(s: &AppState, p: &Palette) -> Option<El> {
    let f = &s.voiceimport;
    if !f.open {
        return None;
    }
    let busy = f.pending.is_some();
    let form = f.form();
    let head = col().w_full().gap(4.0).child(text("Import voice model").font(ty::TITLE).color(p.text)).child(
        text("Use an RVC or Applio model you already have. It needs no training.")
            .font(ty::CAPTION)
            .color(p.text_dim)
            .wrap(),
    );
    let name = kit::field(p, &f.name, "Voice name", |s: &mut AppState| &mut s.voiceimport.name).key("import-name");
    let code = if f.language.is_empty() { "en".to_owned() } else { f.language.clone() };
    let chip = lang::chip(p, "import-lang", &lang::label(&code), lang::Target::Import, !busy);
    let clip_hint = row().w_full().gap(12.0).child(el().w(LABEL_W)).child(
        text("Speech is cloned from this clip; the model refines the timbre.")
            .font(ty::CAPTION)
            .color(p.text_dim)
            .wrap()
            .grow(1.0)
            .min_w(0.0),
    );
    let consent = kit::checkbox(p, f.consent, "This is my voice, or I have the owner's permission to use it", |s| {
        if s.voiceimport.pending.is_none() {
            s.voiceimport.consent = !s.voiceimport.consent;
        }
    })
    .key("import-consent");
    let mut fields = col()
        .w_full()
        .gap(10.0)
        .child(line(p, "Name", name))
        .child(path_line(s, p, "Model (.pth)", Field::Model, "The voice model file"))
        .child(path_line(s, p, "Index (optional)", Field::Index, "The .index file, if you have one"))
        .child(path_line(s, p, "Reference clip", Field::Reference, "A short, clean recording of the same voice"))
        .child(clip_hint)
        .child(line(p, "Language", row().child(chip)))
        .child(el().h(2.0))
        .child(consent);
    if let Some(why) = problem(&form) {
        fields = fields.child(kit::notice(p, Tone::Warning, why));
    }
    if let Some(err) = &f.error {
        fields = fields.child(kit::notice(p, Tone::Danger, err).key("import-error"));
    }
    let body = col()
        .key("import-fields")
        .w_full()
        .shrink()
        .min_h(0.0)
        .scroll(ScrollSpec { stick_bottom: false })
        .scroll_edge_fade(12.0)
        .child(fields);
    let actions = row()
        .w_full()
        .justify_end()
        .gap(10.0)
        .child(Button::new("Cancel").small().secondary().key("import-cancel").disabled(busy).on_press(close).build(*p))
        .child(
            Button::new(if busy { "Importing…" } else { "Import" })
                .small()
                .primary()
                .key("import-submit")
                .disabled(busy || !can_import(&form))
                .on_press(submit)
                .build(*p),
        );
    let card = col()
        .key("import-dialog")
        .w(600.0)
        .shrink()
        .min_h(0.0)
        .padding(22.0)
        .gap(14.0)
        .radius(radius::PANEL)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 1.0))
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(18.0, 50.0, p.shadow)
        .on_press(|_: &mut AppState| {})
        .child(head)
        .child(body)
        .child(actions);
    Some(
        col()
            .key("import-layer")
            .pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0))
            .padding(24.0)
            .center()
            .fill(Color::BLACK.alpha(0.55))
            .on_press(|_: &mut AppState| {})
            .child(card),
    )
}

fn scene_base(s: &mut AppState) {
    super::voices::scenes()[0].1(s);
    open(s);
}

fn scene_empty(s: &mut AppState) {
    scene_base(s);
}

fn fill(s: &mut AppState) {
    let f = &mut s.voiceimport;
    f.name = TextEdit::new("Anna");
    f.model = TextEdit::new("/home/me/Models/anna_300e_9000s.pth");
    f.index = TextEdit::new("/home/me/Models/added_IVF256_Flat_nprobe_1_anna_v2.index");
    f.reference = TextEdit::new("/home/me/Recordings/anna-clean.wav");
    f.language = "uk".into();
    f.consent = true;
}

fn scene_filled(s: &mut AppState) {
    scene_base(s);
    fill(s);
}

fn scene_error(s: &mut AppState) {
    scene_filled(s);
    s.voiceimport.error = Some(
        "The model file is a training checkpoint (G_ or D_), not a voice model. Choose the smaller .pth that RVC or \
         Applio saves for inference."
            .into(),
    );
}

fn scene_wrong_type(s: &mut AppState) {
    scene_filled(s);
    s.voiceimport.model = TextEdit::new("/home/me/Downloads/anna-rvc.zip");
    s.voiceimport.consent = false;
}

pub fn scenes() -> &'static [super::Scene] {
    &[
        ("voices-import", scene_empty),
        ("voices-import-filled", scene_filled),
        ("voices-import-error", scene_error),
        ("voices-import-wrong-type", scene_wrong_type),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form<'a>(name: &'a str, model: &'a str, index: &'a str, reference: &'a str, consent: bool) -> Form<'a> {
        Form { name, model, index, reference, consent }
    }

    #[test]
    fn import_needs_name_model_reference_and_consent() {
        assert!(can_import(&form("Anna", "/m/anna.pth", "", "/r/clip.mp3", true)));
        assert!(can_import(&form("Anna", "C:\\m\\ANNA.PTH", "C:\\m\\anna.Index", "clip.wav", true)));
        assert!(!can_import(&form("", "/m/anna.pth", "", "/r/clip.mp3", true)));
        assert!(!can_import(&form("Anna", "", "", "/r/clip.mp3", true)));
        assert!(!can_import(&form("Anna", "/m/anna.pth", "", "", true)));
        assert!(!can_import(&form("Anna", "/m/anna.pth", "", "/r/clip.mp3", false)));
    }

    #[test]
    fn wrong_file_types_are_named() {
        let zip = form("Anna", "/m/anna.zip", "", "/r/clip.mp3", true);
        assert_eq!(problem(&zip), Some("The model must be a .pth file."));
        assert!(!can_import(&zip));
        let idx = form("Anna", "/m/anna.pth", "/m/anna.pth", "/r/clip.mp3", true);
        assert_eq!(problem(&idx), Some("The index must be an .index file."));
        assert!(!can_import(&idx));
        assert_eq!(problem(&form("", "", "", "", false)), None);
    }

    #[test]
    fn trimmed_fields_count() {
        let mut s = ImportState { consent: true, ..ImportState::default() };
        s.name = TextEdit::new("   ");
        s.model = TextEdit::new(" /m/a.pth ");
        s.reference = TextEdit::new("/r/a.wav");
        assert!(!can_import(&s.form()));
        s.name = TextEdit::new(" Anna ");
        assert!(can_import(&s.form()));
        assert_eq!(s.form().model, "/m/a.pth");
    }

    #[test]
    fn the_name_comes_from_the_model_file() {
        assert_eq!(name_from_model("/m/anna_300e_9000s.pth"), "anna");
        assert_eq!(name_from_model("/m/mimvox_taras_320e_6720s.pth"), "mimvox_taras");
        assert_eq!(name_from_model("/m/Narrator.pth"), "Narrator");
        assert_eq!(name_from_model("/m/v2_e_s.pth"), "v2_e_s");
    }
}
