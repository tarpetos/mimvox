# Engine protocol (app <-> engine)

The desktop app (`app/`, Rust) starts the ML engine (`engine/`, Python) as a child process and talks to it over
**stdin/stdout, one JSON object per line (UTF-8, `\n`-terminated)**. The engine writes logs only to **stderr**.
Audio is never sent inline: both sides exchange **absolute file paths** (WAV, mono, 16-bit PCM unless stated).

This file is the single source of truth. Rust types live in `app/src/engine/protocol.rs`, Python handlers in
`engine/src/mimvox_engine/`. Change this file first, then both sides.

## Process

```
<python> -m mimvox_engine serve --runtime <runtime.toml> --data-dir <data dir>
```

- `<python>` is the engine virtualenv interpreter named in `runtime.toml` (`[python] engine`), with
  `PYTHONPATH=<repo or bundle>/engine/src`. The reading presets are found by walking up from the engine sources to the
  first folder that has `presets/<lang>/*.md` (or a legacy flat `presets/*.md`, read as English): `<repo>/presets` in
  the repository and `<RES>/presets` next to `<RES>/engine/src` in an installed bundle. `MIMVOX_PRESETS=<dir>`
  overrides it (the folder that holds the `<lang>` folders).
- On start the engine prints exactly one line `{"event":"ready","job":null,"data":{...hello result...}}` once it can
  answer requests. Models load in the background (about 20-30 s from a cold start, including a warm-up); `models` in the
  `ready` data is `loading` for both until the `models` event says otherwise.
- The engine exits when stdin closes or on `shutdown`. Stdout carries the protocol only; logs go to stderr.
- Requests are handled concurrently; replies may arrive in any order, match them by `id`.
- **Process model (event-driven).** The engine runs on one asyncio event loop; nothing in the engine or its workers
  polls. Every request is a task; long work (`*.generate`, Redub, training) is a job task that `*.cancel` cancels. Blocking
  work runs in executors (a single-thread **GPU executor** for in-process model calls, a small IO pool for file/CPU work)
  or in worker processes (OmniVoice speech, RVC, Whisper/NLLB) whose request queues serialize their model calls. Events and results are written by one writer in order.
- **stdin.** The engine (and each worker) moves its protocol pipe to a private descriptor and points its standard input
  (fd 0 and, on Windows, `STD_INPUT_HANDLE`) at NUL before reading. On Windows a thread waiting in `ReadFile` on a
  synchronous pipe blocks every other request on that pipe's file object, and loading DLLs (`import torch`) makes one on
  the standard input handle, so a model load hung until the next line arrived (this was the `redub.prepare`
  deadlock). Lines are read by one dedicated thread on Windows and by an asyncio pipe reader elsewhere. A client needs
  to do nothing for this; it only means the engine never depends on more input arriving.
- **Workers** (RVC, speech): child processes with the same JSON-lines framing. The engine side has one owner task per
  worker that consumes a request queue, writes one request (`{"id","op",...}`), and awaits its reply; a reader task
  resolves replies by `id` and turns `progress` lines into callbacks. A worker that exits fails every pending request
  at once (the exit is awaited, not polled) and the next request starts a new process. Workers run ops on one thread
  and use timers for idle unload/exit.
- **`shutdown` / stdin EOF.** The engine answers `shutdown` with `{}`, lets requests in flight answer (up to 10 s),
  stops a running training's Applio process tree (Windows: `taskkill /T`; the run stays resumable as an interrupted
  one), cancels the remaining jobs, stops the workers (`quit`, exit awaited, killed after 5 s) and exits. A request
  still running at that point answers `cancelled`. No child process outlives it.

Command line helper (no running engine needed, prints the `voices.import` result as JSON):

```
<python> -m mimvox_engine import --data-dir <data dir> --name <name> [--reference <wav>] [--dataset-dir <dir>]
                                     [--rvc-model <pth>] [--rvc-index <index>] [--language <code>]
```

## Message shapes

| Direction | Shape |
| --- | --- |
| app -> engine request | `{"id":"<string>","method":"<name>","params":{...}}` (`params` may be omitted) |
| engine -> app success | `{"id":"<same>","result":{...}}` |
| engine -> app failure | `{"id":"<same>","error":{"code":"<snake_case>","message":"<human text>"}}` |
| engine -> app event | `{"event":"<name>","job":"<job id or null>","data":{...}}` |

Requests may be sent while others run. Long work (`tts.generate`, `train.start`) answers at once with a `job` id
(`job-1`, `job-2`, ...) and then reports through events carrying that `job`. Events without a job have `"job":null`.
Error codes: `bad_request`, `not_found`, `busy`, `not_ready`, `paused`, `cancelled`, `internal`. An unknown method is
`not_found`; a malformed line is `bad_request`. As a last resort a job that crashes without reporting itself emits
`job.error {"code":"internal","message"}`.

### Events that are not tied to a request

| Event | Data | When |
| --- | --- | --- |
| `ready` | the `hello` result | once, at start |
| `models` | `{"tts":"ready|loading|paused|missing","rvc":"ready|loading|paused|missing"}` | every time a model state changes (start-up, training hand-over, reload) |
| `voice.warm` | `{"id","ok":true,"elapsed_s"}` or `{"id","ok":false,"message"}` | a `voices.warm` finished |

`paused` means the models were unloaded on purpose while a voice trains (the GPU and host RAM are handed to Applio); `tts.generate` and redub answer `not_ready` until both are `ready` again after `train.done`/`train.error`.

| `redub.prepared` | `{"ok":true,"device":"cuda|cpu","model":"turbo","elapsed_s"}` or `{"ok":false,"message"}` | a `redub.prepare` finished |

Job events of the reference-from-media job (`voices.media.progress`, `voices.media.done`, `voices.media.error`) are
described in [Reference from media](#reference-from-media-voicesreference_from_media--voicesmedia_cancel--voicesclear_media).

Job events of the Redub feature (`redub.progress`, `redub.transcript`, `redub.done`, `redub.error`) are described in the
[Redub](#redub-transcribe-convert-and-re-speak-a-file) section; `audio.export` and `formats.list` in
[Audio formats](#audio-formats-import-and-export).

## runtime.toml (written by `tools/setup/setup.py`)

```toml
[python]
engine = "/abs/path/runtime/engine/.venv/bin/python"   # runs the engine, the OmniVoice worker and the speech worker
rvc    = "/abs/path/runtime/applio/.venv/bin/python"   # runs the RVC worker and training (Applio deps)

[paths]
applio          = "/abs/path/runtime/applio"           # Applio repo with prerequisites downloaded
omnivoice_model = "/abs/path/runtime/models/omnivoice" # k2-fsa/OmniVoice snapshot (3.3 GB)

[device]
backend = "cuda"   # cuda | mps | cpu (detected by setup)
```

`[python] omnivoice` (optional) runs the OmniVoice worker with another interpreter; by default it is the engine's. When
`omnivoice_model` is absent the engine looks in `<runtime>/models/omnivoice` (next to the Applio folder) and reports
`tts: missing` when nothing is there. Runtimes of earlier versions had `engine` in `runtime/index-tts/.venv` and an
`indextts` path; setup rewrites the file (the old `runtime/index-tts` is not used any more).

## Where the data directory is

The runtime alone is about 31 GB, so it can live on any drive. Setup asks where (default below) and the app finds it
in this order:

1. `MIMVOX_DATA` environment variable (an absolute path);
2. the pointer file `location.toml` in the fixed per-user config folder, content `data_dir = "<absolute path>"`;
3. the default data directory.

| Platform | Default data directory | `location.toml` |
| --- | --- | --- |
| Windows | `%LOCALAPPDATA%\mimvox` (never Roaming) | `%LOCALAPPDATA%\mimvox\location.toml` |
| macOS | `~/Library/Application Support/mimvox` | `~/Library/Application Support/mimvox/location.toml` |
| Linux | `$XDG_DATA_HOME` or `~/.local/share`, plus `/mimvox` | `${XDG_CONFIG_HOME:-~/.config}/mimvox/location.toml` |

Setup writes `location.toml` only when the chosen data directory is not the default one (and removes a stale one when
it is the default again). `runtime.toml` always lives in the data directory itself. Running setup again with an
existing runtime in the chosen directory reuses it and only fills in what is missing.

### Setup and moving the install (subprocess interface)

`setup.cmd` / `./setup.sh` take `--data-dir DIR`, `--yes`, `--force` and others (`--help`). Before anything is
downloaded setup prints the install location, the free space there and the size (about 28 GB: engine + OmniVoice ~11 GB,
Applio/RVC ~12 GB, Redub models ~4.5 GB, plus ~0.1 GB per voice), or "Existing runtime found, nothing big to download". It
refuses when the drive has under 32 GB free unless `--force`. An existing `runtime/index-tts` (IndexTTS-2.5 of earlier
versions, ~19 GB) is deleted only with `--remove-indextts` or a "y" at the prompt in an interactive terminal; with `--yes`
setup only prints a hint. Without `--yes` it asks (interactive; the app opens a terminal running plain
`setup.cmd`/`setup.sh`, so this prompt is what the user sees); with `--yes` it needs `--data-dir` (or
`MIMVOX_DATA`) and never picks a place silently.

Moving an existing install, run by the app as a subprocess **after stopping the engine** (it refuses while the engine or
anything started from the old runtime is running):

```
setup.cmd --move-to <new dir> [--data-dir <current dir>] [--force]
```

The current directory defaults to `location.toml`, then the default directory. The new directory must be empty or
absent, and not inside the old one. Output is line based, every line starts with `[move] `:

| Line | Meaning |
| --- | --- |
| `[move] 37% 11.4/31.0 GB` | copy progress (printed at each percent; a same-volume move is a rename and prints `[move] 100% done`) |
| `[move] ...` (other text) | information or `warning: ...` (for example a console launcher `.exe` that still holds the old path; the engine does not use them) |
| `[move] done` | success, exit code 0 |
| `[move] failed: <reason>` | failure, exit code 1 (the old copy is untouched unless the copy was verified) |

Steps: refuse if running, check free space, rename (same volume) or copy (other volume, resumable: rerun the same command
to continue an interrupted copy), verify (same files and sizes), rewrite absolute paths inside the virtual environments
(`.pth`, editable finders, `direct_url.json`, `pyvenv.cfg`, scripts), rewrite `runtime.toml`, update `location.toml`, run
the quick check, and only then delete the old directory.

## Data directory layout (owned by the engine; the app only reads files the engine names)

```
<data>/voices/<voice_id>/voice.json          # {"id","name","created","preset","language","reference":"reference.wav"} (+ after training or
                                             #  import: "trained":true,"trained_epochs","trained_at","imported"?; while a training runs or is paused: "training_run" /
                                             #  "training_paused", see Pause and resume)
<data>/voices/<voice_id>/reference.wav       # the 15-20 s clip the speech model copies tempo and manner from (picked by dataset.add/remove, or by voices.set_reference)
<data>/voices/<voice_id>/dataset/part<N>.wav # recorded parts (N from 1), trimmed and normalised
<data>/voices/<voice_id>/dataset/meta.json   # per-part quality results, speaking rate, candidate reference windows, the chosen reference,
                                             #  and "media": the media pool (files, best windows), see Reference from media
<data>/voices/<voice_id>/reference_media.wav # optional media pool (voices.reference_from_media): many media files of the speaker,
                                             #  cleaned and joined (mono 16-bit 44.1 kHz, at most 20 min); not a dataset part, never trained on
<data>/tmp/refcand/<voice_id>/               # preview WAVs of voices.reference_candidates (rewritten on every call)
<data>/voices/<voice_id>/rvc/model.pth       # trained or imported RVC model
<data>/voices/<voice_id>/rvc/model.index
<data>/outputs/<YYYYMMDD_HHMMSS>.wav|.mp3    # generated speech (44.1 kHz; a `_2`, `_3` suffix on a name clash)
<data>/outputs/redub_<YYYYMMDD_HHMMSS>.wav|.mp3  # Redub results (same rules)
<data>/runtime/models/whisper/               # Whisper turbo (large-v3-turbo.pt, 1.6 GB), fetched by setup (else on the first Redub)
<data>/runtime/models/nllb-200-distilled-600M/  # NLLB-200 translation model (2.5 GB), fetched by setup with `hf download`; never auto-downloaded
<data>/runtime/engine/.venv/                 # the engine environment (setup): engine, OmniVoice worker, speech worker
<data>/runtime/models/omnivoice/             # OmniVoice weights k2-fsa/OmniVoice (3.3 GB), fetched by setup with `hf download`; never auto-downloaded
<data>/voices/<voice_id>/reference.omnivoice.json  # OmniVoice's cut of reference.wav: {"version","sig":[mtime_ns,size],"start","end","text"}
                                             #  (whole clip up to 12 s, else 7-12 s of whole sentences, + Whisper transcript; rebuilt when it changes)
<data>/voices/<voice_id>/reference_<lang>.wav  # optional per-language reference (e.g. reference_uk.wav, 7-12 s of speech in that
                                             #  language); OmniVoice uses it instead of reference.wav for <lang> (+ its own .omnivoice.json)
<data>/tmp/                                  # scratch (job chunks, training logs), wiped at every engine start
```

Voice ids are `[a-z0-9][a-z0-9_-]*`, made from the name (`Anna` -> `anna`, a second one `anna-2`). Names in another
script are transliterated first: Cyrillic by the Ukrainian national system (KMU 2010; `Мій голос` -> `mii-holos`,
`Згода` -> `zghoda`, word-initial `є ї й ю я` -> `ye yi y yu ya`; Russian/Belarusian/other Cyrillic letters too), Greek
letters, Latin accents dropped (`Zoë` -> `zoe`, `ß` -> `ss`); a name with nothing left gives `voice`.

## Methods

### `hello` -> engine info
`{}` -> `{"version":"0.1.0","device":{"backend":"cuda|mps|cpu","name":"NVIDIA GeForce RTX 4070","vram_gb":12.0},
"models":{"tts":"ready|loading|paused|missing","rvc":"ready|loading|paused|missing"},
"languages":[{"code":"en","name":"English","native":"English","common":true},{"code":"uk","name":"Ukrainian",
"native":"Українська","common":true},...]}`
- `languages`: every language the speech model speaks (see [Speech model and languages](#speech-model-and-languages-omnivoice)),
  `name` in English, `native` in the language itself, the common ones first (`common: true`, in the order
  `en uk de fr es pl it pt cs ru`), then the rest by English name. The same list is in the `ready` line and answers
  `tts.languages {}` -> `{"languages":[...]}`.
- `device.name` is read from `nvidia-smi` for CUDA (`"NVIDIA GPU"` when that fails), `"Apple Silicon GPU"` for MPS and
  `"CPU"` for CPU. `vram_gb` is the total VRAM rounded to 0.1 and is `0.0` for MPS and CPU.
- `models`: `loading` until the model is loaded and warmed up, `missing` when it could not be loaded.

### `voices.list`
`{}` -> `{"voices":[{"id","name","created","preset","language":"en","dataset_minutes":6.4,"parts":5,"trained":true,
"trained_epochs":300,"trained_at":"2026-11-11T10:20:30","training_paused":null,"references":["uk"]}]}`, oldest first.
`references` lists the languages that have their own reference clip (`reference_<lang>.wav`, see
[Per-language references](#per-language-references-voicesadd_reference--voicesremove_reference)). `trained_epochs` and `trained_at` are `null`
when the voice has no RVC model (or when the number is unknown). `created` is UTC ISO 8601 (`...Z`), `trained_at` is
local time without zone. `preset` is the preset id the voice was recorded with or `null`. `language` is the language
the voice was recorded in ([language code](#languages); `"en"` for voices made before languages existed): the preset
`preset` is looked up in that language (for the reference-clip part titles). `training_paused` is `null`
or `{"epoch","total_epochs","paused_at","interrupted"}`: a training of this voice is paused (or was interrupted by a
closed or crashed app, `interrupted: true`) and `epoch` is the epoch of the saved checkpoint where `train.resume`
continues (see [Pause and resume](#pause-and-resume)).

### `voices.create` / `voices.rename` / `voices.delete`
`{"name","preset"?,"language"?}` -> `{"voice":{...as in list...}}` (a new voice has no dataset and no reference;
`language` defaults to `"en"`, `"auto"` also means `"en"`, an unsupported code is `bad_request`) ·
`voices.rename {"id","name"?,"language"?}` -> `{"voice":{...}}` (`name` renames; `language` changes the recording
language, e.g. when the first texts read into a still empty voice are in another language; at least one of the two) ·
`{"id"}` -> `{}` (removes the whole folder). Names are trimmed to 80 characters.
`not_found` for an unknown id.

### `voices.import` (existing recordings and model; no retraining)
`{"name","reference_path"?,"dataset_dir"?,"rvc_model"?,"rvc_index"?,"preset"?,"language"?}` -> `{"voice":{...as in list...}}`
- Creates a voice, then: every `part<N>.<ext>` in `dataset_dir` (any importable extension, `wav` by default) goes through `dataset.add` (trim, normalise, verdict;
  the reference is chosen by it); `reference_path` (any importable format), when given, replaces `reference.wav` as it is (converted to mono 16-bit WAV, no
  trimming; a later `dataset.add`/`dataset.remove` picks the reference again); `rvc_model` (and `rvc_index`) are copied
  to `rvc/model.pth` / `rvc/model.index` and the voice is marked `trained` with `trained_epochs` read from a
  `..._<N>e_...` file name when present and `"imported":true` in `voice.json`.
- Needs at least a `reference_path` or a `dataset_dir`; `rvc_index` needs `rvc_model`. Errors: `bad_request`,
  `not_found` (a path is missing). On any failure the half-made voice is removed again.

### `voices.warm`
`{"id"}` -> `{"warming":true}` | `{"warming":false,"reason":"already_warm|busy"}`. Call it when the user selects a voice:
it computes the speaker conditioning of the voice's `reference.wav` and, for a trained voice, loads its RVC model
and index, in the background (5-22 s, depending on what is already loaded; the first chunk of the next `tts.generate`
then takes about 5 s instead of about 26 s) and ends with the `voice.warm` event. Call it again after the models have
reloaded (the `models` event shows `ready` again after a training), because the reload drops the warm state. `busy` means a generation is running (it warms the voice itself). Errors: `not_found` (voice, or
no `reference.wav` yet), `not_ready` (speech model not loaded or handed over to training).

### Reference clip (`voices.reference_candidates` / `voices.set_reference`)
The speech model copies the tempo and manner of `reference.wav`, so a slow or halting clip makes the cloned voice slow.

**Automatic pick** (`dataset.add` / `dataset.remove`, CPU-light, numpy/scipy only). Each part is analysed once into up to
10 candidate windows of 15-20 s (cut in pauses of 200 ms or more; widened to 30 s, then fixed windows, when a part has
none) stored in `dataset/meta.json` (`parts.<n>.rate`, `parts.<n>.cands`). Per window:
`rate` = syllable nuclei (peaks of the smoothed 10 ms energy envelope, 70 ms apart, 3 dB prominence) per second of active
time (window minus gaps of 400 ms or more), `pauses` = fraction of the window spent in gaps of 400 ms or more (0-1),
`snr_db` = 90th minus 10th percentile of the frame level. The voice's median rate is the median of the per-part rates.
Score = `0.5 * min(snr, 40)` - `50 * pauses` - `30 * |ln(rate / median)|` - `0.2 * |duration - 17.5|` minus 10 when the
part's preset title suggests unnatural reading (tongue twisters, questions, numbers, dates, times, spelling) minus 40
for clipping; parts rated `bad` are skipped (unless all are). The best window becomes `reference.wav`. An unchanged
pick is not rewritten (a new file makes the speech model condition on the speaker again). A manual pick is kept until its part
is removed or re-recorded.

`voices.reference_candidates {"id"}` -> `{"candidates":[{"source":"part|media","file","part","title","start_s","end_s","duration_s","rate","pauses",
"snr_db","hint":"natural|slow|fast","recommended":bool,"is_current":bool,"preview_path"}],"current":{...}|null,
"rate_median","media":{...}|null}`: the best 5 windows over all parts (at most 2 per part, no overlaps), best first (the first is the
automatic pick, `recommended`). `rate` is syllables per second, `pauses` a fraction (0.12 or more is "many pauses"),
`hint` compares `rate` with `rate_median` (under 0.88 of it `slow`, over 1.12 `fast`). `title` is the reading-text title
of the part (null for a voice without a known preset). `preview_path`: a WAV written to `<data>/tmp/refcand/<id>/`
(rewritten on every call). `current` describes `reference.wav` with the same keys plus `"manual":bool` (its
`preview_path` is `reference.wav` itself); `part` is null when no part explains the file (imported reference).
`candidates` is empty for a voice without recordings. Parts stored before this analysis existed are analysed on the first
call (a second or two). Errors: `bad_request`, `not_found` (voice).
Part candidates have `"source":"part"` and `"file":null`. When the voice has a media pool, its best windows follow the part
candidates (up to 5, at most 2 per file when the pool has 3 or more files) with `"source":"media"`, `"part":null`, `file`
(1-based index of the file in the request's `paths`), `title` (the file name) and `start_s`/`end_s` in the pool; the
first of them is `recommended`, `is_current` marks the window in the pool's target file, `hint` compares with the pool's
own median rate. `media` summarises the pool: `{"files","duration_s","language":null|"<code>","target":"reference.wav"|
"reference_<code>.wav","created","skipped"}` (null without a pool). `current` from the pool has `"source":"media"`,
`file`, `title` (and `start_s`/`end_s` in the pool).

`voices.set_reference {"id","part","start_s","end_s"}` -> `{"reference":{"part","start_s","end_s","duration_s","manual":true,
"rate","pauses"},"warming":bool}`: cuts that range (3-40 s; clamped to the part) into `reference.wav` and records it as a
manual choice. The speaker-conditioning cache is dropped, and the existing `voices.warm` path is started (`warming`
tells whether it started; `voice.warm` ends it), so the next generation uses the new clip. The next `tts.generate` of
that voice re-conditions at the first segment otherwise (about 20 s). Errors: `bad_request`, `not_found` (voice or
part), `busy` (a generation is running).

`voices.set_reference {"id","source":"media","start_s","end_s"}` -> `{"reference":{"source":"media","start_s","end_s",
"duration_s","manual":true,"pool","file","title","rate","pauses","snr_db"},"target":"reference.wav"|"reference_<code>.wav",
"language":null|"<code>","warming":bool}`: cuts that range of the media pool (3-40 s, clamped) into the pool's target
(`reference.wav` when the pool was made for the voice's own language, else `reference_<language>.wav`), deletes the
target's `.omnivoice.json`, drops the conditioning cache and (for `reference.wav`) starts `voices.warm`. Errors:
`bad_request`, `not_found` (voice, or no media pool), `busy`.

A manual choice from the media pool (`meta.json` `reference` with `"source":"media"`) is never replaced by `dataset.add` /
`dataset.remove`; if `reference.wav` goes missing it is cut again from the pool (when it is still the same pool).

- **App side (not part of the wire protocol).** The Speak "Style" panel (shared by Redub) has a **Speed** slider, 0.8x
  ("Slower") to 1.3x ("Faster"), step 0.05, default 1.0x; its value is sent as `speed` with `tts.generate` and
  `redub.render` (respeak) and remembered as `speed` in `<data>/settings.json`. On the Voices screen the active voice's
  card has a "Reference clip" section: the clip in use (part, title, "Natural pace" / "Slow" / "Fast" badge, pause note)
  with a play button, and "Choose..." opening a list of the `voices.reference_candidates` with play buttons and
  "Use this one" (`voices.set_reference`). The candidates are fetched when the Voices screen opens and after a change.

### Per-language references (`voices.add_reference` / `voices.remove_reference`)
A voice speaks every language of the speech model, with the accent of its reference clip. A short clip in another
language (about 10 s, read in that language) is stored next to `reference.wav` and used for speech in that language.

`voices.add_reference {"id","language","path"}` -> `{"language":"uk","duration_s":10.4,"references":["de","uk"]}`:
`path` is any importable audio file (a recording of the app, an imported file). It is converted to mono, DC-removed,
trimmed at both ends (the same trim as `dataset.add`) and normalised, then written to `reference_<language>.wav`
(an existing one is replaced). The speaker-conditioning cache is dropped. `language` must be a language the speech model
speaks. Needs 3-40 s of speech after trimming (`bad_request` otherwise; about 10 s is best). Errors: `bad_request`,
`not_found` (voice or file), `busy` (a generation is running).

`voices.remove_reference {"id","language"}` -> `{"references":[...]}`: deletes `reference_<language>.wav` and its
`.omnivoice.json`. Errors: `bad_request`, `not_found` (voice, or no such reference), `busy`.

- **App side.** The Voices card's "Reference clip" section lists the languages with a clip (play, remove) and "Add a
  language" (language picker): a native sample text to read (the first sentences of that language's `quick` preset, or
  a built-in sentence), record with the Record screen's recorder or import a file, `recording.check` (`expected_s` 10),
  then "Use for <language>" (`voices.add_reference`). Speak shows "Record a <language> reference" when it speaks a
  language the voice was not recorded in and has no clip for.

### Combine media files (`media.combine`)

`media.combine {"paths":["...",...]}` -> `{"path","duration_s","files":[{"name","duration_s","truncated"}],
"skipped":[{"name","path","reason"}],"unused":["name",...]}`. Joins many audio / video files of one speaker into one
mono 16-bit 44.1 kHz WAV under `<data>/tmp/combined/`, cleaned like the media pool below (DC removed, trimmed, long
pauses shortened, loudness normalised, 0.4 s between files, at most 20 minutes; files past that are `unused`). Files
that cannot be read or have no clear speech are `skipped`; none usable -> `bad_request`. The app's Record screen
imports the result as one take (`recording.check` -> `dataset.add`).

### Reference from media (`voices.reference_from_media` / `voices.media_cancel` / `voices.clear_media`)
The speech model clones from one short clip, so one noisy source makes a metallic clone. Instead, many audio / video files
of the same speaker (any importable format) are combined and the cleanest window becomes the reference. CPU only
(numpy/scipy and ffmpeg), no model is loaded.

`voices.reference_from_media {"id","paths":["...",...],"language"?}` -> `{"job","files":n}` at once, then events.
- `language`: omitted, `""`, `"auto"` or the voice's own language: the result is `reference.wav` (a manual choice,
  recorded in `meta.json` `reference` with `"source":"media"`). Another speech language: `reference_<language>.wav`
  (like `voices.add_reference`; `voices.list` then lists it in `references`).
- Per file, in order, on the IO pool: decode to mono 44.1 kHz (only as far as the pool still has room plus 2 min),
  remove DC, trim silence at both ends (as `dataset.add`), shorten inner silences longer than 0.7 s to 0.3 s (only when the
  file's SNR is 15 dB or more), normalise loudness (as `dataset.add`). A file that cannot be read, has no audio track, no
  clear speech or under 3 s of speech is **skipped** (listed with a reason) instead of failing the job.
- The cleaned files are joined with 0.4 s gaps into `reference_media.wav` (mono 16-bit, 44.1 kHz). The pool stops at
  **20 minutes**: a long file is cut at a pause, later files are listed as `unused`.
- Each file is analysed on its own (the pace / SNR analysis of [Reference clip](#reference-clip-voicesreference_candidates--voicesset_reference)) into
  pause-aligned windows of **8-12 s** (the speech model uses a clip of up to 12 s whole), so no window spans two files.
  Score = `0.5 * min(snr, 40)` - `50 * pauses` - `30 * |ln(rate / median)|` - `0.2 * |duration - 10|` (minus 40 for a
  clipping file); `median` is the median rate of the pool's files. The best 24 distinct windows are stored in
  `meta.json` `media.cands`; the best one is written to the target, its `.omnivoice.json` deleted, the conditioning
  cache dropped and (for `reference.wav`) `voices.warm` started.
- The pool, `meta.json` `media` and the reference are installed together at the end; a cancelled or failed job changes
  nothing. A new job replaces the previous pool. The dataset (training data) is never touched.
- Events: `voices.media.progress {"id","stage":"decode|combine|analyse|apply","pct":0-100,"files":n,"file"?,"name"?}`
  (`file` 1-based and `name` while a file is decoded and cleaned: `decode` covers 0-85), then
  `voices.media.done {"id","files":<used>,"duration_s":<pool>,"language":null|"<code>","target","applied":bool,
  "reference":{...as in set_reference...}|null,"warming":bool,"preview_path":<the target>|null,"skipped":[{"name","path",
  "reason"}],"unused":["<name>"],"truncated":["<name>"],"message":null|"<text>","elapsed_s"}` or
  `voices.media.error {"code","message","skipped"?}` (`cancelled`; `bad_request` "None of the files has usable speech."
  with `skipped`; `not_found`, `internal`). `applied` is false (with a `message`) when a generation started meanwhile
  (pick a clip with `voices.set_reference` later) or no 8-12 s window was found.
- Errors at start: `bad_request` (`paths` not a non-empty list of strings, more than 500 files, `language` not a speech
  language), `not_found` (voice), `busy` (a generation is running, or a media job already runs for this voice).
- Memory: one file is decoded at a time (at most about 22 min of float32 audio, ~230 MB) and the pool is kept as 16-bit.

`voices.media_cancel {"job"}` -> `{}`: cancels the job (it ends with `voices.media.error` `cancelled`).

`voices.clear_media {"id"}` -> `{}`: deletes `reference_media.wav` and `meta.json` `media`; references already cut from it
stay. Errors: `not_found` (voice), `busy` (a media job runs for this voice).

- **App side.** The Voices card's reference section has a "Reference from media" block: a language chip (default the
  voice's own language; the shared language picker), a drop well with "Add media..." (native multi-file picker) and, while
  the job runs, "Cleaning · File k of n · name", a progress bar and Cancel. Dropping files anywhere on the Voices screen
  starts the job for the active voice (the well lights up while files hover). When done: "12 files combined · 7:32 ·
  reference set: 11.4 s from file 3", a play button for the new reference, skipped files with reasons, and "Remove
  combined media". The pool's windows appear in the "Choose..." list under "From your media".

### Languages
Reading texts (presets, `text.analyze`) and voices carry a language as an **ISO 639-1 code**. Supported today:

| Code | Language | Native name | Reading pace (words/min) | Phoneme inventory | Analysis |
| --- | --- | --- | --- | --- | --- |
| `en` | English | English | 145 | 39 ARPAbet phonemes (`AA` ... `ZH`) | g2p_en (CMUdict + neural fallback), else NLTK CMUdict, else letter heuristic |
| `uk` | Ukrainian | Українська | 110 | 38 IPA: 6 vowels, 32 consonants incl. soft `dʲ tʲ zʲ sʲ tsʲ lʲ nʲ rʲ dzʲ`, `dz dʒ ɦ g` | rule-based G2P (iotation of я ю є ї, softening by ь and before і я ю є, дж/дз, -ться) |
| `de` | German | Deutsch | 120 | 42 IPA: long/short vowels, `ə ɐ`, `aɪ aʊ ɔʏ`, `ç x pf ts tʃ ŋ ʁ` | grapheme-cluster heuristic (vowel length, final devoicing, ich/ach) |
| `fr` | French | Français | 155 | 34 IPA: 14 vowels incl. nasal `ɛ̃ ɑ̃ ɔ̃`, `j w ɥ`, `ɲ ʁ` | grapheme-cluster heuristic (silent endings, nasal vowels) |
| `es` | Spanish | Español | 140 | 25 IPA: 5 vowels, `j w`, `θ x tʃ ʝ ɲ ɾ r` | rule-based G2P |
| `pl` | Polish | Polski | 110 | 37 IPA: 6 vowels + nasal `ɛ̃ ɔ̃`, `ɕ ʑ ʂ ʐ tɕ dʑ tʂ dʐ dz ts x ɲ w` | rule-based G2P (soft i spellings, rz, nasal vowels) |

- Parameters named `language` accept the code in any case and with a region (`"uk-UA"`, `"en_US"` -> `uk`, `en`).
  Anything else is `bad_request` (the message lists the supported codes). More codes (e.g. `it`, `pt`, `cs`) follow
  the same scheme when presets and an inventory are added.
- The reading pace turns word counts into `est_seconds` / `est_minutes`. It differs per language because words differ
  in length (in the presets a Ukrainian or Polish word has about 2.0 syllables, a German one 1.6, an English one 1.4);
  the figures are estimates that keep the syllable rate of every language at about 3.3-4.2 per second.
- Phoneme symbols in `missing_phonemes` are those of the language's inventory (ARPAbet for English, IPA otherwise).
- The language of a voice (`voices.create` / `voices.import` `language`) does not change how the voice is trained; it
  records which presets it was read from, and it is the language where the RVC index is used at full strength (other
  languages are capped, see [Speech model and languages](#speech-model-and-languages-omnivoice)). What the voice can
  speak is the speech model's list (`hello` `languages`).
- **App side.** Record has a "Reading language" choice (the `presets.list` `languages`, remembered as `record_language`
  in `settings.json`); "Use my own text" offers "Detect" or one of them (`text.analyze` `language`). A new voice is made
  in the language that is read; an empty active voice in another language is switched to it (`voices.rename`
  `language`); one with recordings in another language is kept and a new voice "<name> (<Language>)" is made.

### `presets.list`
`{"language"?}` -> `{"language":"uk","languages":[{"code":"en","name":"English","native":"English"},
{"code":"uk","name":"Ukrainian","native":"Українська"},...],"presets":[{"id":"quick|standard|pro","title",
"description","target_minutes","trains_rvc":bool,"language":"uk","parts":[{"index":1,"title","text","est_seconds"}]}]}`
in the order quick, standard, pro.
- `language`: the language of the presets (default `"en"`; `"auto"` and `null` also mean `"en"`). `languages` lists
  every language that has presets, English first, so the app can offer a language picker. A supported code without
  presets, or an unsupported code, is `bad_request`. A missing presets folder is `not_found`.
- Every language has its own texts written for it (not translated): the same three presets, the same parts and about
  the same reading times, so the word counts differ. `title` and `description` are UI strings (English); part `title`
  and `text` are in the preset's language. `est_seconds` uses the language's reading pace.
- `quick`: about 1 minute, zero-shot only (no training), voice usable at once.
- `standard`: about 6-7 minutes, trains RVC (the validated setup).
- `pro`: about 15-20 minutes, best similarity.

### `text.analyze` (user's own reading text)
`{"text","language"?}` -> `{"language":"uk","language_detected":true,"phonemes_total":38,"words","est_minutes",
"phoneme_coverage":0.0-1.0,"missing_phonemes":["dzʲ",...],"level":"too_short|zero_shot|trainable|great","message",
"parts":[{"index","title","text","est_seconds"}]}`
- `language`: the language the text is read in. Omitted, `null` or `"auto"`: detected from the text (Cyrillic script ->
  `uk`; Latin script -> the language with the most common short words plus language-specific letters such as ä ö ü ß,
  ą ę ł ś ż, ñ ¿ ¡, è ç œ; `en` when nothing points elsewhere, also for an empty text); `language_detected` tells
  whether it was detected. Unsupported code: `bad_request`.
- Words of another script than the language's (e.g. Latin words in a Ukrainian text) count as words but add no
  phonemes; digits are read one by one in the text's language.
- Parts are split at paragraph/sentence boundaries to about 60-90 s each. Reading time is estimated from the word count
  at the language's reading pace.
- `phoneme_coverage` is the share of the language's inventory (`phonemes_total` phonemes) the text contains;
  `missing_phonemes` are in the inventory's order. Coverage under the language's threshold (0.80 for every language
  today) lowers `trainable`/`great` by one level; under 0.95 the `message` names up to four missing sounds with a
  native example word each. Every standard and pro preset covers its whole inventory (1.0).

### `recording.check`
`path` may be any audio or video file the shared decoder reads (see [Audio formats](#audio-formats-import-and-export)).
`{"path","expected_s"?}` -> `{"duration_s","peak","clipping_pct","rms_db","noise_floor_db","snr_db","silence_pct",
"verdict":"ok|warn|bad","issues":["Too quiet: speak closer to the mic", ...]}`
`expected_s` (the estimated reading time of the text) adds a "shorter than expected" warning below 60 percent of it.
Errors: `not_found` (no such file), `bad_request` (unreadable or empty audio).

### `dataset.add` / `dataset.remove` / `dataset.summary`
- `dataset.add {"voice_id","part_index","path","expected_s"?}`: reads any audio file (also the audio track of a
  video; so a user can import an existing file as a dataset part instead of recording), trims leading/trailing silence
  (0.25 s kept), normalises loudness (about -20 dBFS active speech, peak at most 0.9), stores mono 16-bit
  `dataset/part<N>.wav` (resampled to 44.1 kHz only above 48 kHz), replaces an older take of that part, refreshes
  `reference.wav` (see [Reference clip](#reference-clip-voicesreference_candidates--voicesset_reference): a 15-20 s
  window scored by SNR, speaking rate near the voice's median, few long pauses and the part's reading-text title; a
  manual choice is kept). The verdict is about the trimmed take.
  -> `{"part":{"index","duration_s","verdict","issues"},"summary":{...}}`. `bad_request` for takes under 0.3 s or below
  16 kHz; `not_found` for an unknown voice or file.
- `dataset.remove {"voice_id","part_index"}` -> `{"summary":{...}}` (removing a missing part is not an error).
- `dataset.summary {"voice_id"}` -> `{"total_minutes","usable_minutes","parts":[{"index","duration_s","verdict","issues"}],
  "level":"too_short|zero_shot|trainable|great","message","recommended_epochs","has_reference","reference"?}`
  - `usable_minutes` counts only parts whose verdict is not `bad`; `level` is computed from it (under 0.5 min
    `too_short`, under 3 `zero_shot`, under 10 `trainable`, else `great`).
  - `recommended_epochs` is `0` below 3 usable minutes (not trainable), otherwise exactly what `train.start` with
    `epochs: null` would use (`300 * sqrt(6.4 / minutes)` clamped to 150..450, rounded to 10).
  - `has_reference`: `reference.wav` exists. `reference` (present only then): `{"part","start_s","end_s","duration_s"}`,
    the clip cut from that part (plus `"manual":true` after `voices.set_reference`, `"rate"` and `"pauses"` when known). A clip from the media pool is `{"source":"media","start_s","end_s","duration_s","manual":true,"pool","file","title",...}` (no `part`). After `voices.import` with a `reference_path` it can describe an older choice.
  - Parts copied into `dataset/` by hand are analysed on the first summary.

### `train.start` / `train.pause` / `train.resume` / `train.cancel`
`train.start {"voice_id","epochs":null,"restart"?:false}` (null = auto: `recommended_epochs` of the dataset minutes; an integer
1..2000 otherwise) -> `{"job"}`; `train.cancel {"job"}` -> `{}`. `train.pause`, `train.resume` and the paused-run forms of
`train.start` / `train.cancel` are described in [Pause and resume](#pause-and-resume).
- Needs the voice to have at least 1 minute of recorded dataset. A new training starts from scratch and
  replaces `rvc/model.pth` / `model.index` only when it finishes. Errors at start: `bad_request` (epochs out of range,
  no or too little dataset), `not_found`, `not_ready` (Applio runtime missing, or the speech models are still loading),
  `busy` (a training is running, **or a generation is running**: wait for it or `tts.cancel` it first), `paused` (this voice
  has a paused or interrupted training and `restart` is not `true`).
- **GPU hand-over.** The training runs inside `tts.gpu_released`: when `train.start` answers, the OmniVoice worker and
  the RVC worker are stopped, so both `models` entries are `loading` (a `models` event is sent) and `tts.generate` /
  `voices.warm` answer `not_ready` for the whole training. When the job ends (`train.done`, `train.error`, a
  cancel or a pause) the models reload in the background (about 20-30 s, `models` events, `ready` again), and the new RVC model is
  read from disk by the fresh worker.
- Events: `train.progress {"stage":"preprocess|extract|train|index","epoch","total_epochs","pct":0-100,"eta_s"}`
  about every 0.5 s and at each stage change; `train.done {"voice_id","epochs","elapsed_s"}`;
  `train.error {"code","message"}` with `cancelled` or `internal` (out of memory is reported in plain words); `train.paused`
  (see below).
  - `pct` is overall (preprocess 0-5, extract 5-15, train 15-95, index 95-100). `epoch` is 0 outside `train`/`index`
    and 1-based while training. `total_epochs` is the planned count.
  - `eta_s` is `null` until three epoch ends have been measured and always `null` outside the `train` stage;
    afterwards it is the mean of the recent epoch times times the remaining epochs, plus 20 s for export and index.
- On success `voice.json` gets `"trained":true`, `"trained_epochs"` (the epochs actually done) and
  `"trained_at"` (local time, `YYYY-MM-DDTHH:MM:SS`), which `voices.list` reports. The full Applio output is kept in
  `<data>/tmp/train_<voice_id>.log` until the next engine start.

#### Pause and resume
How it works: Applio resumes by itself when `core.py train` is started again on the same model folder
(`<applio>/logs/mimvox_<voice id>/`): it loads the newest `G_*.pth` / `D_*.pth` pair, reads the finished epoch stored in
it and continues with the next one. The engine therefore asks Applio for one checkpoint pair every **5 epochs**
(about 1.3 GB written per save at 40 kHz, HiFi-GAN) and keeps only the newest pair. It does **not** use Applio's
`--save-only-latest` (one file overwritten in place: a kill during the save would destroy the only checkpoint); Applio
writes `G_<step>.pth`/`D_<step>.pth` and the engine deletes the older pair once the next epoch is running, i.e. once the
save is known to be complete. A checkpoint is only trusted when both files are complete zip archives; a torn newer
pair is removed so that Applio picks the older valid one. At most 4 epochs of work are lost.

- `train.pause {"job"}` -> `{"pausing":true}` at once. Only while the job is in the `train` stage (`not_ready` before
  that: preprocessing and feature extraction cannot be paused; `not_found` for an unknown or finished job). The engine
  waits until a checkpoint is safely on disk: it stops immediately when the newest checkpoint is the current epoch,
  otherwise after the next checkpoint save (at most 5 epochs; after 90 s without one it stops and accepts the loss).
  If the training is in its last epoch it just finishes (`train.done` arrives instead of `train.paused`). Then the
  Applio process tree is stopped, **Applio's folder is kept**, `voice.json` gets
  `"training_paused":{"epoch","total_epochs","model_name","paused_at",...}`, and the event
  `train.paused {"voice_id","epoch","total_epochs","paused_at","reached_epoch"}` (job = the training job) ends the job.
  `epoch` is the epoch of the saved checkpoint (**where a resume continues**), `reached_epoch` how far the training had
  got. The speech models are reloaded at once (the GPU hand-over is closed like after `train.done`: `models` events,
  `ready` again after about 50 s), so Speak and Redub work while paused.
- `train.resume {"voice_id"}` -> `{"job"}`. Hands the GPU over again (same `busy` / `not_ready` rules as `train.start`;
  the models must be `ready`, so the app enables Resume only then), skips preprocess and extract (the features are in
  Applio's folder; if the folder is missing or incomplete the full pipeline runs again from epoch 1), repairs the
  checkpoints, and starts `core.py train` with the same total epochs, batch size and cache setting. `train.progress`
  continues from the saved epoch: the first event is `stage:"train"` with `epoch` = the saved epoch and the `pct`
  that epoch had before; `eta_s` is `null` for the first three epochs. Afterwards the index is built and the model
  copied as usual and `training_paused` is cleared (`train.done`). Errors: `not_found`, `bad_request` (no paused run, or
  the recordings changed since the pause: discard it and train again), `not_ready`, `busy`.
- `train.start {"voice_id","restart":true}` on a voice with a paused run discards it (Applio folder and flag) and starts
  from scratch; without `restart` it answers `paused` ("A training of this voice is paused at epoch k of N. Resume it,
  or discard it to start over.").
- `train.cancel {"voice_id"}` (instead of `{"job"}`) discards a paused run: deletes the Applio folder and clears
  `training_paused` -> `{}`. `busy` when that voice is training right now. Cancelling a running job always discards
  its checkpoints, too. `voices.delete` removes the Applio folder as well.
- **Interrupted runs.** While a training runs, `voice.json` holds an internal `"training_run":{...}` marker (model
  name, total epochs, batch size, newest confirmed checkpoint epoch / step, steps per epoch). If the engine goes away
  without a pause (app closed with "Close anyway", crash, killed process), the next engine start finds the marker
  before it answers `ready`, stops Applio workers that survived (the PIDs are in Applio's `config.json`), repairs the
  checkpoints and, when a valid one exists, turns the run into `"training_paused":{...,"interrupted":true}` with the
  epoch of the newest checkpoint (at most 4 epochs lost); `voices.list` then shows it and `train.resume` works as
  after a pause. Applio folders are **never** cleaned at engine start (only `<data>/tmp` is). A run that fails while
  learning (for example out of memory) keeps its latest checkpoint the same way (`train.error` message says
  "Progress up to epoch k was kept"), and a graceful engine shutdown stops a running Applio.
- **App side (not part of the wire protocol).** Closing the window while a training runs shows a dialog (Rune cannot veto
  a close, so the window hides and is reopened at once): "Pause and close" sends `train.pause` and quits on
  `train.paused`; "Keep training" dismisses it; "Close anyway" shuts the engine down (the engine stops Applio) and quits,
  and the run is found again as an interrupted one at the next start. The Train tab shows `pause k/N` while a run is
  paused, the Voices card "Paused k/N" (or "Interrupted k/N").
- `voice.json` keys written by training: `training_run` (internal, only while running), `training_paused`
  (`epoch,total_epochs,model_name,paused_at,interrupted?` plus internals `steps_per_epoch,checkpoint_step,batch_size,
  cache_in_gpu,dataset_sig`).

### `tts.generate` / `tts.cancel`
`{"voice_id","text","expressiveness":0.8,"voice_match":1.0,"quality":"best","speed":1.0,"language"?:"en"}` ->
`{"job","segments":3,"segment_chars":[64,150,92]}`; `tts.cancel {"job"}` -> `{}`
- `expressiveness` (0.1-1.5, default 0.8) maps to OmniVoice's sampling: `position_temperature = 5.0 * e / 0.8` (how
  freely the order of the unmasked tokens, i.e. phrasing and pauses, varies; 5.0 is the model default) and
  `class_temperature = max(0, e - 0.8)` (token sampling; 0 = greedy, the default). 0.8 gives the model's defaults, lower
  is flatter and steadier, higher varies more. `voice_match` (0-1, default 1.0) is the RVC index rate in the language the
  voice was recorded in (`voice.json` `language`, default `en`) and is capped at 0.3 in any other language (see
  [Speech model and languages](#speech-model-and-languages-omnivoice)); it is ignored when the voice has no RVC model
  (zero-shot output is returned as is), `0` skips RVC. Out-of-range numbers are clamped, non-numbers are `bad_request`.
- `speed` (0.8-1.3, default 1.0, clamped like the other numbers): speaking speed, OmniVoice's `speed` control (the
  model sizes the speech to `1 / speed` of its natural length; pitch unchanged, RVC afterwards as usual). Every segment
  uses the same value. The tempo copied from `reference.wav` is the baseline, so `speed` is relative to that clip (see
  `voices.reference_candidates`).
- `quality`: `best` (default, 32 OmniVoice decoding steps) or `fast` (16 steps, about twice as fast per segment,
  slightly less clean). The first segment always uses 16 steps for a quicker first sound.
- `language` (optional, default `"en"`): the language of `text`, an ISO 639-1 code from the list in
  [Speech model and languages](#speech-model-and-languages-omnivoice); another code is `bad_request`. It selects
  OmniVoice's language and the text normalization (numbers, units, abbreviations spelled out in that language).
  **App side:** the Speak "Style" panel has a language chip (searchable picker, "Common" then "All languages");
  the default is the active voice's `language`, a different choice is remembered per voice in `settings.json`
  (`speak_languages: {voice_id: code}`); a language the voice was not recorded in shows a hint and "Record a
  <language> reference".
- Errors: `bad_request` (empty text, bad `quality`, unknown `language`), `not_found` (voice, or it has no `reference.wav`), `not_ready`
  (models loading, missing, or handed over to training), `busy` (another generation is running; only one at a time).
- The text is cut into `segments` pieces: paragraphs are never merged, the first segment is the first sentence only
  (so audio starts fast), later ones are packed up to about 2.5 times the previous size (at most 70 tokens), and a
  paragraph never ends in a tiny segment. `segment_chars` lets a client estimate the remaining speech per segment.
- Defaults: OmniVoice fp16, 32 steps (16 for `fast` and segment 0), guidance scale 2.0, reference cut to 7-12 s of
  whole sentences; RVC rmvpe, protect 0.33, pitch 0.
- Events, in order: `tts.chunk {"index","path","duration_s","segments","ready_s","rtf"}` (one per segment, final audio
  at 44.1 kHz, already includes the pause after it, play at once), then `tts.done {"path_wav","path_mp3","duration_s",
  "elapsed_s"}`, or `tts.error {"code","message"}` (`cancelled` after `tts.cancel`; the slot is free again when the
  event arrives). `ready_s` is the wall time from the start of the job to this chunk; `rtf` is the wall time since the
  previous chunk (or the start) divided by this chunk's speech seconds, so `rtf < 1` means generation is faster than
  playback. Chunk files live in `<data>/tmp/<job>/` and are deleted after an hour.
- **Pre-buffer rule (client).** Chunk 0 is played as soon as it arrives and later chunks are queued gaplessly behind
  it; the segment growth schedule keeps `rtf` of the later chunks near 1 (measured 1.05-1.4). A client that wants to
  avoid underruns on slower hardware estimates the total speech `S = duration_s(chunk 0) * sum(segment_chars) /
  segment_chars[0]` and holds chunk 0 back for `max(0, rtf * (S - duration_s(chunk 0)) - S)` seconds, taking `rtf`
  from chunk 0 (a first estimate) and refining it with each new chunk. With `rtf <= 1` this is 0: start at once.

### Speech model and languages (OmniVoice)

Every segment is spoken by OmniVoice (k2-fsa, 0.6B, CC BY-NC 4.0 weights), run by a persistent worker process with the
engine's interpreter (`backends/omnivoice_worker.py`, wire format `worker_kit.py`, ops `load`, `synth`, `warm`, `release`,
`status`). Its state is the `tts` entry of the `models` event. IndexTTS-2.5 is no longer used. The rest of the pipeline
(RVC, resampling, pauses, slot fitting, events) does not depend on the model (`backends/__init__.py`).

Languages (`language` of `tts.generate`, Redub's language guard; ISO 639-1, every language with at least ~100 h of
OmniVoice training data): `en zh ja es fr de ru pt yue th it ko vi id no ca hr lt sk sv el nl bg sr uk be da kk lv sl ar
et pl bs is mt eu fi ta sw fa bn as mn hu ba te ur gl ml ka mr cs pa cy kn tr hi uz ne`.

- **Text**: before synthesis the worker spells out numbers, symbols and common abbreviations
  (`backends/omnivoice_text.py`): for Ukrainian with case after prepositions and agreement of units
  (`у 2026 році` -> `у дві тисячі двадцять шостому році`, `5 травня` -> `п'ятого травня`, `15%` -> `п'ятнадцять відсотків`,
  `3,5 години` -> `три з половиною години`, `1 250 грн`, `т.д.`, `вул.`, `о 10:30`), for English years, ordinals,
  money, time and decimals, for other languages integers and decimals through `num2words`. Stress marks (combining
  acute, `+`) are removed: OmniVoice places Ukrainian stress itself.
- **Reference**: `reference.wav` used whole up to 12 s, longer clips are cut to 7-12 s of whole sentences (statements
  preferred over questions; Whisper turbo finds the sentences once per clip, cached in `reference.omnivoice.json`).
  `reference_<lang>.wav`, when present, is used for that language instead: OmniVoice carries the accent of the reference
  into the target language.
- **RVC in other languages**: the voice's RVC index only holds sounds of the language it was recorded in
  (`voice.json` `language`, default `en`). For speech in another language the index rate is `min(voice_match, 0.3)`
  (`rvc.index_rate_for`; `tts.generate`, Redub respeak and dub); index 1.0 garbled Ukrainian words (Whisper WER 25 %
  instead of 6 %), 0.3 keeps them and most of the timbre. `MIMVOX_RVC_FOREIGN_INDEX` overrides the 0.3.
- **Training hand-over**: the worker process is ended (all its VRAM and RAM back) and started again after the training.

### `shutdown`
`{}` -> `{}` then exit.

## Audio formats (import and export)

One shared layer (`engine/src/mimvox_engine/media.py`) reads every audio input of the engine (`recording.check`,
`dataset.add`, `voices.import`, `redub.*`) and writes every export. WAV, FLAC and AIFF are read natively; everything else
(and anything the native reader refuses) is decoded by **ffmpeg** to mono float samples (the first audio track; video
files give their audio). A file without an audio track is `bad_request` ("This file has no audio track."), an unreadable
file `bad_request`, a missing file `not_found`.

**ffmpeg location** (first hit wins): the `MIMVOX_FFMPEG` environment variable (a folder or the ffmpeg file), the Applio
runtime root (`paths.applio` in runtime.toml: on Windows Applio's prerequisites put `ffmpeg.exe` and `ffprobe.exe`
there), then `PATH`. On **Linux** `setup.sh` installs the system `ffmpeg` package (it is in the package set, which also
provides `ffprobe`); on **macOS** setup runs `brew install ffmpeg` when Homebrew exists, otherwise it stops with that
instruction. Setup's quick check verifies both executables and that `whisper` imports. Without ffmpeg only WAV, FLAC,
OGG and AIFF can be read, and `audio.export` falls back to libsndfile (wav, mp3, flac, ogg, aiff, no quality control);
anything needing ffmpeg answers `not_ready`.

### `formats.list`
(All lists are plain lower-case extension strings without a dot.)
`{}` -> `{"import":["wav","mp3","m4a","aac","flac","ogg","opus","wma","aiff","mp4","mkv","mov","webm","avi"],
"import_video":["mp4","mkv","mov","webm","avi"],"export":["wav","mp3","flac","ogg","opus","m4a","aiff"],
"qualities":["high","standard","small"],"ffmpeg":"ffmpeg version ..."|null}`
- The lists reflect what the found ffmpeg can demux/decode (`import`: lower-case extensions without a dot, `wma` appears
  only when the build has a WMA decoder) and encode (`export`). `import_video` is the subset that are video containers.
  Use it to fill file pickers; the engine ignores the extension when decoding (it probes the content).

### `audio.export`
`{"path","format":"wav|mp3|flac|ogg|opus|m4a|aiff","dest"?:"<absolute file path>","quality"?:"high|standard|small"}` ->
`{"path","size_bytes"}`
- Converts any readable audio file (a `tts.done` / `redub.done` / `tts.chunk` file, a dataset part, an imported file) to
  `format`. `quality` (default `standard`) only matters for lossy formats:

  | format | codec | high / standard / small |
  | --- | --- | --- |
  | `wav` | PCM 16-bit, 44.1 kHz | (lossless, no quality) |
  | `flac` | FLAC 16-bit, source rate | (lossless, no quality) |
  | `aiff` | PCM 16-bit big-endian, 44.1 kHz | (lossless, no quality) |
  | `mp3` | LAME, 44.1 kHz | 320 / 192 / 128 kbps |
  | `ogg` | Vorbis, source rate | quality 7 / 5 / 3 (about 224 / 160 / 112 kbps) |
  | `opus` (`.opus`, Ogg container) | Opus, 48 kHz | 128 / 96 / 64 kbps |
  | `m4a` | AAC-LC in MP4, 44.1 kHz | 256 / 192 / 128 kbps |

- Default `dest`: next to the source with the new extension (`.../20261111_101112.wav` -> `.../20261111_101112.ogg`; when
  that is the source itself, `<name>_export.<ext>`). An existing `dest` is replaced (written to a temporary file first).
  The file is mono or stereo as the source is; tags/cover art are dropped.
- Errors: `bad_request` (unknown `format` / `quality`, relative or folder `dest`, `dest` equal to the source, unreadable
  source), `not_found`, `not_ready` (ffmpeg or the needed encoder missing).

## Redub (transcribe, convert and re-speak a file)

The user drops an audio or video file. It is either **converted** (the same recording, timing untouched, in a voice) or
**re-spoken** (transcribed, optionally edited by the user, then spoken again in a voice, each sentence at its original
time). Everything is `mono 44.1 kHz`; the result is audio only (video is not muxed back in).

All Redub jobs that use the GPU take the same single slot as `tts.generate`: while one runs, `tts.generate`, another
Redub job and `train.start` answer `busy`; during a training (models `loading`) every Redub job answers `not_ready`.
`redub.cancel` works for all of them.

Conventions: `redub.transcribe`, `redub.dub` and `redub.render` answer `{"job":"job-N"}`; every `redub.progress.pct` is a
number from 0 to 100 (overall for that job, never decreasing); the terminal events are `redub.transcript` / `redub.done`
(success) or `redub.error` (failure or cancel). `recording.check`, `dataset.add` and `voices.import` accept non-WAV input (any
importable format, also video) and convert it themselves: the app never has to.

### `redub.probe`
`{"path"}` -> `{"duration_s":69.68,"sample_rate":44100,"channels":1,"has_audio":true,"format":"wav",
"preview_path":"<data>/tmp/redub_preview_<id>.wav"}`
- Any file ffprobe understands (see [Audio formats](#audio-formats-import-and-export)). `format` is the file's
  extension-style name (`wav`, `mp3`, `m4a`, `mp4`, ...). `has_audio: false` (and `sample_rate`/`channels` 0) for e.g. a
  silent video; `duration_s` is then the container's. A container without a duration header (some webm) is measured by
  decoding it. `preview_path` is the absolute path of the source decoded to a
  mono 16-bit **44.1 kHz WAV** in `<data>/tmp` (`null` when there is no audio track): the app plays only WAV, so use it for
  "play original" and A/B against the result. It is cached per file state and deleted at the next engine start. The call
  answers directly (no job) but decodes the file, so it takes about 1 s per 10 minutes of audio.
- Errors: `not_found`, `bad_request` (not a readable media file), `not_ready` (no ffprobe).

### `redub.prepare`
`{"path"}` -> `{"warming":true}` or `{"warming":false,"reason":"unavailable|paused|busy"}`
- Call it when the user drops a file. It starts the **speech worker** (see below) and loads Whisper turbo in the
  background so the following `redub.transcribe` / `redub.dub` starts at once; it answers immediately and later emits
  `redub.prepared`. A no-op when Whisper is already loaded. While the speech model is still loading at engine start it waits
  (up to 3 minutes) instead of competing for the GPU. A job that arrives meanwhile simply queues behind the load.
  `redub.prepared` always follows a `warming:true` answer: `ok:false` also when a training took the GPU meanwhile, or
  when a GPU job (`tts.generate`, `redub.render`) asked the preload to stop.
- Errors: `not_found` (file), `bad_request`.

### `redub.transcribe` / `redub.cancel`
`{"path","language":"auto","translate_to":null}` -> `{"job"}`; `redub.cancel {"job"}` -> `{}`
- `language`: `"auto"` (Whisper detects it from the first 30 s) or a two/three letter code (`"en"`, `"uk"`, `"de"`, ...).
  Anything else is `bad_request`. The speech is always transcribed **in its own language** (Whisper turbo). `translate_to`:
  a language code additionally translates every segment to that language (skipped when the speech already is in it),
  `null` does not translate. Any language the speech model speaks is a target (all of them are NLLB languages; see
  [Speech model and languages](#speech-model-and-languages-omnivoice)); another code is `bad_request` (the message lists
  the targets). `translate_to_en: true` is the old spelling of `translate_to: "en"` and still accepted. (The earlier Whisper `medium` translate mode is gone: it dropped and paraphrased speech.)
- **Speech worker.** Transcription and translation run in one persistent child process of the engine
  (`speech_worker.py`, engine interpreter, JSON lines on stdin/stdout like the RVC worker), started lazily by the first
  of `redub.prepare` / `redub.transcribe` / `redub.dub`, restarted automatically (and the request retried once) when it
  dies, one op at a time (requests queue in the engine's owner task for the worker; a `redub.prepare` preload that is
  still running when a GPU job starts is asked to stop first). Models load on demand and stay warm: they are dropped after **3 minutes idle**, and the
  process exits after **10 minutes idle**. It never hogs the GPU: before speech-model / RVC work starts (`tts.generate`,
  `redub.render`, the speak stage of `redub.dub`) the engine tells it to release its GPU memory (the models are dropped
  and load again from the disk cache in a few seconds, no download), and a training stops it altogether. Whisper turbo
  runs on the GPU with fp16 weights (about 1.6 GB) when about 2.6 GB of VRAM are free (OmniVoice and RVC
  hold about 7 GB of a 12 GB card); NLLB (fp16, 1.2 GB, needs 2 GB) is never loaded next to Whisper unless both fit,
  otherwise Whisper is dropped from the GPU first; with too little VRAM, or without CUDA, the model runs on the CPU
  (correct, slower; at most one model is then resident). Expect the worker at about 2-4 GB of host RAM while a model is
  loaded (peak while loading the 2.5 GB NLLB checkpoint), about 1-1.5 GB with no model, plus a CUDA context (about
  0.4 GB VRAM) while the process lives.
- **Translation.** NLLB-200 distilled 600M (`facebook/nllb-200-distilled-600M`, via transformers, local files only) in
  `<data>/runtime/models/nllb-200-distilled-600M/`; if the files are missing the call answers `not_ready` ("The
  translation model is not installed ... Run setup"). It translates **per segment** (the segment count and order never
  change; a text longer than 160 characters is translated sentence by sentence and re-joined; a segment the model
  returns empty keeps its source text), batched, beam search 4, no length penalty, no profanity filter. The source
  language is Whisper's code (or `language`) mapped to NLLB: `uk`->`ukr_Cyrl`, `ru`->`rus_Cyrl`, `de`->`deu_Latn`,
  `fr`->`fra_Latn`, `es`->`spa_Latn`, `pl`->`pol_Latn`, `it`->`ita_Latn`, `pt`->`por_Latn`, `en`->`eng_Latn`, `zh`->`zho_Hans`,
  `ja`->`jpn_Jpan`, `ar`->`arb_Arab`, ... (about 95 languages; a language NLLB lacks gives `bad_request`); the target
  `translate_to` is mapped the same way (`en`->`eng_Latn`, `uk`->`ukr_Cyrl`, `ne`->`npi_Deva`, `fa`->`pes_Arab`,
  `yue`->`yue_Hant`, ...; table in `redub/langs.py`).
- Models: Whisper turbo (1.6 GB) and NLLB (2.5 GB) are downloaded by setup; a missing Whisper checkpoint is still
  fetched on first use into `<data>/runtime/models/whisper/` (never the user's home cache), with `load_model` progress.
  Segment timestamps only, no word timestamps. A transcript is not saved by the engine: the app keeps it.
- Events: `redub.progress {"stage":"decode|load_model|transcribe|translate","pct":0-100}` (`pct` is **overall**,
  monotonic. Without translation: decode 0-5, load_model 5-30, transcribe 30-100. With `translate_to`: decode 0-5,
  load_model 5-25, transcribe 25-80, translate 80-100; the NLLB load is part of `translate`; a warm model skips most of
  `load_model`), then
  `redub.transcript {"language":"uk","source_language":"uk","duration_s":44.1,"segments":[{"index":0,"start":0.0,"end":6.4,"text":"..."}],
  "model":"turbo","translated":false}` - with a translation `"translated":true,"target_language":"en",
  "translation_model":"nllb-200-distilled-600M"` and each segment also has `"source_text"` (`text` is then the
  translation into `target_language`).
  `language` / `source_language` is the detected (or given) language of the speech, also when translated. `segments` may
  be empty for a file without speech; times in seconds, 2 decimals. Or `redub.error {"code","message"}` (`cancelled`,
  `bad_request` for a file with no usable audio or an untranslatable language, `not_ready`, `internal`).
- Errors at start: `bad_request`, `not_found` (file), `not_ready` (models loading / training running, `openai-whisper`
  missing, or - with `translate_to` - the translation model or `transformers` / `sentencepiece` missing), `busy`.
- Timing (RTX 4070, 44 s Ukrainian clip, 11 segments): see "Measured" at the end of this section.

### `redub.dub` (the whole pipeline in one job)
`{"path","voice_id","source_language":"auto","target_language":"en","review":false,"voice_match":1.0,"expressiveness":0.8,
"quality":"best","speed":1.0}` -> `{"job"}`
- Decode -> transcribe in the source language -> translate every segment to `target_language` -> **speak** in that
  language (the same respeak render as `redub.render`, see below, with the translated segments at their original times)
  -> save. One job, one GPU slot for the whole run; the speech worker releases the GPU before the speak stage starts.
- `source_language`: `"auto"` (detected) or a code; `target_language` (default `"en"`): any language the speech model
  speaks (as `translate_to` of `redub.transcribe`; another code is `bad_request`). A source that already is in the target
  language is not translated. The voice's RVC index rate is capped for a target the voice was not recorded in (as for
  `tts.generate`), and `reference_<target>.wav` is used when the voice has one.
- `review: true`: **stop after the translation** and let the user edit it. The job ends with the normal terminal event
  `redub.transcript` (same shape as for `redub.transcribe` with `translate_to`: `segments` with translated `text` and
  `source_text`, `translated:true`, `source_language`, `target_language`, plus `"review":true` and `"timings_s"`); no voice
  model is needed for this and nothing is spoken. The app then calls **`redub.render`** with `mode:"respeak"`, `voice_id`,
  the (edited) segments' `{start,end,text}`, `"translated":true`, `"target_language"` and the user's sliders, exactly as
  after `redub.transcribe`.
  (No separate `redub.review` event exists: the transcript event is the review.)
- `review: false` (default): continues straight into speaking and ends with `redub.done {"path_wav","path_mp3",
  "duration_s","elapsed_s","mode":"dub","language":"uk","target_language":"en","translated":true,"segments":[...as in the transcript...],
  "stretched","overrun","pushed","drift_s","max_push_s","timings_s":{"decode":..,"load_model":..,"transcribe":..,
  "translate":..,"speak":..,"save":..}}` (seconds per stage).
- Events: `redub.progress {"stage":"decode|load_model|transcribe|translate|speak|save","pct":0-100,"segment"?,"segments"?}`
  with one **monotonic** overall `pct` across the whole pipeline. `review:false`: decode 0-3, load_model 3-8, transcribe
  8-24, translate 24-30, speak 30-95 (`segment` / `segments` as in `redub.render`), save 95-100. `review:true`: decode 0-5,
  load_model 5-25, transcribe 25-80, translate 80-100.
- Errors at start: `bad_request` (parameters, `target_language`), `not_found` (file, voice, `reference.wav`), `not_ready`
  (speech model / RVC not ready for `review:false`; Whisper, translation model missing), `busy`. In the job: `redub.error`
  with `bad_request` ("No speech was found in this file."), `cancelled`, `not_ready`, `internal`.

### `redub.render`
`{"voice_id","path","mode":"convert|respeak","segments"?:[{"start","end","text"}],"language"?:"uk","translated"?:false,
"voice_match":1.0,"expressiveness":0.8,"quality":"best","speed":1.0}` -> `{"job"}`
- `convert`: the voice's RVC model is run over the **whole source** (voice, noise and music alike; background/music
  separation is not implemented) in chunks of about 15-30 s cut at the quietest point (a pause), then joined. The result
  has exactly the source's length, so the original timing is kept. Silent stretches are copied. The voice must be
  trained: otherwise `bad_request` ("This voice has no trained voice-conversion model ..."). Uses `voice_match` (RVC
  index rate, 0-1, `0` = no index); `expressiveness` and `segments` are ignored.
- `respeak`: needs `segments` (`bad_request` when missing or empty; `text` is whitespace-normalised, an emptied segment
  stays silent, unsorted input is sorted by `start`; at most 5000). Each segment is spoken with the normal speech
  pipeline (OmniVoice, then RVC when the voice is trained and `voice_match > 0`; an untrained voice answers zero-shot
  output) and placed at its `start` on a silent track of the source duration. Fitting a clip into its slot (the time from
  its start to the next segment's start, minus a 0.12 s gap; the last segment's slot ends at the source end):
  1. aim at the segment's own length (`end - start`, but never more than the slot): the speech length factor is set
     to aim / (estimated natural length), the estimate calibrating itself from the clips already spoken, clamped to
     0.75-1.25; the user's `speed` multiplies it by `1 / speed` (a slower voice gets longer clips), and when that product no
     longer fits the slot it is lowered to fit (the slot wins), the combined factor staying within 0.65-1.4;
  2. if the clip is still longer than the slot, it is time-stretched (librosa, pitch kept) by at most 1.25x;
  3. if it is still longer, it overruns into the following silence. Two clips never overlap: a clip that would start before
     the previous one ended is pushed later (`pushed`), and the next slot shrinks accordingly. The output is longer than
     the source only when the last clip overruns the end.
  Leading/trailing silence of each clip is trimmed, so speech starts at the segment's `start`.
  This is also how translated text fits (English is often shorter or longer than the original): `duration_factor` 0.8-1.25
  first (x the user's `speed`, slot wins), then a librosa stretch of at most **1.2x**, then the clip overruns and the
  following segments are pushed later; speech is never cut and clips never overlap. The result reports `drift_s` (how much
  longer the output is than the source, 0 when it ends with the source) and `max_push_s` (the largest delay of any
  segment's start).
- **Language guard.** The guard accepts every language the speech model speaks (see
  [Speech model and languages](#speech-model-and-languages-omnivoice); the message lists them). `language` is the transcript's language and
  `translated: true` says the segment texts are translations into `target_language` (default `"en"`; what
  `redub.transcribe` with `translate_to` and `redub.dub` with `review:true` return; they are then spoken in
  `target_language` whatever `language` says). With `translated` false/absent and a `language` outside that list the call
  answers `bad_request` ("The voice cannot speak ... yet ... Translate the text first"); an absent `language` is guessed
  from the script (Cyrillic with
  і/ї/є/ґ is Ukrainian, other Cyrillic Russian, Latin text English). A supported `language` (Ukrainian included) is
  spoken in that language.
- `voice_match` (0-1, default 1.0), `expressiveness` (0.1-1.5, default 0.8), `quality` (`best` | `fast`) as in
  `tts.generate` (out-of-range numbers are clamped, non-numbers `bad_request`). `expressiveness`, `quality` and
  `speed` (0.8-1.3, default 1.0, see `tts.generate`) apply to `respeak` only; `convert` keeps the original pace.
- Events: `redub.progress {"stage":"decode|convert|speak|save","pct":0-100,"segment"?:i,"segments"?:n}` (`pct` overall:
  decode 0-5, convert/speak 5-95, save 95-100; `segment` is the 0-based index of the chunk (`convert`) or transcript
  segment (`speak`) that just finished, `segments` the total; `decode` and `save` carry neither), then
  `redub.done {"path_wav","path_mp3","duration_s","elapsed_s","mode"}` plus, for `respeak`, `"language"` (the language
  spoken) and `"stretched":n,"overrun":n,"pushed":n,"drift_s":s,"max_push_s":s` (informational), or `redub.error {"code","message"}` (`cancelled` after
  `redub.cancel`; `bad_request` for a file without audio; `internal`). The files are
  `<data>/outputs/redub_<YYYYMMDD_HHMMSS>.wav|.mp3` (44.1 kHz mono; `_2`, `_3` on a clash). The job slot is free again when
  the end event arrives. Export to other formats with `audio.export`.
- Errors at start: `bad_request` (mode, quality, numbers, untrained voice for `convert`, bad `segments`), `not_found`
  (voice, file, or no `reference.wav` for `respeak`), `not_ready` (the needed models are not `ready`), `busy`.
- Typical flows: `redub.probe` (show duration) + `redub.prepare` (on drop) ->
  1. **couple of clicks**: `redub.dub` (`review:false`) -> `redub.done`;
  2. **with review**: `redub.dub` (`review:true`) or `redub.transcribe` (`translate_to`) -> `redub.transcript` -> the user
     edits the translation -> `redub.render` (`mode:"respeak"`, `translated:true`, `target_language`);
  3. same-language re-speak: `redub.transcribe` -> edit -> `redub.render` (`mode:"respeak"`, `language` from the transcript;
     any language the speech model speaks);
  4. `redub.render` with `mode: "convert"` straight away.
- **App side.** Translate & dub has "From" (Auto-detect or a language) and "To" (default English) language pickers with
  the engine's `languages`; Re-speak has "Spoken language" and "Translate the transcript" with "To". The choices are
  remembered in `settings.json` (`redub_language`, `redub_target`). Badges read "Ukrainian · from English"; a render of a
  translated text sends `translated:true` and `target_language`, a same-language one `language`.
