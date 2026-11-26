# Changelog

## Unreleased

* **Linux:** the window no longer has a second, grey title bar above its own on GNOME (Wayland): it draws its own
  minimize, maximize and close buttons as on Windows, moves by its header, resizes by its edges and corners, and
  rounds its corners. Tiling window managers (i3, sway, Hyprland and others) keep their own frame.
* **AppImage:** at each start it adds itself to the list of apps with its icon, keeps that entry pointing at the file
  after an update renames it, and shows its icon on the file in the file manager. The dock shows Mimvox's icon for
  the window (the AppImage and the .deb alike).

## 0.6.0

* **More voice-cloning models:** two optional parts add other models beside OmniVoice: **VoxCPM2** (6 GB, 30
  languages), which often stays closer to your voice in another language, and **Chatterbox** (4.3 GB, 23 languages),
  very close to your voice in English. Install them in **Settings** > **Modules** (or with setup's `voxcpm` and
  `chatterbox` components); Speak's Style panel then shows **Model** (*Auto*, *VoxCPM2*, *Chatterbox*), and Book uses
  the same choice.
  * Only the models that speak the chosen language are offered. *Auto* is OmniVoice; with one of the new models
    installed it also speaks Hebrew, Malay, Tagalog, Burmese, Khmer and Lao.
  * They take longer than OmniVoice (VoxCPM2 about 7 times, Chatterbox about 3 times), and the first sentence waits
    10-20 s while the model loads. Only one of them is kept in memory at a time.
* **For programs:** `tts.generate` and `book.start` take `model`, `hello` lists `speech_models`, and every result
  says which model spoke. `/v1/audio/speech` takes `voxcpm` or `chatterbox` as `model` (OpenAI's `tts-1` and others
  mean *Auto*), and the MCP tools `speak` and `start_book` take `model` too.
* **Newer speech engine:** the engine and Ukrainian speech now run on Python 3.14 and PyTorch 2.11 (AMD GPUs: ROCm
  7.2), and the models start faster (9.5 s instead of 15 s on an RTX 4070). After the update a banner asks to run
  setup again: it makes the engine's environments again (a few GB of packages; the models stay) and keeps the old
  ones if that fails.
* **Moving the data folder on Windows** (**Settings** > **Data folder** > **Move data…**, or `setup --move-to`) no
  longer breaks the speech engine: the engine's Python still pointed to the old folder after the move.
* **Fixes:**
  * Installing Ukrainian speech failed with *Connection refused*: its packages now come from GitHub itself instead of
    a mirror that stopped answering.
  * The engine no longer logs an error about the native accent's converter at every shutdown.
  * The Windows-only message *no memory left to back it* no longer shows on Linux.

## 0.5.0

* **Local API:** other programs can now use Mimvox over HTTP. Turn it on in **Settings** > **API**; it is off by
  default and stops with the app.
  * **OpenAI-compatible speech:** `POST /v1/audio/speech` at `http://127.0.0.1:8765/v1` speaks text in one of your
    voices (by id or name) as MP3, WAV, FLAC, Opus, AAC or 24 kHz PCM, so tools made for OpenAI's speech API work
    with your own voice.
  * **Everything else too:** `POST /rpc` runs any engine method (voices, Speak, books, Redub, training),
    `GET /events` streams the engine's progress events, and `/files` uploads files and downloads results.
  * **Interactive docs** at `http://127.0.0.1:8765/docs` list every endpoint and engine method with examples and let
    you try them; `/openapi.json` is the same description for other tools. They work offline.
  * **Token and network:** programs on this computer get in without a token unless **Require token** is on; web pages
    never do. **Local network** makes the API reachable from other computers and always needs the token. The
    token is shown, copied or replaced in Settings.
  * The API uses the app's own engine, so the app and the API share one engine and wait for each other.
* **Without the window:** `mimvox-server`, installed next to the app, runs the engine and the API with no window, for
  a server or a script; `mimvox serve` starts it too (also from the AppImage). `mimvox-server token show` prints the
  token. Only one engine runs per data folder: the app and the server cannot use the same one at once.
* **MCP server for AI assistants:** `mimvox-mcp` (or `mimvox mcp`) lets Claude Desktop, Claude Code and other
  Model Context Protocol clients use Mimvox through the API: list voices, speak text and get the audio back, read a
  document as an audiobook, pause and resume it, follow long jobs with progress, upload and download files, and call
  any engine method. The usage guide has ready-made configs for each system. `mimvox-server` and `mimvox-mcp` have
  icons of their own.
* **For programs that publish audiobooks (through the API):**
  * Each finished chapter gets a timing file with the text and start and end time of every sentence group, for
    subtitles (`segments_path`, served by `GET /files`).
  * Importing a book also returns its authors, description, language tag, date and cover (EPUB, FB2, DOCX, ODT).
  * `text.import` with `"split": "auto"` cuts very long chapters into parts at once.
  * `book.start` with a `ref` is safe to repeat: the same ref finds the book already started, resumes it or reports
    it done, instead of starting it twice.
  * A voice can be exported as one `.mimvoice` file with a content hash (`voices.export`) and imported on another
    computer (`voices.import` with `bundle`).
* **Settings:** the status button at the top right (*Ready*, with a small sliders icon) now opens **Settings**, with
  three folding sections: **Modules** (what was the Modules window), **Data folder** (the folder, its free space and
  **Move data…**, which used to be at the bottom of Voices) and **API**. The engine's state is at the top of
  Settings, with the reason when speech cannot run and **Retry** when the engine has stopped. Settings opens on the
  section you left open, also after a restart, or on **Modules** when models are missing or the current screen's part
  is turned off.
* **Import an existing RVC model:** **Import voice model…** on Voices turns an RVC or Applio model (`.pth`, with an
  optional `.index`) and a short clip of the same voice into a trained voice, with no training. Only your own voice
  or one you have permission for: the dialog asks you to confirm. A file that is not such a model (a `.zip`, a
  training checkpoint) is refused with the reason; it is checked without running anything in it.
* **Undo and redo in Speak:** two buttons next to the trash button, also Ctrl+Z and Ctrl+Shift+Z (Cmd on macOS).
  Clearing the text and opening a text file can be undone too.
* **A steady player:** the knob moves smoothly and no longer stops or jumps back while the computer is busy
  generating. While Speak is still generating, the bar is a loading bar with the part made so far and the total shows
  *…*; the knob and seeking come when the whole text is ready.
* **Smoother tab switches:** a tab switch now animates for about 0.3 s instead of 0.7 s, and on displays of 120 Hz and
  more animations draw at about 60 frames a second, so a click no longer drives the graphics card at the display's
  full rate.
* **Speech on the processor (no supported GPU):**
  * Slow speech is no longer cut off after 5 minutes: as long as the model keeps working, it is given the time it
    needs.
  * Mimvox checks the memory before each step. A computer with less than 11 GB of RAM cannot hold the speech model:
    the model is not loaded at all, the status button says *Not enough memory* and Speak explains why, instead of
    swapping for minutes per sentence. When the memory is only full at the moment, speech stops with a plain message
    and frees the model, and a book waits and goes on. Programs get the error code `no_memory` for both.
  * Whisper takes 3.8 GB instead of 5.5 GB, and the speech model is no longer pushed out of RAM right after it loads,
    which made the first sentences slower.
* **Setup:**
  * An install made with `--cpu` stays on the processor in later runs, including the app's own; `--gpu` switches to
    the graphics card.
  * The check (`--quick-check`, also after `--move-to`) looks only at the parts you installed, so a speech-only install
    passes.
  * The Pythons the engine runs on now live in the data folder (`runtime/python`), so deleting the data folder, or
    uninstalling the speech engine, removes them too. Setup moves an existing install's environments onto them, from
    the packages it already has, and keeps the old ones if that fails.
* **Windows installer:** an update no longer fails while an AI assistant keeps `mimvox-mcp` running (it is closed
  first), and waits for a running `mimvox-server` like it does for the app. Uninstalling with **Everything** now also
  removes the API's files and the last few files the app keeps in the data folder.
* **Fixes:**
  * A book chapter whose last part was made right before a pause or a crash is finished on resume; it used to be left
    out of the book.
  * A book with a plain-text table of contents right before chapter 1 keeps chapter 1, a short sentence that repeats
    a chapter's title is no longer dropped as a heading, and Belarusian books are no longer taken for Russian ones.
  * The engine says *Ready* only once the speech model is warmed up, so a first request no longer waits behind the
    warm-up (there is none on the processor).
  * The switches in Settings have their knob in the middle.

## 0.4.3

* **Updates download on a slow connection:** the download of a new version was cut off after 30 seconds, so below
  about 7 Mbit/s an update could never finish. It now has up to 30 minutes.
* **Settings survive a crash:** settings are written to a temporary file and then swapped in, so a crash while saving
  no longer resets every setting.
* **Setup:**
  * Moving the runtime with `--move-to` no longer fails its check when only some parts are installed.
  * The check accepts the ffmpeg setup puts in the runtime folder, so an install without training no longer reports
    *ffmpeg / ffprobe not found*.
* **Engine:** a request line over 64 MiB no longer hangs the engine (it shuts down and is started again), and a number
  of more than 4300 digits in a text is left as digits instead of failing the speech.

Thanks to @shiwng for all of these (#53, #54, #55).

## 0.4.2

* **Modules:** the status pill at the top (*Ready*) opens a window with every optional part, its size and state.
  **Install** adds one, **Turn off** keeps its files but stops using it, **Turn on** brings it back, and **Remove**
  deletes it to free the disk. `setup --remove` does the same from a terminal.
* **Ukrainian with exact stress keeps the voice:** HolosTTS on its own kept little of a voice's timbre (speaker
  similarity 0.76 to the reference, against 0.94 for the main speech model). In Speak a voice that is not trained now
  goes through the native-accent converter after it, which brings it to 0.88; it is then about as fast as the main
  model. **Exact stress (HolosTTS)** in Style turns it off, to compare or to keep the main model's voice and guess the
  stress.
* **Books stay fast:** a Ukrainian book keeps HolosTTS at about 20× real time by default; **Closer voice with exact
  stress** under Output chooses the converter for a book, about four times slower.
* **Speak keeps your text:** what you typed is still there after a restart, and a trash button next to **Open text
  file…** clears it.
* **Fixes:**
  * Setup started from the app stops the engine first. On Windows a running engine keeps its files open, so a setup
    that had to update the engine's environment (for example **Install** of a part) failed halfway and could leave it
    broken (*Models missing*). The engine starts again when setup is done.
  * When Windows runs out of memory, CUDA reports "out of GPU memory" although the card has room; the message now says
    it is the RAM that has to be freed.

## 0.4.1

* **macOS: the microphone works.** The app never told macOS it uses the microphone, so macOS did not ask for access,
  left Mimvox out of Privacy & Security > Microphone and handed it silence: recordings came out empty and the input
  level never moved, whichever microphone was picked. The app now asks for access the first time it listens, and the
  build signs it (ad hoc when no Apple identity is set) so macOS honours the request. Signed builds also carry the
  microphone entitlement the hardened runtime needs.
* **Book languages:** a Russian, Belarusian or Bulgarian book is no longer taken for a Ukrainian one when its
  language is detected; ї є ґ, ы э ё, ў and ъ tell them apart.

## 0.4.0

* **Change a book's voice:** a book started with the wrong voice no longer has to be imported again. **Change voice**
  in the bar of a paused or finished book picks another one: read every chapter again with it, or keep the chapters
  made so far and read only the rest with it (a chapter half read starts again, so no chapter mixes two voices). The
  circular arrow on a finished chapter reads just that one again. Either way the book goes on at once, and the old
  files are replaced.
* **Exact Ukrainian stress (new optional part, "Ukrainian speech", 1.5 GB):** Ukrainian in Speak, Book and Redub is
  read from phonemes in which the stress is written, so every word is spoken with the stress Mimvox chose, not a guess
  of the speech model. The stress comes from a Ukrainian stress dictionary, with the grammar of the sentence choosing
  between the readings of a word (ру́ки / руки́), and a small model for words the dictionary lacks; numbers, dates and
  abbreviations are spelled out with their cases first. A **+** before a vowel still overrides one word. It works with
  every voice (only the timbre comes from the reference clip) and is about twenty times faster than real time, but
  reads evenly, like a narrator: Expressiveness and Quality do not apply to it. Install it in setup or with **Install**
  in Speak's Style panel when the language is Ukrainian; without it, Ukrainian is spoken as before. Built on HolosTTS
  (MIT) and patriotyk's ukrainian-word-stress with Stanza.
* **Faster speech:** the speech model's decoding loop was rewritten (same model, steps and sampling): a long text is
  made about 1.4 times faster, and Speak's first sound comes about a third sooner. Measured on an RTX 4070; the
  numbers are in docs/performance.md.
* **Roman numerals are read as words,** in every language and with every speech model: *Розділ XII* is "розділ
  дванадцятий", *Частина II* "частина друга", *Chapter IV* "Chapter four", *Людовік XIV* "Людовік чотирнадцятий",
  *Henry VIII* "Henry the Eighth". Cyrillic letters typed in place of Latin ones (*РОЗДІЛ І*) count too.
* **Split a book automatically:** a file without chapter marks (a whole novella in one two-hour chapter) or with very
  long chapters gets **Split automatically** above its chapter list, and **Undo split** puts the file's chapters back.
  It works in every language: chapter titles that were not marked as headings (*Chapter 4*, *XII*, a line in capitals)
  become chapters; scene breaks (`* * *`) are grouped into parts of about 12 minutes; otherwise even parts of about 12
  minutes are cut between paragraphs where the story moves on, never inside a sentence. Parts are named *Частина 1*,
  *Part 1*... in the book's language.
* **A book draft survives a restart:** an opened file with its chapters as you left them (split, joined, renamed,
  ticked) comes back when you open the app again, until you read it or close it.
* **Each book shows its voice:** the book list names the voice each book was read with (or *voice deleted*), the
  active voice's books first; on start the active voice's unfinished book opens.
* **Listening to a book:** when a chapter ends in the player, the next one plays if it is made (not with loop on).
* **A book keeps going when memory runs short:** a part that fails for lack of RAM or video memory no longer stops the
  book. It waits (20 s, then 1, 3 and 5 minutes), frees what it can and reads on from the same sentence, showing
  *Waiting for free memory…* meanwhile; only a shortage that lasts stops it.
* **Clearer book progress:** the line under the book tells the audio's length from the time left to make it ("2 min of
  about 13 h of audio · ready in about 43 min · 20× faster than real time"), and a finished chapter shows how long it
  took to make. The M4B at the end shows its own percentage and time left (joining a 14-hour book takes a few
  minutes), and the book list shows the open book's live state.
* **A chapter's heading is not read twice:** with **Read each chapter's title first** on, a chapter whose text already
  opens with its heading (often after the book's title and author) is read as written.
* **Refresh from disk:** a round ↻ button on Speak's and Redub's Results and on the voices list (the Voices screen
  and the voice menu) reads the folder again, so what you copied in, renamed or deleted outside the app shows or
  leaves at once. Audio and video files you put in the outputs folder yourself are listed too, by file name: audio
  with Speak's results, a video on its own with Redub's.
* **Fixes:** the voice picked in the header is remembered after a restart (it used to fall back to the first trained
  voice). Enter confirms one-line fields: a chapter's new title, a voice's name, the language search (it did nothing).
  The chapter title field also closes on a click elsewhere. The header's pills no longer run into the window buttons
  when the window is narrow or shrinks.

## 0.3.1

* **Updates window:** the version next to the app's name in the header is now a button that opens it. When a newer
  version is waiting, that button's rim glows in and out, slowly. The window lists every version from 0.2.0 on with
  its notes. **Check for updates** asks for the newest version at any time, even one you skipped. Any listed version
  installs from there, an older one too: it is downloaded, its signature is checked, and it installs when you close
  the app or press **Restart now**. A `.deb` install gets each version's download page instead.
* **The update check runs on the welcome screen too:** it used to wait for a running speech engine, so a copy without
  the engine installed never heard of a new version.
* **AppImage: no leftover mount after an update.** The app passed the AppImage's own open files on to every program
  it started. An AppImage started again after an update kept the old image mounted, with its runtime still running,
  until the new one was closed.

## 0.3.0

* **Romanian:** Speak, Book and Redub now offer Romanian. The speech model had under 100 hours of it in training
  (the cut the other languages passed), so it was tested first: the same six sentences in three languages, from the
  two shipped reference clips, written down again by Whisper. Romanian came out with 13 % of the words wrong, Polish
  with 22 % and Hungarian with 39 % (both already in the list). Record has Romanian reading texts too (Quick,
  Standard and Pro, written for Romanian, not translated), with the analysis that says which sounds a text covers:
  a voice can be recorded and trained in Romanian from start to finish.
* **A voice from recordings you already have:** Record's start page now begins with a card for audio or video
  files of one person (drop them or press **Choose files…**): no reading and no preset. A dialog asks whose voice it
  is, **a new voice** (named after the first file, the default) or **more recordings of the active one**, and the
  card then says how much clean speech the voice has, with **Speak with it** and **Train a closer copy**. Files
  dropped on Speak or Train go the same way; they used to replace the active voice's reference without asking, so
  the selected voice suddenly spoke as somebody else.
* **Record says what it is for:** **This is for** chooses the active voice or **A new voice** (with its name) for a
  reading. The voice menu's **New voice** opens Record set to a new voice (it did nothing when Record was already
  open), and creating an empty voice on the Voices screen goes on to Record for it.
* **The window remembers its size and place,** and opens in the middle of the main display, not maximized, the
  first time. A window closed maximized opens maximized, on the display it was closed on (not on the one the app
  was started from).
* **The app opens on the tab it was closed on** (Speak the first time).
* **Less on screen at once.** Settings that are set once and rarely touched are one line that says what is chosen.
  A press opens the choices in a panel over the page, so nothing on the page moves or jumps. Voices: **Use other
  recordings** and **Other languages** under the reference clip (they filled the card). Redub: the mode's options appear once a file is chosen; **Accent** and
  **Output** (video or audio, the background, the timing) are a line each; Re-speak first asks only what transcribing
  needs, and once there is a transcript its settings fold into **Transcription**, **Accent** and **Output**, so the
  transcript gets the room.
* **Clearer empty states:** Train and Speak without a voice offer **Add a voice**; Speak's "Reference from media…"
  is **More recordings of this voice…**.
* **Much lighter on memory:** an idle Mimvox held 19 GB of memory and 4 GB of the graphics card; now the models leave
  after 10 idle minutes (`MIMVOX_IDLE_EXIT_S`) and the engine keeps 0.3 GB and nothing on the card; the next
  generation loads them again (about 10 s). Voice match starts only for a trained voice, not with the app. Each
  process no longer reserves a math buffer per CPU thread that nothing used. Measured on an RTX 4070 PC: a zero-shot
  voice in use 19 GB -> 5.4 GB, a trained voice 19 GB -> 9.3 GB, the peak while a new reference is read 18.9 GB ->
  12.4 GB; the models are ready 10 s after the start instead of 42 s. The native-accent converter now lives in the
  speech process instead of one of its own (native accent in use: 10.7 GB -> 8.4 GB), and every model process gives
  the RAM it only needed while loading back to Windows (with a trained voice and the native accent loaded: 6.5 GB of
  RAM -> 0.6 GB).
* **Playing and moving the pointer are light on the graphics card.** Measured on an RTX 4070 with a 4K 165 Hz
  display (the card idles at 12 W): moving the pointer over the window drew a whole-window frame for every pointer
  report, 36 W for as long as it moved; now a frame is drawn only when something on screen changes (12 W, also when
  moving fast). Playing redrew the whole window for every position report; now the player's knob glides 60 times a
  second and the window is not redrawn for it at all: on Windows the fill and the knob are a small strip of their own
  over the seek bar (about 16 W while a chapter plays; a knob moved through the window's own frames at that rate kept
  the card at 36 W), elsewhere the bar draws them over the cached rest of the window. The header, the background and
  each panel are drawn once and reused until they change; the big panels no longer blur the (already soft) light
  fields behind them; hover highlights settle faster.
* **Player:** no Stop button any more (pause, the knob and Close cover it). The knob moves on by the clock between
  the audio's position reports and takes a seek at once, so it neither hops back and forth nor jumps back after a
  drag.

## 0.2.0

* **Installers for every platform:** Windows x64 (setup .exe), Linux x64 (AppImage
  and .deb) and macOS on Apple Silicon (.dmg), built by GitHub Actions for every release.
* **Updates from inside the app:** a new version shows a banner under the header. **Update** downloads it, checks its
  signature and installs it when the app closes, or at once with **Restart now** (a running training or book pauses
  and continues). A `.deb` install gets a link to the download page. `"check_updates": false` in settings.json turns
  the check off. An updated AppImage named after the old version is renamed after the new one.
* **Dark Windows installer:** the setup `.exe` is now built with Inno Setup in the app's palette (the banner with the
  app's light fields, our icon, an accent progress bar), with one settings page (folder, desktop shortcut, Start menu).
  The uninstaller asks what to remove: only the app, also the speech engine and its models, or everything. The `.msi`
  is no longer shipped.
* **Install only what you need:** the welcome screen lists the parts of the runtime with their sizes: Speech
  (required, about 13 GB: Speak, Book, Re-speak), Voice training (12 GB), Translation (2.5 GB), Better translation
  (2.7 GB), Native accent (2 GB) and Music removal (0.3 GB). A part left out installs later in one click where it is
  needed (Train, Redub's modes, the accent, voice match, music options), with its progress there; the engine restarts
  by itself afterwards. Setup takes `--components` / `--add` and keeps the chosen set for later runs; Windows without
  voice training gets a portable ffmpeg.
* **Runtime check:** when a new version needs newer engine packages or models, a banner asks to run setup again
  (it installs only what changed) and the engine restarts when setup is done.
* **Setup runs inside the app:** **Run setup** on the welcome screen installs the engine and the models in the app's
  window, with the steps, a progress bar and the log; no terminal, no script and no administrator rights (git and
  ffmpeg are downloaded where missing). **Cancel** stops it, and the next run continues where it stopped. The
  installers carry `uv` for it; `setup.sh` / `setup.cmd` are for developers working in the repository. Without a
  runtime the engine is no longer started (no "cannot read runtime.toml" error).
## 0.1.5

* **Audiobooks:** Speak has a Book mode. Open an EPUB, FB2, DOCX, ODT, RTF, PDF (with a text layer), HTML, Markdown,
  TXT (any encoding) or subtitle file; chapters are found from the book's structure and can be renamed, split, joined
  or left out. The book is read chapter by chapter into one file per chapter (plus an optional M4B with chapter marks),
  with pauses that follow the punctuation and the structure, steady loudness, and a re-take of a segment that comes out
  off. Pause, close the app and resume from the same sentence; listen to finished chapters while the rest is read.
* **Text files in Speak:** "Open text file…" and dropping a text file: short texts go into the editor, long ones open
  as a book.
* **Results persist:** Speak and Redub results come back after a restart (newest first, "Show older"), refresh when the
  screen opens, and leave the list when their files were deleted. A video redub is titled by its video and has its own
  "Open video" button.
* **Player bar:** Stop rewinds to the start and keeps the bar; dragging the seek knob jumps once where it is let go;
  the sound comes back by itself after the PC slept or the output device changed.

## 0.1.4

* **Native accent sounds cleaner on trained voices:** it now goes through the zero-shot converter (Seed-VC) for
  trained voices too. RVC, trained on a few minutes of the user, made the native speaker's audio metallic and its
  pitch shift overshot; RVC is still used for speech with your own accent, and for the native accent when the
  converter is not installed.
* **Player bar:** no longer keeps the GPU busy while something plays (each position update was animated).
* **Build scripts:** a binary built into another `CARGO_TARGET_DIR` is packaged correctly.

## 0.1.3

* **Video redub keeps the music:** for a video source Redub (Translate & dub, Re-speak, Convert voice) separates the
  original voice from the background with Hybrid Demucs (cached, started when the file is chosen), puts the new voice
  over the original music and effects at the original voice's loudness, and writes a copy of the video with the new
  sound. The video stream is copied, never re-encoded; subtitles and metadata are kept. "Output: Video with new voice /
  Audio only" and "Keep background music and sounds" are remembered; a video output always keeps the original timing.
* **Player bar:** everything that plays (results, Original/Result, reference clips, takes) plays in one bar at the bottom
  with play/pause, stop, seek, loop and speed from 0.5x to 2x with the pitch kept.
* **Delete results:** a trash button on every Speak and Redub result removes its files (and the video of a video
  redub) after a confirmation.

## 0.1.2

* **Native accent without training:** an untrained voice can speak English with an American accent. OmniVoice speaks
  the text from a native US reference and a zero-shot voice converter (Seed-VC, installed by setup, GPL-3.0, never
  bundled) gives it the timbre of the voice's reference clip. Native is the default for English from a clip in another
  language. The converter uses 12 diffusion steps (8 with Quality Fast).
* **Reference language:** the engine detects the real language of each reference clip. Voices shows it, warns when it
  differs from the voice's language, and offers "Set the voice to <language>"; the Accent choice appears whenever
  English would carry another language's accent, also in Re-speak.
* **Redub timing:** "Keep original timing" in Translate & dub and Re-speak (on for videos, off for audio files): off
  speaks at a natural pace, one phrase after the other. When fitting, speech is never squeezed below 0.85 of its natural
  length (lower factors swallowed short words), and Speed no longer compounds with the fitting.
* **AMD GPUs on Linux:** setup finds a Radeon card and installs the ROCm builds of PyTorch; the app shows the ROCm
  backend. Setup also always reinstalls torch when it builds an environment, so a CPU build no longer stays after a
  switch to a GPU (contributed by Mykola Bezrukyi, #1).
* **Toasts:** long messages wrap instead of being cut off, and stay longer.

## 0.1.1

* **Native accent (American):** English from a voice recorded in another language can be spoken by a native US
  reference voice and turned into your timbre by your trained voice model (Speak, and Translate & dub to English).
* **Better translation:** NLLB-200 1.3B on GPUs (600M stays as the fallback), sentences split by the transcription are
  translated whole, and a small Ukrainian glossary for slang and clock times ("пів на восьму" is 7:30).
* **Word stress (Ukrainian):** type `+` before the stressed vowel (`співак+ом`) to fix a wrong stress; the engine passes
  it to the speech model in the form it follows.
* **Speak:** a sentence cut to fit a generation chunk no longer leaves a word or two on their own.
* **Redub:** cleaner timing with ffmpeg time-stretch (no more metallic phase-vocoder sound) and use of the pause after
  each phrase.
* **Reference from media:** optional removal of background music (Hybrid Demucs).
* **Results:** one saved file per result, in a format you choose (WAV, MP3, FLAC, OGG, Opus, M4A, AIFF).

## 0.1.0: first public version

Early version: expect bugs.

### Features

* **Record:** guided reading presets (Quick, Standard, Pro) in English, Ukrainian, German, French, Spanish and Polish,
  or your own text with phoneme coverage analysis; microphone picker with a live level meter; a quality check of every
  take; import of existing audio or video files, several files joined into one take.
* **Train:** one-click RVC training (Applio) with progress, time estimate, pause and resume (also across app
  restarts).
* **Speak:** OmniVoice speech in English, Ukrainian and about 60 more languages, streamed sentence by sentence, with
  speed, expressiveness, voice match and quality controls; export to common audio formats.
* **Redub:** Translate & dub (Whisper transcription, NLLB-200 translation with whole sentences kept together and a
  "Translation quality" choice between the 1.3B and 600M models, speech in your voice, optional review of the text
  first), Convert voice and Re-speak, for audio and video files.
* **Voices:** several voices, reference clip choice, a reference picked from many media files, per-language reference
  clips.
* **Setup and build:** one-click setup scripts (`setup.sh`, `setup.cmd`) that install the engine and the models, and
  installer builds for Windows, Linux and macOS (`scripts/build/`).
