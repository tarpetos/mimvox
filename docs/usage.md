# Using Mimvox

This guide walks you through the app from installation to your first dubbed video. It describes what you see on screen
and what happens behind it, with concrete examples. For the background (how the parts fit together, measured speed)
see [architecture.md](architecture.md) and [performance.md](performance.md); for microphone and room advice see the
[recording guide](recording-guide.md).

> **Consent first.** Clone only your own voice, or the voice of a person who has given you explicit permission. The model
> weights are licensed for non-commercial use only. Read [DISCLAIMER.md](../DISCLAIMER.md) before you record anything.

## Contents

1. [Install and first start](#1-install-and-first-start)
2. [The window at a glance](#2-the-window-at-a-glance)
3. [Record](#3-record)
4. [Voices](#4-voices)
5. [Train](#5-train)
6. [Speak](#6-speak)
7. [Redub](#7-redub)
8. [Tips for quality](#8-tips-for-quality)
9. [Troubleshooting](#9-troubleshooting)
10. [Your data folder](#10-your-data-folder)
11. [Consent and licences](#11-consent-and-licences)

---

## 1. Install and first start

### What you need

- An **NVIDIA GPU with 8 GB of video memory or more** is recommended (Windows or Linux). Apple Silicon works on a
  best-effort basis through MPS. Without a supported GPU the app still runs on the CPU, but training a voice takes many
  hours and speech is generated several times slower than real time. Intel Macs are not supported.
- About **32 GB of free disk space** on the drive you install to (the runtime itself is about 28 GB).
- A microphone, and an internet connection for the one-time setup. After setup the app works offline.

### Run the one-click setup

From the project folder:

```bash
./setup.sh          # Linux and macOS
```

```powershell
.\setup.cmd         # Windows
```

The script installs `uv` if it is missing, then runs `tools/setup/setup.py`. This is what it asks and does:

1. **Where to install (the data folder).** It prints the free space on each drive and a default:

   | System | Default data folder |
   | --- | --- |
   | Windows | `%LOCALAPPDATA%\mimvox` |
   | macOS | `~/Library/Application Support/mimvox` |
   | Linux | `~/.local/share/mimvox` (or `$XDG_DATA_HOME/mimvox`) |

   Press **Enter** for the default, or type a full path to another folder (for example on a bigger drive). A folder
   other than the default is remembered in a small pointer file, `location.toml`, so the app finds it later.
2. **The size warning.** Before anything is downloaded, setup shows the install location, the free space there and
   what it is about to download, then asks `Continue? [Y/n, or type another folder]`. Answer `n` or type another
   path to install elsewhere. If the drive has less than 32 GB free, setup stops (`--force` overrides this).
3. **GPU or CPU.** Setup detects the hardware by itself and prints the backend it will use: `cuda` (an NVIDIA GPU,
   found with `nvidia-smi`), `mps` (Apple Silicon) or `cpu` with a warning. Pass `--cpu` to ignore a GPU on purpose.
4. **Toolchains.** It installs what is missing to build the app: Rust, a C/C++ toolchain (Visual Studio Build Tools
   on Windows, the Xcode command line tools on macOS, the system packages on Linux) and `ffmpeg`.
5. **Downloads.** Then it builds the runtime and fetches the models:

   | What | Size (about) | Used for |
   | --- | --- | --- |
   | Engine environment with PyTorch, plus the OmniVoice speech model (3.3 GB) | 11 GB | Speaking text in about 60 languages |
   | Applio environment (RVC) with its pretrained models (about 1.5 GB) | 12 GB | Training your voice and converting speech to it |
   | Whisper turbo (1.6 GB) and the NLLB-200 translation model (2.5 GB) | 4.5 GB | Redub: transcription and translation |
   | Each voice you make | 0.1 GB | Your recordings and trained model |

6. At the end it writes `runtime.toml` into the data folder and prints the command that starts the app.

Setup can be run again at any time. Every step checks what is already there and skips it, and an interrupted download
resumes where it stopped, so after a lost connection or Ctrl-C you simply run it again.

Useful options (all of them: `./setup.sh --help`):

| Option | What it does |
| --- | --- |
| `--data-dir DIR` | Install into `DIR` without asking. |
| `--yes` | Do not ask for confirmation (then `--data-dir` is required). |
| `--cpu` | Set up the CPU backend even when a GPU is present. |
| `--quick-check` | Verify an existing install, download nothing. |
| `--move-to DIR` | Move an existing install to `DIR` (see below). |

### Start the app

```bash
cd app
cargo run --release
```

If you installed the app from an installer (see `scripts/build/README.md`), start it like any other program. An
installed app does not include the models: if they are missing, the first screen offers **Run setup** (see below).

### First launch: the welcome screen

The first start shows a welcome screen with three steps (Record, Train, Speak) and the engine card:

![The welcome screen and the engine status](screenshots/usage-onboarding.png)

1. **Engine and device.** *Runtime found* means setup has run. If it says *Runtime missing*, the card shows the size
   warning and an **Install location** field with **Change location…**. Press **Run setup**: a terminal window opens
   and runs the same setup script with that folder. When it is done, the screen updates by itself (or press
   **Check again**). Once the engine is running, the card shows your device, its video memory and the backend. On a
   computer without a GPU it says that training would take many hours, while a quick (zero-shot) voice still works.
2. Press **Get started**. You land on the Voices screen.

The first start of the engine takes a while: the models load in the background (about 20-30 seconds on a GPU). The
status pill at the top right says *Starting engine…*, then *Loading models…*, then **Ready**.

### Moving the data folder later

To move everything (runtime, models and your voices) to another drive:

1. Open **Voices** (the voice menu at the top right, then **Manage voices**).
2. At the bottom, under **Data location**, press **Move data…** and pick an empty folder.
3. The app shows how much it will move and the free space on the new drive, then asks you to confirm. The voice engine
   stops while the files are copied and starts again afterwards. Nothing is deleted from the old folder until the copy
   is complete and checked; if the move fails, your data stays where it was and keeps working.

The same can be done from a terminal while the app is closed: `./setup.sh --move-to <new folder>` (or
`setup.cmd --move-to <new folder>` on Windows). An interrupted move resumes when you run the same command again. A data folder
fixed by the `MIMVOX_DATA` environment variable cannot be moved from the app.

---

## 2. The window at a glance

- **Tabs** in the middle of the header: **Record**, **Train**, **Speak**, **Redub**.
- **Engine pill** on the right: *Ready*, *Loading models…*, *Training · speech paused*, *Models missing* or
  *Engine stopped*. When the engine has stopped, press it to restart.
- **Voice pill** at the far right: the active voice. Open it to switch voices, see whether each one is *Trained* or
  *Zero-shot*, and reach **Manage voices** (the Voices screen) or **New voice**.
- **Dropping files** anywhere on the window works on every screen: on Record they become the current part's take, on
  Voices they build a reference from media, and elsewhere a single file opens in Redub (see the sections below).

---

## 3. Record

Recording goes into the **active voice**. If you have none yet, the app creates one named "My voice" and selects it.

### Pick a preset and a reading language

![Choosing a preset and the reading language](screenshots/usage-record-picker.png)

At the top, choose the **Reading language**: English, Ukrainian, German, French, Spanish or Polish. Each language has
its own texts, written for that language, in `presets/<lang>/` (`quick.md`, `standard.md`, `pro.md`). Then pick a
preset:

| Preset | Reading | Parts | What you get |
| --- | --- | --- | --- |
| **Quick** | about 1 minute | 1 | *Ready at once*: no training. Your recording becomes the reference clip and the voice works immediately (zero-shot). |
| **Standard** | about 6-7 minutes | 5 | *Trains your voice*: enough to train a timbre model (RVC) for a close match. The validated setup. |
| **Pro** | about 17 minutes | 13 | *Trains your voice*: more data, the best similarity and the most stable result. |

The voice is created in the language you read. If the active voice already has recordings in another language, the
app keeps it as it is and makes a new voice named, for example, "My voice (Ukrainian)".

**Example: a Standard voice in English.**

1. Open **Record**, set **Reading language** to English and press the **Standard** card.
2. Part 1 of 5 appears in large text. Press **Space** (or the red button) and read it aloud at your normal pace.
3. Press **Space** again when you finish. The take is checked, saved, and the next part opens by itself.
4. Repeat for all five parts. Then press **Ready to train**.

Keys: **Space** starts and stops a recording, **Left** and **Right** move between parts. The **Parts** list on the
right shows which parts are done; you can click any part to go back to it. The bar at the bottom shows the minutes
recorded against the goal. **All presets** goes back to the preset list; **Continue recording** picks up a reading you
started earlier.

![Recording a part of the Standard preset](screenshots/record.png)

### Your own text, with phoneme analysis

Prefer to read something else? On the preset screen press **Paste text** under *Use my own text*.

1. Paste or type the text (a few minutes of text works best). Set **Language** to **Detect** or pick one.
2. Press **Analyze text**. The app shows the word count, the estimated reading time, the number of parts, the
   language, and a level: *Too short*, *Quick clone*, *Trainable* or *Great*.
3. **Speech sound coverage** is the share of the language's speech sounds (phonemes) your text contains. **Missing
   sounds** lists the ones it lacks, and the message under it gives an example word for each. Add a few sentences with
   those words and analyse again: the more sounds the text covers, the more complete the voice. Coverage below 80%
   lowers the level by one step.
4. Press **Record these N parts**. The text is split into parts of about 60-90 seconds, recorded exactly like a preset.

![Analysing your own reading text](screenshots/usage-record-own-text.png)

Write one language per text, and write numbers and abbreviations the way they are said.

### Choose the microphone and check the level

The microphone chip next to *Space record / stop* shows the input in use (*System default* until you choose).

![The microphone list with a live level test](screenshots/usage-record-mic.png)

1. Press the chip. The list shows *System default* and every input device the system offers.
2. While the list is open, a live **Input level** test runs on the selected device: say a few words and watch the bar
   move. Nothing is recorded during this test.
3. Pick a device. The choice is remembered. If it is unplugged later, the app falls back to the system default and
   shows a warning such as *"… is not available, using the system default"*.

While you record, a level meter runs under the text. Aim for the bar to move well into the green and yellow during
speech, without touching red. If it says *"Too loud. Move back from the microphone or lower the input volume."*, do
that and record the part again. You cannot switch microphones while a recording runs.

### Quality verdicts and re-recording

Every take is checked as soon as you stop: duration, peak level, clipping, loudness, background noise,
signal-to-noise ratio and silence. You get one of three verdicts:

| Verdict | Meaning | What happens |
| --- | --- | --- |
| **Sounds good** | No problems. | Saved, and the next part opens automatically. |
| **Usable, with notes** | Saved, with advice (for example *"Background noise: a quieter room or a closer microphone would help"*). | Read the notes; press **Continue** to move on, or **Record again** to improve it. |
| **Please record again** | Not usable (too short, too quiet, clipped, mostly silence, too noisy, or no clear speech). | The take is *Not saved*. Press **Record again**. |

![A saved take with notes](screenshots/usage-record-issues.png)

To redo any part, select it in the **Parts** list and press **Record again**. A new take replaces the older one of that
part. **Play take** plays back what you recorded.

### Import a file instead of recording

You can use existing recordings of yourself instead of reading live.

- **Import file…** (top right of the part card) opens a file picker. Choose one file, and it becomes the take of the
  current part. It goes through the same check as a live take.
- **Drag and drop** a file anywhere on the window while a part is open. The part card lights up with *"Release to use
  this file for part N"*.
- **Several files at once** (picked together, or dropped together) are combined into one take. The engine joins them
  (`media.combine`): each file is cleaned (silence trimmed at both ends, long pauses shortened, loudness evened out),
  then they are joined with a short gap, up to 20 minutes in total. You see *"Combining N files into one recording…"*
  and then, for example, *"3 files combined into one recording (4.2 min)."* Files that cannot be read or have no clear
  speech are skipped and named in that message.

![Dropping a file on a part](screenshots/usage-record-drop.png)

Accepted formats are audio and video: WAV, MP3, M4A, AAC, FLAC, OGG, Opus, AIFF, WMA (when your ffmpeg can decode it),
MP4, MKV, MOV, WebM and AVI. From a video, only the sound is used. Only import recordings of your own voice, or of a
person who has given you permission.

---

## 4. Voices

Open the Voices screen from the voice pill at the top right (**Manage voices**).

![Your voices](screenshots/usage-voices.png)

### Create, rename, delete and switch voices

- **Create:** type a name under **New voice** (for example "My voice") and press **Create voice**. A new voice has no
  recordings yet: go to Record next. (Record also creates "My voice" by itself when there is none.)
- **Active voice:** the voice with the *Active* badge is the one Record, Train, Speak and Redub use. Press **Use this
  voice** on another card, or pick it from the voice pill at the top, to switch.
- **Rename:** press **Rename**, edit the name, press **Save**. Only the shown name changes; the voice's folder keeps its
  original id.
- **Delete:** press **Delete**, then confirm *"Delete this voice and its recordings?"*. This removes the voice's whole
  folder (recordings, reference clips and trained model) and cannot be undone. Back it up first if you might want it
  again (see [Your data folder](#10-your-data-folder)).

Each card shows *Trained* or *Untrained*, the minutes recorded, and the number of epochs after training. A paused or
interrupted training shows as *Paused 115/300* or *Interrupted 115/300*.

### The reference clip and "Choose…"

The speech model clones your voice from one short clip of your recordings, the **reference clip**. It copies the pace
and manner of that clip, so a slow or halting clip makes the cloned voice slow. The app picks it automatically from
your best take: a 15-20 second window with a clean signal, a natural pace and few long pauses. (The speech model itself
uses up to 12 seconds of it.)

The active voice's card has a **Reference clip** section showing the clip in use (part, title, and a *Natural pace*,
*Slow* or *Fast* badge) with a play button.

1. Press **Choose…** to see the best candidates from your recordings. The first one is marked *Recommended*.
2. Play a few. Pick one that sounds like your natural, relaxed reading.
3. Press **Use this one**. The next speech uses the new clip. A clip you choose is marked *Chosen by you* and is kept
   until you re-record or remove the part it came from.

![Choosing a reference clip](screenshots/usage-voices-reference.png)

### Reference from media

You can also make the reference from existing audio or video of the same speaker, for example several talks or voice
memos. One noisy source makes a metallic clone, so the app combines many files and picks the cleanest piece.

1. In the **Reference from media** block of the active voice, choose the language the media is in (the chip; by default
   the voice's own language).
2. Press **Add media…** and pick the files, or drop them on the Voices screen. Many files are fine. (Dropping files on
   Speak, or several files on Train or Redub, also brings you here.)
3. The app cleans each file (*Cleaning · File k of n*), combines them, and looks for the best window. You can press
   **Cancel**; a cancelled run changes nothing.
4. When done it reports, for example, *"12 files combined · 7:32 · reference set: 11.4 s from file 3"*, with a play
   button for the new reference and a list of skipped files with the reason.

![A reference made from media files](screenshots/usage-voices-media.png)

Details:

- Up to 20 minutes of the files are used; later files are listed as not used.
- The pool is cut into windows of **8-12 seconds**, scored by signal-to-noise ratio, pace and pauses. The best one
  becomes the reference. The other good windows appear in the **Choose…** list under *From your media*.
- If you chose the voice's own language, the result replaces the voice's main reference clip. If you chose another
  language, it becomes the per-language clip for that language (see below).
- Your recordings for training are never touched: media files are used only for the reference.
- **Remove combined media** deletes the combined files; the reference clip in use stays.
- Only use media of your own voice, or of a person who has given you permission.

### Per-language reference clips

A voice speaks every language of the speech model, but with the accent of its reference clip: an English reference
gives Ukrainian with a slight English colour. A short clip of you speaking the other language fixes that.

1. In the **Other languages** part of the reference section, press **Add a language** and pick the language.
2. The app shows a short sample text in that language. Press **Record** and read it at your natural pace (about 10
   seconds), or press **Import file…** to use an existing recording in that language.
3. The take is checked (*Sounds good*, *Usable* or *Not usable*). Press **Use for <language>**.

The clip is used whenever you speak or dub into that language. The voice's card lists the languages that have one;
each can be played or removed. A clip needs 3 to 40 seconds of speech; about 10 seconds is best.

![Per-language reference clips](screenshots/usage-voices-languages.png)

---

## 5. Train

### What training adds

Without training, a voice is **zero-shot**: the speech model imitates you from the reference clip alone. This works at
once (the Quick preset) and sounds similar to you, but not the same. **Training** builds a timbre model (RVC) from all
your recordings. Every phrase is then converted with it, which brings the timbre much closer to your real voice. In
the project's measurements on one speaker, speaker similarity rose from about 0.81 to about 0.94 on an English
paragraph, where 0.93 was the similarity of that speaker's own real recordings to each other
([performance.md](performance.md)).

Training needs at least 3 usable minutes of recordings (the *Trainable* level). With less, the screen offers **Go to
Record**, or **Go to Speak** to use the voice zero-shot.

### Start a training

![Training in progress](screenshots/usage-train.png)

1. Open **Train**. *Your recordings* shows the minutes recorded, the number of parts, the recommended epochs and the
   level.
2. Leave **Epochs** on **Auto** (recommended). Auto picks the number from the length of your recordings: about 300
   for 6-7 minutes, more for less data and fewer for more, between 150 and 450. **Custom** lets you set it yourself.
3. Press **Start training**. The screen shows the stage (*Preparing your recordings*, *Analysing your voice*,
   *Training your voice*, *Building the voice index*), the epoch, the progress, the elapsed time and an estimate of
   the time left.

**How long it takes:** about 55 minutes for 300 epochs on 6.4 minutes of recordings on an RTX 4070 with 12 GB. The Train
screen shows its own estimate before you start. Expect several hours on an Apple GPU and many hours without a GPU.

While a voice trains, the GPU belongs to training: Speak and Redub are unavailable (the engine pill says *Training ·
speech paused*). You can still use the other screens. When training finishes, the speech models reload (about half a
minute) and the voice is marked *Trained*. Press **Go to Speak**.

### Pause and resume

- **Pause** becomes available once the actual learning starts (not during the preparation stages). The app stops at
  the next saved checkpoint, which is written every 5 epochs, so pausing can take up to about a minute. At most a few
  epochs of work are lost.
- While paused, **Speak and Redub work again**. The Train tab and the voice card show *Paused 115/300*.
- Press **Resume** (available once the speech models have loaded again) to continue from the saved epoch.
- **Discard** throws the paused run away. **Cancel** during a training stops it and discards its progress; the voice
  keeps its previous model, if it had one.

### Closing the app during training

If you close the window while a voice trains, the app asks first (*Training is running*):

- **Pause and close:** waits for the next checkpoint, then closes. Resume later from Train.
- **Keep training:** cancels the close.
- **Close anyway:** closes at once. On the next start the run shows as *Interrupted* and can be resumed from the last
  checkpoint (if learning had not started yet, there is no checkpoint and the run starts over).

The same recovery applies after a crash or a power cut.

---

## 6. Speak

![Speaking a Ukrainian text with a voice recorded in English](screenshots/usage-speak.png)

### Generate speech

1. Make sure the voice you want is active (top right). *Speaking as My voice* under the text box confirms it.
2. Type or paste text into **What should I say?**. The counter shows the words and the estimated length of speech.
3. Press **Generate** (or **Ctrl+Enter**).

### The Style panel

| Control | What it does |
| --- | --- |
| **Language** | The language the text is written in. Any of the about 60 languages; the common ones are listed first and you can search. It defaults to the language the voice was recorded in, and the choice is remembered per voice. |
| **Speed** | 0.8x (*Slower*) to 1.3x (*Faster*), default 1.00x. Relative to the pace of the reference clip. |
| **Expressiveness** | *Steady* to *Lively*, default 0.80. Lower is flatter and steadier, higher varies phrasing and pauses more. |
| **Voice match** | *Natural* to *Closest to me*, default 1.00. How strongly the trained timbre model pulls the sound towards your recordings. Applies only to a trained voice. In a language the voice was not recorded in it is capped at 0.3 so the words stay clear. |
| **Quality** | **Best** (32 steps per sentence, the richest detail) or **Fast** (16 steps, about twice as fast, slightly less clean). |

The defaults are the validated settings; you do not need to change them.

### Streaming playback

Long text is split into segments: the first sentence alone, then growing groups of sentences. Each segment is played
as soon as it is ready while the rest is still being made, so you hear the first words after a few seconds. On slower
hardware the app holds the first segment back a little (*Getting ahead · part k of n*) so playback never stutters.

The very first phrase after the app starts is slower, because the voice has to be prepared (*Preparing voice…*). Later
phrases are much faster: leave the app open and the models stay loaded.

### Results and export

Each finished phrase appears under **Results** with its length and how long it took to make. For each one:

- the **play** button plays it again;
- **Export as…** saves a copy in another format (see below);
- the folder button shows the file in your file manager, and the other button opens it in your default player.

Every result is also saved automatically as one file in the `outputs` folder of your data folder
(`outputs/<date>_<time>.wav`). The Results list itself shows the phrases made since the app was started.

**Save as** (the chip next to the Results title, "Save as WAV" at first) chooses the format of that file: WAV, MP3,
FLAC, OGG, Opus, M4A or AIFF, each with a one-line description. The choice is remembered and shared with Redub, and
applies to the next result. WAV is the default and the best for editing; MP3, Opus or M4A make much smaller files
(saved at high quality, for example MP3 at 320 kbps). Playback in the app always works, whatever the format: for a
format other than WAV the app plays a temporary WAV copy that is removed when the app restarts. The folder button,
**Open file** and **Export as…** use the saved file.

**Export as…** offers WAV, MP3, FLAC, OGG, Opus, M4A and AIFF, with a quality of **High**, **Standard** or **Small**
for the lossy formats (for example MP3 at 320, 192 or 128 kbps). Your last choice is remembered.

![The Export as menu](screenshots/usage-export.png)

### Example: a Ukrainian paragraph with a voice recorded in English

1. Pick your English voice (for example "My voice", recorded with the English Standard preset).
2. Open **Speak** and set **Language** to **Ukrainian**. A hint appears: *"Voice recorded in English: timbre kept, a
   Ukrainian reference clip improves the accent."*
3. Paste the text:

   > Сьогодні вранці я вийшов на прогулянку вздовж річки. Повітря було свіже, а небо ясне, і я подумав, що це
   > слушний момент записати свій голос.

4. Press **Generate**. Numbers, dates and abbreviations are spelled out in Ukrainian before speaking (for example
   `у 2026 році` is read as "у дві тисячі двадцять шостому році").
5. The result keeps your timbre but has a slight English accent. To remove it, press **Record a Ukrainian reference**
   in the hint. It opens the Voices screen with the per-language panel ready: read the sample text for about 10
   seconds, press **Use for Ukrainian**, come back and generate again.

---

## 7. Redub

Redub takes an existing audio or video file and says it again in your voice. Open **Redub**, then drop a file on the
drop zone (or anywhere on the window) or press **Choose file…**. The file card shows its length and format, with a
button to play the original and **Change…** to pick another one.

There are three modes.

| Mode | What it does | Needs |
| --- | --- | --- |
| **Translate & dub** (default) | Transcribes the speech, translates it to the language you choose and speaks it in your voice, each sentence at its original time. | Any voice, trained or not. |
| **Convert voice** | Keeps the original recording, its timing and delivery, and swaps in your trained voice. | A trained voice. |
| **Re-speak** | Transcribes the speech, lets you edit (or translate) the words, then speaks them again in your voice. | Any voice. |

The Style panel on the right (Speed, Expressiveness, Voice match, Quality) is the same as on Speak. Speed and
Expressiveness do not apply to Convert voice, which keeps the original pace.

### Example: a Ukrainian video to English in a couple of clicks

1. Open **Redub** and drop `interview.mp4` (a video of you speaking Ukrainian).
2. Keep **Translate & dub**. Set **From** to **Auto-detect** (or Ukrainian) and **To** to **English** (the default).
3. Press **Translate & dub**.

The job reads the audio, loads the speech models (the first time takes longer), transcribes, translates, speaks each
sentence and saves the result. You can use other tabs meanwhile; **Cancel** stops it. As a measured example, a
44-second Ukrainian clip took about 41 seconds end to end on an RTX 4070. The result appears under **Results** with
**Original** and **Result** buttons so you can compare them.

![A finished dub](screenshots/usage-redub-done.png)

How the timing works: each translated sentence is fitted into the time of the original sentence. The speech is made a
bit faster or slower first, then stretched slightly if needed; if it still does not fit, it runs into the following
pause and later sentences move a little. Speech is never cut off and sentences never overlap.

### The same, with "Review text before dubbing"

Machine translation is not perfect, and names are easy to get wrong. To check the text first:

1. Tick **Review text before dubbing** before you start, then press **Translate & dub**.
2. The job stops after the translation and shows a **Review** table: the original on the left, the English on the
   right, with the time of each line.
3. Fix any line on the right. Empty lines are skipped; the trash button removes a line.
4. Press **Dub it**. Only now is the text spoken in your voice. **Discard review** drops it.

![Reviewing the translation before dubbing](screenshots/usage-redub-review.png)

### Convert voice

1. Choose a file and select **Convert voice**.
2. Press **Render**.

Your trained timbre model runs over the whole file, so the result has exactly the original length and timing. Music
and background noise are converted along with the voice (they are not separated), so this works best on a clean
recording of speech. With an untrained voice the mode offers **Go to Train** or **Use Translate & dub** instead.

### Re-speak

1. Choose a file and select **Re-speak**.
2. Set **Spoken language** (or Auto-detect). Tick **Translate the transcript** and pick **To** if you want it in
   another language.
3. Press **Transcribe**. The transcript appears as editable lines with their times.
4. Fix the words, delete lines you do not want, then press **Render**. Transcribing again replaces your edits.

### Supported formats and exporting

- **Input:** WAV, MP3, M4A, AAC, FLAC, OGG, Opus, AIFF, WMA (when your ffmpeg can decode it), and the video formats
  MP4, MKV, MOV, WebM and AVI. From a video, the first audio track is used.
- **Output:** audio only, mono 44.1 kHz. The video picture is not put back. Each result is saved automatically as one
  file, `outputs/redub_<date>_<time>.wav`, or in the format chosen with **Save as** in the Results header (shared with
  Speak, see [Results and export](#results-and-export)).
- **Export as…** on a result saves it as WAV, MP3, FLAC, OGG, Opus, M4A or AIFF, as on Speak.

To put the new audio back into your video, use a video editor and replace the original sound track with the exported
file.

Redub uses the GPU like Speak: only one job runs at a time, and Redub is unavailable while a voice trains.

---

## 8. Tips for quality

- **A clean reference.** The speech model copies whatever is in the reference clip: noise, echo, a tired voice, a slow
  pace. Listen to it on the Voices screen and use **Choose…** to switch to a relaxed, natural-sounding clip.
- **No music, no noise.** Record in a quiet, soft room. For media files and Redub sources, prefer clips with speech
  only: background music and other voices end up in the clone or the conversion.
- **A consistent microphone.** Use the same microphone, distance (15-20 cm) and room for every part, and for parts
  you add later. Turn off noise suppression and automatic gain in your system if you can.
- **Reference length.** The automatic reference is a 15-20 second window, and the speech model uses up to 12 seconds
  of it; media references are 8-12 seconds; per-language clips need 3-40 seconds, about 10 is best. Longer is not
  better here: clean and natural is what counts.
- **Accent across languages.** The voice keeps the accent of its reference. For the most natural result in a language
  you did not record in, add a per-language reference clip in that language. For trained voices, Voice match is
  capped at 0.3 in other languages so the words stay clear.
- **Quality beats quantity.** Six clean minutes are better than twenty noisy ones. Re-record a part as soon as the check
  says *Please record again*, not after training.

More detail: the [recording guide](recording-guide.md).

---

## 9. Troubleshooting

| Problem | What to do |
| --- | --- |
| **"The voice engine stopped"** (banner under the header, the pill says *Engine stopped*) | Press **Retry** in the banner, or press the pill. The banner shows the last error line. If it keeps stopping, look at the engine log (below) and run `./setup.sh --quick-check` (or `setup.cmd --quick-check`) to verify the install. A *"not set up yet"* message means setup has not completed: run it again. |
| **Generation is slow** | The first phrase after start is always slower (the models and the voice are being prepared). Check the device on the welcome screen or by hovering the *Ready* pill: it should name your NVIDIA GPU. If setup reported the `cpu` backend, re-run setup on a machine with a supported GPU and current NVIDIA driver. Close other programs that use the GPU (games, other AI tools, browsers with hardware acceleration). **Fast** quality is about twice as fast per sentence. Do not train while you want to speak. |
| **"CUDA out of memory"** | Close other programs that use the GPU. 8 GB of video memory is the practical minimum. If it persists, restart the app so the engine reloads cleanly. |
| **The microphone is missing** | Open the microphone chip on Record: the list is refreshed every time it opens. Plug the microphone in before opening it, check that it is enabled and not muted in your system's sound settings, and that the app is allowed to use the microphone (macOS and Windows privacy settings). *"Cannot use this microphone"* shows the system's reason. A remembered microphone that is gone falls back to the system default with a warning. |
| **"No clear speech found"** on every take | The wrong input is selected, or it is muted. Use the level test in the microphone list: the bar must move when you speak. |
| **A file is refused** | *"That kind of file is not supported"*: use one of the formats listed in [Redub](#supported-formats-and-exporting). A video without sound gives *"This file has no audio track."* |
| **Redub says the translation model is not installed** | Run setup again; it downloads the missing Redub models. |

### Where the logs are

All paths are relative to your data folder (see [the table above](#run-the-one-click-setup)):

| File | What is in it |
| --- | --- |
| `logs/engine.log` | Everything the engine writes while it runs (model loading, errors). It is started fresh every time the engine starts, so copy it **before** you press Retry or restart the app if you want to keep it. |
| `tmp/train_<voice id>.log` | The full output of the last training of that voice. The `tmp` folder is emptied at every engine start. |

Setup prints its progress to the terminal it runs in; there is no separate setup log.

---

## 10. Your data folder

Everything the app makes lives in one folder (the one you chose during setup; see the defaults in
[Install](#run-the-one-click-setup)). The **Data location** card at the bottom of the Voices screen shows its path.

```
<data folder>/
  voices/
    my-voice/                    one folder per voice (the id is made from the name)
      voice.json                 name, language, training state
      reference.wav              the reference clip
      reference_uk.wav           optional per-language reference clips
      reference_media.wav        optional combined media for "Reference from media"
      dataset/part1.wav ...      your recorded parts
      rvc/model.pth, model.index the trained voice model (after training)
  outputs/                       every generated phrase and redub, one file each (WAV unless "Save as" says otherwise)
  logs/engine.log                the engine log
  tmp/                           scratch files, emptied at every engine start
  settings.json                  your app settings (microphone, last choices, "Save as" format)
  runtime/, runtime.toml         the installed engine and models (about 28 GB)
```

The id of a voice folder is made from its name: lower case, with letters of other scripts transliterated (for example
"My voice" becomes `my-voice`). Renaming a voice in the app does not rename its folder.

### Back up a voice

1. Close the app (so nothing is being written).
2. Copy the voice's folder, for example `voices/my-voice/`, somewhere safe. It holds the recordings, the reference
   clips and the trained model.

To restore it, close the app and copy the folder back into `voices/`. It shows up in the list on the next start. A
paused training is not part of this backup (its checkpoints live in the runtime folder); finish or discard it first.

Your generated speech is in `outputs/`; copy that folder too if you want to keep it.

### Delete a voice

On the Voices screen press **Delete** on the voice's card and confirm. This deletes the voice's folder for good. You can
also delete the folder by hand while the app is closed. To remove everything the app made, delete the whole data
folder (and `location.toml`, if setup created one for a custom location).

---

## 11. Consent and licences

- **Clone only your own voice, or the voice of a person who has given you explicit permission** (ideally in writing). Never
  use recordings of someone else taken from calls, videos or podcasts. This applies to recording, importing files,
  reference from media and Redub alike.
- **Do not use generated speech to deceive anyone**, and say that it is synthetic where the law or common decency
  expects it.
- **Non-commercial use only.** The OmniVoice speech model weights and the NLLB-200 translation model are licensed
  CC BY-NC 4.0.

The full terms are in [DISCLAIMER.md](../DISCLAIMER.md), and the licences of every component in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
