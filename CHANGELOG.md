# Changelog

## Unreleased

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
