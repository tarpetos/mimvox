# Changelog

## 0.1.1

* **Native accent (American):** English from a voice recorded in another language can be spoken by a native US
  reference voice and turned into your timbre by your trained voice model (Speak, and Translate & dub to English).
  Untrained voices get it too, with no training: a zero-shot voice converter (Seed-VC, installed by setup) takes the
  timbre from the reference clip. Native is the default for English from a clip in another language.
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
