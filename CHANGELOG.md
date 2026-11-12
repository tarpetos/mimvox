# Changelog

## 0.1.0 (unreleased): first public version

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
