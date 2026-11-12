# Performance and quality

All numbers below were measured on one speaker (the developer) and one machine (Windows 11, **NVIDIA RTX 4070, 12 GB**,
32 GB RAM). They show what to expect, not a guarantee. Your hardware, text and voice will give different numbers.

The speech model is **OmniVoice** (k2-fsa) since the multilingual release; it replaced IndexTTS-2.5 after the
head-to-head below. The IndexTTS numbers further down are kept for reference ("Earlier: IndexTTS-2.5").

## Multilingual speech (OmniVoice)

### How the model was chosen

Candidates that speak Ukrainian and clone a voice zero-shot (checked November 2026): **OmniVoice** (k2-fsa, 0.6B,
646 languages, 1852 h of Ukrainian), **FireRedTTS3-Base** (24 languages, Apache-2.0, ~3B with its autoencoder) and
Higgs Audio v3 (4B, research licence, needs SGLang/vLLM serving on Linux: not run). XTTS-v2, Chatterbox (all versions),
CosyVoice 3 and Qwen3-TTS do not speak Ukrainian; MMS-TTS and StyleTTS2-uk do not clone a voice.

Test: a test voice (reference cut to 6.9 s of its English `reference.wav`), five Ukrainian sentences
(stress homographs `замок`/`замок`, `атлас`/`атлас`, numbers, a question) and the English *The cow on the tracks*
paragraph of the Standard preset (73 s of speech); RVC = its trained model (rmvpe, protect 0.33). WER: Whisper
large-v3-turbo back-transcription (numbers normalised to words, so a few points of the Ukrainian WER are Whisper's
spelling and declension of numbers); similarity: ECAPA (speechbrain) cosine to the centroid of the speaker's 13 dataset parts,
where the real recordings score **0.93** against each other. RTF = generation time / speech time, warm model.

| System | English WER (raw / +RVC) | English similarity (raw / +RVC) | Ukrainian WER (raw / +RVC 0.3) | Ukrainian similarity +RVC 0.3 | RTF | First audio | VRAM / host RAM | Download |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IndexTTS-2.5 (engine, 50 steps) | 0.6 % / 1.2 % | 0.69 / 0.93 | does not speak it | - | 1.02-1.05 | 4.5-5.1 s | ~7 GB / 9.4 GB | ~19 GB |
| **OmniVoice, 32 steps** | 0.6 % / **0.0 %** | **0.81 / 0.94** | **6.3 % / 6.3 %** | 0.84 | **0.28** (en), 0.33 (uk) | ~2.6 s | **~4 GB / 2.2 GB** | 3.3 GB |
| OmniVoice, 16 steps | 0.0 % / 1.2 % | 0.80 / 0.94 | 6.3 % / 8.8 % | 0.85 | 0.13 / 0.17 | ~1.5 s | ~4 GB / 2.2 GB | 3.3 GB |
| OmniVoice, 19 s reference | 0.0 % / 2.4 % | 0.88 / 0.93 | 6.3 % / 5.0 % | 0.86 | 0.20 / 0.26 | ~2.5 s | ~7 GB / 2.2 GB | 3.3 GB |
| FireRedTTS3-Base (bf16, SDPA) | 0.0 % / 0.6 % | 0.80 / 0.92 | 8.8 % / 15 % | 0.83 | **~2.0** | >3 s | ~6 GB / 13.5 GB peak | 12.3 GB |

- OmniVoice is as intelligible as IndexTTS in English, clones the speaker more closely before RVC (0.81 vs 0.69) and
  ends a little closer after it, at a quarter of the time and less than half of the VRAM and RAM. FireRedTTS3 is
  slower than real time on Windows (no flash-attention) and loads 13.5 GB into RAM.
- A longer reference clones closer (0.88 raw at 19 s) but costs ~3 GB more VRAM; next to the RVC worker that filled the
  12 GB card and slowed segments 2-3x, so the engine cuts references to 7-12 s.
- **RVC in a foreign language:** at index rate 1.0 the English-only feature index garbled the Ukrainian (WER 25 % vs
  6 %); at 0.3 the words survive and similarity drops only from 0.87 to 0.84. The engine caps the index at 0.3 for
  languages the voice was not recorded in.
- The reference is English, so the Ukrainian carries a light English colour (OmniVoice copies the accent of the
  reference). A 7-12 s Ukrainian clip as `reference_uk.wav` removes it.

### End to end (real engine, OmniVoice + RVC, Windows)

Real server process, a real data directory with two test voices, quality *best*:

| Step | Measured |
| --- | --- |
| Engine start until `tts` and `rvc` are `ready` (incl. warm-up) | **21-27 s** (+8 s the first time a voice's reference is cut by Whisper) |
| English paragraph (73 s of speech, 8 segments) | first chunk **1.2-1.3 s**, RTF **0.25-0.28** (all done in ~15 s) |
| Ukrainian, five sentences (40 s of speech, 3 segments, first segment 81 characters) | first chunk **3.5-4.1 s**, RTF **0.22-0.24** |
| First use of a voice's reference (Whisper cuts it once, cached in `reference.omnivoice.json`) | +7-8 s on that first chunk |
| Translate & dub, 44 s Ukrainian file -> English | **41 s** (decode 2.4, Whisper load 3.5, transcribe 6.1, translate 8.6, speak 19.9, save 0.2) |
| GPU memory in use (whole card) | peak **9.9 GB** of 12.3 GB, of which ~2.9 GB were in use by other programs before the engine started |
| Host RAM (working sets, peak) | OmniVoice worker 1.1 GB (1.9), RVC worker 2.3 GB (2.8), engine 0.25 GB |

The engine environment is one venv (`runtime/engine`, ~7.6 GB) plus the OmniVoice model (3.3 GB); the whole runtime is
about 28 GB instead of 36 GB with IndexTTS.

## Redub translation (NLLB-200 1.3B and 600M)

RTX 4070 (12 GB), Windows, the speech worker alone (`translate` op), fp16 on CUDA, beam search 5, five Ukrainian
sentences (the owner's real phrases) to English:

| | 600M | 1.3B |
| --- | --- | --- |
| Download / on disk | 2.5 GB / 2.5 GB (fp32 `.bin`) | 5.5 GB / **2.7 GB** (stored as fp16 safetensors; download 218 s, conversion 19 s) |
| Load (warm disk cache) | 4.5-5.8 s | 3.7-4.2 s |
| VRAM: weights / peak allocated while translating | 1.35 GB / 1.31-1.39 GB | 2.73 GB / 2.72-2.89 GB (about 3.3 GB for the process with the CUDA context) |
| Five sentences, warm | 0.40 s | 0.63 s |
| Beam 4 instead of 5 | same text, 0.40 s | same text, 0.61 s |

Worker host RAM peaked at 5.4 GB while the 600M's fp32 checkpoint loaded; loading the 1.3B after it did not raise the
peak.

What changed the translations most was not the model size but the context and the glossary (both models):

| Source | 600M before | 1.3B before | Now (glossary; 600M / 1.3B) |
| --- | --- | --- | --- |
| Цей перс спавнюється біля замку кожні п'ять хвилин. | This Persian sleeps at the castle every five minutes. | This Persian breeds at the castle every five minutes. | This character appears / shows up at the castle every five minutes. |
| Зустрінемось о чверть на десяту біля вокзалу. | I'll meet you at 4:00 at the train station. | Meet me at 10:15 at the station. | I'll meet you at 9:15 at the train station. / Meet me at the station at 9:15. |
| Поїзд прибуває о пів на восьму. | The train arrives at half past eight. | The train arrives at half past eight. | The train arrives at 7:30. |
| Я сьогодні цілий день гриндив цей рейд, але так і не випав лут. | I've been grinding this raid all day today, but I've never been out of the woods. | I've been laughing at this raid all day, but it's never happened. | unchanged (not in the glossary: no replacement measured better for both models) |
| Короче, він передумав, і ми пішли назад. | In short, he changed his mind, and we went back. | Anyway, he changed his mind and we went back. | unchanged |

Whisper often ends a segment in the middle of a sentence. The same sentences cut into nine such fragments, translated
one by one, lost their meaning ("This Persian is falling asleep." / "I'm at the castle every five minutes."); translated
as whole sentences and cut back into the fragments they read "This Persian sleeps" / "at the castle every five
minutes." (600M, without the glossary) and "This character shows" / "up at the castle every five minutes." (1.3B, with the glossary).
German and French work the same way (for example "Wir treffen uns um 9:15 Uhr am Bahnhof.", "Le train arrive à
7h30.").

## Training

| Dataset | Setup | Time |
| --- | --- | --- |
| 6.4 minutes (5 parts) | RVC, 300 epochs, batch size 8, 40 kHz, ContentVec embedder, HiFi-GAN vocoder, pretrained start, RTX 4070 | about **55 minutes** |

The number of epochs is chosen automatically from the length of the dataset (300 for 6.4 minutes, more for less
data, fewer for more, between 150 and 450). Training uses the GPU fully, so the app unloads the speech models while it
runs and does not generate speech until it is done.

## Earlier: IndexTTS-2.5 (before the multilingual release)

### Speed

| What | Measured | Why |
| --- | --- | --- |
| One phrase, without a persistent engine | about **23 s** | The models are loaded for the call instead of staying loaded. |
| One phrase, **persistent engine** | about **8 s** | The models stay loaded in GPU memory; only synthesis is paid. |
| RVC, cold start | about **16 s** | The first conversion has to load the RVC model and its embedder. |
| RVC, warm | about **1.7 s** | The persistent RVC worker keeps them loaded. |

Measured with the streaming engine (text split into segments, IndexTTS and RVC overlapped), voice selected and warm:

| Text | Time | Notes |
| --- | --- | --- |
| First audio (first chunk) | about **4.7-5.2 s** | The first segment is just the first sentence, generated with fewer diffusion steps. |
| Two sentences, all done | about **11.3 s** | The second chunk is ready while the first one is still playing. |
| One paragraph, quality *best* | about **34 s** | 50 diffusion steps. |
| One paragraph, quality *fast* | about **31 s** | 35 diffusion steps, about 10 percent quicker, slightly less clean. |
| First chunk, voice warmed up in advance | about **5 s** | The app calls `voices.warm` when you pick a voice. |
| First chunk, voice not warmed up | about **26 s** | The first generation has to compute the speaker conditioning of the reference clip. |

End to end through the real engine process (JSON lines, imported voice from real recordings and a trained model, RTX 4070):

| Step | Measured |
| --- | --- |
| Engine start until both models are ready | about **31-38 s** |
| `voices.warm` of a trained voice (conditioning and RVC model) | about **13-22 s**, nothing else needed afterwards |
| Generate three sentences (two segments, about 15 s of speech), quality *best*, voice warm | first chunk **5.4-6.5 s**, all done in **15.8-16.5 s** |
| Same, quality *fast* | first chunk about **5.0 s**, all done in about **13.2 s** |
| Hand-over to training (`train.start` answers) | about **1.5 s**; both models report `loading` until the job ends |
| 3 epochs on 2.5 minutes of speech, including the one-off kernel compile of the first step | about **145 s** |
| Models back (`ready`) after the training ended | about **34 s** |
| First chunk right after that reload, voice not warmed again | **15-26 s** (call `voices.warm` again) |

Other findings:

- **bf16 was slower** than the default precision on this machine, so the app does not use it.
- Long text is split into segments and each is **streamed**: playback starts when the first segment is ready (the app
  receives a `tts.chunk` event for each), so the wait you feel is that of the first segment, not of the whole text.
  Each later segment may be about 2.5 times larger than the one before (measured to give the lowest total time), so a
  chunk finishes playing after the next one is ready, as long as generation runs at about real time (1.05-1.4x).
- The first phrase after the app starts is slower than the rest: the models are loading (about 50 s at start-up).
- While a voice is being trained the speech models are unloaded to give the GPU to training; they reload in about 50 s
  afterwards.

### Voice similarity (IndexTTS)

Speaker similarity was measured with ECAPA speaker embeddings (cosine similarity, 1.0 means identical). The ceiling is
how similar the speaker's own real recordings are to each other.

| Output | ECAPA similarity to the real voice |
| --- | --- |
| IndexTTS-2.5 zero-shot (reference clip only) | about **0.75-0.80** |
| IndexTTS-2.5 + RVC trained on 6.4 minutes of the speaker | about **0.86-0.88** |
| Ceiling: the speaker's real recordings against each other | **0.88** |

So the timbre model closes almost the whole gap between a zero-shot clone and a recording of the real person.

### Validated IndexTTS settings ("B")

Reference 15 s; diffusion steps 50; CFG 0.7; temperature 0.8 (*Expressiveness*); top_p 0.8; top_k 30; num_beams 1;
repetition penalty 10; max 120 text tokens per segment; language `EN`; emotion equal to the reference clip; then RVC with
index rate 1.0 (*Voice match*), rmvpe pitch extraction, protect 0.33, pitch 0.

Using emotion vectors instead of the reference clip's own emotion made the voice drift (an Indian-sounding accent was
the clearest symptom), which is why the app does not use them.

## Disk and memory

- About 28 GB for the runtime and the models: the engine environment with OmniVoice ~11 GB, Applio/RVC ~12 GB (two
  Python environments with their own PyTorch), Whisper and NLLB 600M for Redub ~4.5 GB, Hybrid Demucs for music removal
  ~0.3 GB. With an NVIDIA GPU setup adds NLLB 1.3B (2.7 GB in fp16, a 5.5 GB download): about 31 GB.
- OmniVoice needs about 4-5 GB of VRAM with the 7-12 s reference cut; the RVC worker adds ~1-2 GB. 8 GB or more is
  recommended and 12 GB was used in testing.
