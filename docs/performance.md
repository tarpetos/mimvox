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

## Zero-shot native accent (Seed-VC)

The native accent of an **untrained** voice: OmniVoice speaks the English from the shipped native clip
(`en-us-male.wav`, CMU ARCTIC `bdl`), then a zero-shot voice converter gives it the timbre of the voice's reference
clip. Three converters were measured on the owner's voice (only Ukrainian recordings: an 11 s clip `reference_uk.wav`,
the 17.7 s `reference.wav`, 43 s of dataset audio), RTX 4070 next to the running app, three English test sentences of
7-8 s each, every number the mean over the three:

- **ECAPA hold-out**: cosine similarity (speechbrain `spkrec-ecapa-voxceleb`) to 25 s of the owner's dataset audio that
  is in neither reference clip; the main speaker-similarity score. **ECAPA uk**: to the 11 s Ukrainian clip.
  **WavLM-SV**: `microsoft/wavlm-base-plus-sv` to the dataset audio (it favours kNN-VC, which is built on WavLM features,
  and conversions from `reference.wav`, a crop of that audio).
- **WER**: Whisper large-v3-turbo / Whisper base (the weaker model is more sensitive), language forced to English.
- **non-EN %**: 100 x (1 - Whisper turbo's English language probability), an accent proxy: the non-English mass of
  the accented speech goes mostly to Russian.
- **UTMOS**: predicted naturalness (UTMOS22 strong). The owner's own clips score 1.5-2.2 (phone-like recordings), so
  matching the timbre closely also lowers it.

| System | Target clip | ECAPA hold-out | ECAPA uk | WavLM-SV | WER % | non-EN % | UTMOS | RTF | VRAM peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OmniVoice from the Ukrainian clip (accented baseline) | uk 11 s | 0.630 | 0.777 | 0.953 | 0 / 2.4 | 1.68 | 3.10 | - | - |
| OmniVoice from `reference.wav` (accented baseline) | main 17.7 s | 0.650 | 0.642 | 0.973 | 0 / 0 | 3.07 | 3.13 | - | - |
| Native clip, no conversion | - | 0.041 | 0.008 | 0.606 | 0 / 0 | 0.05 | 4.47 | - | - |
| **Seed-VC v1, 25 steps** | main 17.7 s | **0.438** | 0.401 | 0.855 | 0 / 0 | 0.26 | 3.72 | 0.15 | 1.5 GB |
| Seed-VC v1, 25 steps | uk 11 s | 0.349 | 0.539 | 0.742 | 0 / 0 | 0.05 | 3.60 | 0.13 | 1.5 GB |
| Seed-VC v1, 10 steps | main 17.7 s | 0.364 | 0.318 | 0.837 | 0 / 0 | 0.11 | 3.74 | 0.09 | 1.5 GB |
| kNN-VC (WavLM-Large, prematched HiFi-GAN) | uk 11 s | 0.390 | 0.567 | 0.895 | 0 / 4.9 | 0.04 | 3.03 | 0.02 | 1.5 GB |
| kNN-VC | main 17.7 s | 0.358 | 0.303 | 0.893 | 0 / 4.1 | 0.01 | 3.13 | 0.01 | 1.7 GB |
| kNN-VC | uk + 43 s dataset* | 0.323 | 0.312 | 0.916 | 0 / 2.4 | 0.02 | 3.21 | 0.03 | 2.2 GB |
| OpenVoice V2 tone colour converter (tau 0.3) | uk 11 s | 0.192 | 0.211 | 0.751 | 2.5 / 0 | 0.01 | 3.91 | 0.03 | 0.4 GB |
| OpenVoice V2 | main 17.7 s | 0.117 | 0.083 | 0.771 | 0 / 0 | 0.01 | 3.80 | 0.01 | 0.4 GB |

\* that target contains the hold-out audio, so its hold-out score is not independent (and still lower).

**Choice: Seed-VC v1** (`DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned`, 25 diffusion steps (12 since, see
below), CFG 0.7, the
voice's `reference.wav` up to 20 s). The best independent speaker similarity (hold-out ECAPA 0.44, about two thirds of
the accented OmniVoice clone's 0.63-0.65 and ten times the unconverted native clip's 0.04), no word errors even for
Whisper base, about 10x less non-English mass than the accented baseline (0.05-0.26 % against 1.7-3.1 %), the most
natural output of the two converters that change the timbre, 1.5 GB of VRAM, RTF 0.15. kNN-VC matches it on the
Ukrainian clip's own ECAPA but generalizes worse to other recordings (hold-out 0.36-0.39), sounds rougher (UTMOS
3.0-3.2) and loses words for Whisper base (4-5 % WER), as expected with 11-60 s of target audio; OpenVoice V2 barely
moves the timbre (0.12-0.19). The longer `reference.wav` beat the 11 s clip on the hold-out (0.44 against 0.35), so the
converter uses the voice's main clip. Pitch needs no transpose: the model generates the reference's pitch (a 116 Hz
source came out at 121-128 Hz with the 126 Hz clip, 106-113 Hz with the 115 Hz one).

**End to end** (engine modules on Windows, `tts.generate` with `accent: "native-us"` for the untrained voice, one
3-sentence paragraph, 12.9 s of speech, 3 segments converted one by one while the next is synthesized):

| Run | First chunk | Total | Hold-out ECAPA | WER | non-EN % | UTMOS |
| --- | --- | --- | --- | --- | --- | --- |
| `accent: "own"` (OmniVoice from the Ukrainian `reference.wav`) | 9.0 s (first transcription of the clip) | 11.9 s | 0.69 | 0 | 0.74 | 2.77 |
| Native clip, conversion skipped | 3.6 s | 6.6 s | 0.05 | 0 | 0.02 | 4.41 |
| Native + Seed-VC, converter cold (process start 1.6 s + load 8.1 s) | 12.5 s | 14.8 s | - | 0 | 0.09 | - |
| Native + Seed-VC, warm | **2.8 s** | **6.6 s** | 0.36 | 0 | 1.08 | 3.12 |

One paragraph is a single sample: the accent proxy of the two converted runs (0.09 % and 1.08 %) shows the spread of
the sampling. `voices.warm` with `accent: "native-us"` loads the converter and the reference prompt in advance, so the
cold start does not reach the first chunk. The converter's worker holds 1.6 GB of VRAM while loaded (dropped after 10
idle minutes); with the converter busy, an OmniVoice segment took 2.5 s instead of 1.5 s on the shared GPU.

**Diffusion steps (12 since, 25 before).** The same machine and tooling, measured again with Re-speak in mind: Seed-VC
worker driven directly, the owner's `reference.wav` as the target, sources are OmniVoice English from the native male
clip. Speaker similarity is ECAPA (`spkrec-ecapa-voxceleb`) to two stretches of the owner's dataset audio outside
`reference.wav` (hold-out A: 17 s, B: 12 s); WER is Whisper turbo / Whisper base, language forced to English.

| Steps | Convert time, 4.1 s segment alone | Hold-out A, 8 clips (3-4.6 s) | Hold-out A, 28 clips | Hold-out A / B, 6 clips of 4-4.4 s, two runs | WER turbo / base |
| --- | --- | --- | --- | --- | --- |
| 25 | 0.89 s (RTF 0.22) | 0.346 | 0.314 | 0.362, 0.359 / 0.369, 0.384 | 0 / 0 |
| 16 | 0.61 s | 0.346 | - | 0.360 / 0.386 | 0 / 0 |
| **12** | **0.49 s** (RTF 0.12) | 0.329 | **0.338** | **0.390, 0.388 / 0.393, 0.390** | 0 / 0 |
| 10 | 0.43 s | 0.345 | - | 0.374 / 0.375 | 0 / 0 |
| 8 | 0.37 s (RTF 0.09) | 0.324 | - | - | 0 / 0 |

Fewer steps did not lower the similarity: every difference is within the spread of the diffusion sampling (standard
error about 0.01 on 28 clips, where 12 steps came out *above* 25). The earlier 10-against-25 gap (0.36 against 0.44)
was three sentences. The time matters most in Redub, where a conversion overlaps the synthesis of the next segment and
both slow down on the shared GPU. Re-speak pattern, OmniVoice 32 steps at factor 0.85 next to the converter, six
segments of about 4.2 s, per segment in the steady state:

| Converter | OmniVoice synthesis | Conversion | 6 segments |
| --- | --- | --- | --- |
| none (`accent: "own"`) | 1.53-1.64 s | - | 9.4 s |
| 25 steps | 2.5-2.6 s | 2.3 s | 15.3-16.7 s |
| 16 steps | 2.3 s | 1.5-1.6 s | 13.8 s |
| **12 steps** | **2.05-2.1 s** | **1.2-1.3 s** | **12.3-12.7 s** |
| 10 steps | 2.0 s | 1.1 s | 12.1 s |

So `quality: "best"` now converts with 12 steps (the native accent's extra time per segment goes from about 1.05 s to
0.5 s) and `fast` with 8. The first conversion after a load cost 0.2 s more alone and 0.8 s more next to OmniVoice;
`warm` now runs one short throw-away conversion so a real segment never pays for it.

**The slow first segments of the owner's Re-speak** (engine log: 27.2 s and 25.5 s for the first two OmniVoice
segments, about 2.9 s for the other 21; Seed-VC at 1.7-2.3 s per diffusion step instead of about 0.08 s while segment 1 was
converted). Measured and ruled out: model load (both workers were loaded and warm since `voices.warm` seven minutes
earlier; no reload in the log), the native clip's prompt (warm; 0.45 s when built), a CUDA warm-up or compile of
either model (the first 32-step synthesis after a 16-step warm-up took 1.45 s, the first conversion 1.05 s), and GPU
turn-taking (no lock on a 12 GB card, and segment 1 was synthesized while the converter was idle, still at 27 s).
Replaying the same pattern with both workers warm never came close (first segment 1.5 s, first conversion 3.1 s next
to OmniVoice). What the owner's run had in addition is the Whisper turbo transcription that ended 3 s before the
render (`redub.transcribe`, then `redub.render`), with OmniVoice, the converter and the RVC worker resident: the most
likely cause is that this pass pushed the idle workers' memory out (VRAM to shared memory, and on a host with little
free RAM their working sets to the page file) and the first segments paid to bring it back. That was not reproduced
here (it needs the card and the host RAM filled next to the running app). `accent: "own"` runs nothing of the
converter: no worker start, no warm-up, no conversion (`_Voice` only looks up the worker for `native-us`, and a respeak
without `zs` takes neither the GPU turn nor the conversion); its segments stay at about 1.55 s.

## Redub pace (slot fitting and the speech model's speed)

Re-speak and Translate & dub fit each phrase into the original timing by asking OmniVoice for shorter speech
(`duration_factor` < 1). How far that can go before words get lost: OmniVoice 32 steps, the native US male clip, four
sentences (54 words, short function words and weak final consonants), three takes per factor, every sentence
synthesized on its own like a Redub segment, Whisper turbo and Whisper base with the language forced to English.

| Factor | WER turbo | WER base | Speech length vs factor 1.0 | After Seed-VC (12 / 25 steps), WER turbo |
| --- | --- | --- | --- | --- |
| 1.0 | 0 % | 0.5 % | 1.00 | 0 / 0 % |
| 0.9 | 0 % | 0 % | 0.91 | 0 / 0 % |
| **0.85** | **0 %** | **0 %** | 0.86 | 0 / 0 % |
| 0.8 | 0 % | 0 % | 0.81 | 0 / 0 % |
| 0.75 | 2.0 % | 1.5 % | 0.77 | 0 / 0 % |
| 0.7 | 6.0 % | 6.0 % | 0.72 | 3 / 3 % |
| 0.65 | 19.9 % | 22.4 % | 0.66 | 24 / 24 % |

The model follows the factor closely (the length ratio tracks it), and when it has too little time it does not speak
faster everywhere: it drops the end of the phrase ("we took it at dusk" -> "we took it", "step around them" -> "step
around", "a hot meal and a good night's sleep" -> "a meal and a good night's"). The first lost words appear at 0.75;
at 0.65, a factor the old fitting used (its floor with the user's speed was 0.65), one word in five is gone. The
owner's 1.3x speed made every factor 0.77 x the fit, so most segments landed at 0.65-0.77.

**Floor: 0.85.** Slot fitting never asks for less than 0.85; the last clean factor was 0.80, and the margin covers the
length estimate (it is a running seconds-per-character average) and what Whisper does not count (weakened consonants
inside words). The user's speed is a pace preference applied first: 1.3x gives 0.77 on its own and fitting adds nothing
below that; at 1.0x fitting may go down to 0.85. Whatever still does not fit runs into the following silence, then
gets the ffmpeg stretch (at most 1.15x, which keeps every sound), then overruns and pushes the next segments later.
With "Keep original timing" off (`timing: "natural"`) nothing is fitted at all.

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

- About 30 GB for the runtime and the models: the engine environment with OmniVoice ~11 GB, Applio/RVC ~12 GB (two
  Python environments with their own PyTorch), Whisper and NLLB 600M for Redub ~4.5 GB, Hybrid Demucs for music removal
  ~0.3 GB, Seed-VC for the native accent of untrained voices ~2 GB. With an NVIDIA GPU setup adds NLLB 1.3B (2.7 GB in
  fp16, a 5.5 GB download): about 33 GB.
- OmniVoice needs about 4-5 GB of VRAM with the 7-12 s reference cut; the RVC worker adds ~1-2 GB, the zero-shot
  converter (Seed-VC, only while an untrained voice uses the native accent) ~1.6 GB. Below 10 GB of VRAM OmniVoice and
  the converter take turns instead of overlapping. 8 GB or more is
  recommended and 12 GB was used in testing.
