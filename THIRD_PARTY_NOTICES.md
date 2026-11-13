# Third-party notices

Mimvox builds on the software and models below. This repository does **not** contain any of them: the
Rust crates are fetched by Cargo when the app is built, and setup (`setup.sh` / `setup.cmd`) downloads the Python
packages and the models onto your computer from their own sources, under their own licences.

The licences are taken from the actual sources (the LICENSE files in the upstream repositories and the licence field of
the upstream Hugging Face and crates.io pages, checked 2026-11-11). Where something could not be confirmed it is marked
**verify**. Nothing here is legal advice; the licence of each component controls.

**Summary of what limits you:** the OmniVoice model weights (CC BY-NC 4.0, because of their training data) are the
reason this project is for **personal, non-commercial use only**; its audio tokenizer adds the Boson Higgs Audio 2
Community License (attribution below). The NLLB-200 weights (600M and 1.3B) are CC BY-NC 4.0 too, and the Hybrid
Demucs weights state no licence but come from non-commercial training data (below). The rest is permissive (MIT,
Apache-2.0, BSD).

## Speech models and engine

### OmniVoice (code Apache-2.0, model weights CC BY-NC 4.0)

- Project: <https://github.com/k2-fsa/OmniVoice> (Next-gen Kaldi / k2-fsa), model `k2-fsa/OmniVoice` on Hugging Face,
  Python package `omnivoice` on PyPI. Paper: "OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion
  Language Models", arXiv:2604.00688.
- Licence: the **code is Apache-2.0**; the **pre-trained model is CC BY-NC 4.0** "due to constraints from its training
  data (e.g., Emilia)" (model card). Non-commercial use only, with attribution. This project uses the model unmodified
  and does not redistribute it: setup downloads it into `<data>/runtime/models/omnivoice`.
- The model's LLM backbone is initialised from **Qwen3-0.6B** (Apache-2.0).
- **Audio tokenizer** (`audio_tokenizer/` inside the OmniVoice model, the Higgs Audio 2 tokenizer by Boson AI): under
  the **Boson Higgs Audio 2 Community License Agreement** (its `LICENSE` file next to the weights; based on the Meta
  Llama 3 Community License). It asks products that use it to display the following notice, reproduced here:
  "Built with Higgs Materials licensed from Boson AI USA, Inc., Copyright Boson AI USA, Inc., All Rights Reserved and
  Meta Llama 3 licensed under the Meta Llama 3 Community License, Copyright Meta Platforms, Inc., All Right Reserved",
  and: "Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright © Meta Platforms, Inc. All Rights
  Reserved." / "Boson Higgs Audio 2 is licensed under the Boson Community License, Copyright © Boson AI USA, Inc. All
  Rights Reserved." Its acceptable-use policy applies, and its outputs may not be used to improve another large
  language model.
- Python dependencies installed into the engine environment with it: PyTorch (BSD-3-Clause), torchaudio (BSD-2-Clause),
  transformers and accelerate (Apache-2.0), librosa (ISC), soundfile, NumPy and SciPy (BSD-3-Clause), num2words (LGPL-2.1,
  used unmodified as a library for number spelling), gradio (Apache-2.0, a dependency of the package, not used), and
  others (**verify** with `pip-licenses` in the installed environment).

### Whisper and NLLB-200 (Redub, reference transcripts)

- **OpenAI Whisper** (`openai-whisper` package and the `large-v3-turbo` checkpoint): MIT. Used for Redub transcription
  and once per reference clip to cut it at sentence boundaries for OmniVoice.
- **NLLB-200 distilled 600M** (`facebook/nllb-200-distilled-600M`, Meta): **CC BY-NC 4.0** (model card). Used for Redub
  translation. Non-commercial only.
- **NLLB-200 distilled 1.3B** (`facebook/nllb-200-distilled-1.3B`, Meta): **CC BY-NC 4.0** (model card and the licence
  field of the Hugging Face repo, checked 2026-11-12). The optional "better translation" model of Redub, downloaded by
  setup (by default only with an NVIDIA GPU). Setup stores the weights in half precision (fp16 safetensors) on your
  computer; this changes only the number format, and the project does not redistribute them. Non-commercial only.
- **g2p_en** (Apache-2.0) with NLTK data (CMUdict, BSD-style), for the English recording-text analysis.
- **ukrainian-word-stress** (lang-uk, MIT) with the stress dictionary in its package, and **marisa-trie** (MIT; the
  bundled libmarisa is BSD-2-Clause or LGPL-2.1-or-later), for the optional automatic Ukrainian word stress in the
  OmniVoice text front-end. Installed without its optional Stanza dependency.
- **pypdf** 6.21.4 (BSD-3-Clause, the `LICENSE` file of the package; <https://github.com/py-pdf/pypdf>), for reading
  the text layer of PDF books in the audiobook import. Installed without its optional extras. The other book formats
  (EPUB, FB2, DOCX, ODT, RTF, HTML, Markdown, subtitles) are read with the Python standard library.

### Hybrid Demucs (Remove background music)

- **Model code:** `torchaudio.models.hdemucs_high` in torchaudio (BSD-2-Clause, the `LICENSE` of `pytorch/audio`), a port
  of Hybrid Demucs from `facebookresearch/demucs` (MIT, Copyright (c) Meta Platforms, Inc. and affiliates). Paper:
  Défossez, "Hybrid Spectrogram and Waveform Source Separation", arXiv:2111.03600.
- **Weights:** torchaudio's `HDEMUCS_HIGH_MUSDB_PLUS` bundle, the file `hdemucs_high_trained.pt` from
  `download.pytorch.org/torchaudio/models/` (335 MB), which setup downloads into `<data>/runtime/models/demucs` (checked
  by SHA-256; this project does not redistribute it). Neither the torchaudio documentation of the bundle nor the
  download states a licence for the weights. The torchaudio README says that its pre-trained models "may have their own
  licenses or terms and conditions derived from the dataset used for training" and that it is the user's
  responsibility to determine whether they may use them. Per the torchaudio documentation the weights were trained on
  the training and test sets of **MUSDB18-HQ** plus 150 songs from an internal database produced for Meta. MUSDB18-HQ is
  provided for educational and research purposes only, without commercial use unless the copyright holders allow it
  (its record on Zenodo; part of it is CC BY-NC-SA 4.0). Treat the weights as **non-commercial**, which matches the
  rest of this project (**verify** before any other use).

### IndexTTS and IndexTTS-2.5 (legacy: used by earlier versions, no longer installed)

Earlier versions of this app spoke through IndexTTS-2.5; it was replaced by OmniVoice. Setup no longer downloads it and
offers to delete an existing `<data>/runtime/index-tts`. The notice below applies only to such an older install that is
still on disk.

- Project: <https://github.com/index-tts/index-tts> (code and model weights), model: `IndexTeam/IndexTTS-2.5` on
  Hugging Face and ModelScope. Developed by the Bilibili IndexTTS Team.
- Licence: **bilibili Model Use License Agreement** (the `LICENSE` file of the repository and of the model; the Hugging
  Face page lists it as `other`, `bilibili-model-license`). It is not an OSI licence. Points that matter here (read
  the full text; the Chinese version prevails over the English one, section 9):
  - it grants a worldwide, non-exclusive, non-transferable, royalty-free limited licence to use the model and
    derivative works (section 2.1), including derivatives made by fine-tuning, and a **derivative work** includes the
    model's outputs and works based on them (section 1.5);
  - a company whose products had more than 100 million monthly active users, or whose annual revenue exceeded RMB 1
    billion, needs a separate licence (section 2.2);
  - the model or any derivative work may not be used to improve any AI model except IndexTTS itself, its derivative
    works or **non-commercial** AI models (section 3.4 c);
  - uses that break the law of the place where the output or the model is generated or used are forbidden, and so are
    high-risk uses such as medical diagnosis, autonomous driving, military applications, critical-infrastructure
    control, large-scale biometric surveillance and automated decision-making about people (sections 4.1 c, 4.2);
  - whoever distributes a derivative work must keep the notices and a copy of the licence, and state that modifications
    are not endorsed or warranted by the original right-holder (sections 3.4 b, 4.1 a);
  - the right-holder may revoke the licence on breach, and the agreement is governed by the law of the People's
    Republic of China with arbitration in Shanghai (sections 5.1, 6).
- Separate **DISCLAIMER** in the repository (Chinese, by the Bilibili Index Team, dated 2025-03-17): it restricts
  the use of the project to research, study and lawful creative work, and lists as prohibited, among other things,
  synthesising political or public figures' voices or any unauthorised person's voice, fraud and identity theft,
  spreading false information, infringing rights, deepfake-law violations, and using synthesized voices for commercial
  purposes without authorisation.
- **Commercial use: this project does not offer it.** The IndexTTS README directs commercial use and cooperation to
  indexspeech@bilibili.com.
- Setup applies small local patches to the IndexTTS code (a configurable reference length and diffusion settings) and
  keeps the upstream licence files in the downloaded copy. Modifications made by this project are not endorsed,
  warranted or guaranteed by the original right-holder, and the original right-holder disclaims all liability related
  to them.
- Third-party code inside IndexTTS: its README acknowledges tortoise-tts, XTTSv2 (Coqui), BigVGAN, wenet, icefall,
  MaskGCT (Amphion) and **Seed-VC** (`Plachtaa/seed-vc`, GPL-3.0 on GitHub). The IndexTTS repository ships a single
  licence file and the s2mel part is derived from those works; **verify** how their licences combine before you ever
  redistribute IndexTTS. This project redistributes none of it.
- Citation: IndexTTS 2.5 technical report, arXiv:2601.03888; IndexTTS2, arXiv:2506.21619; IndexTTS, arXiv:2502.05512.

### Auxiliary models downloaded for IndexTTS (legacy)

| Model | Source | Licence | Notes |
| --- | --- | --- | --- |
| **w2v-bert-2.0** (semantic features) | `facebook/w2v-bert-2.0` (Meta) | MIT (model card) | |
| **MaskGCT semantic codec** | `amphion/MaskGCT` (Amphion, OpenMMLab) | **CC BY-NC 4.0** (model card); the Amphion code is MIT | Non-commercial only. Used by IndexTTS as the semantic codec (`semantic_codec`). |
| **CAM++ speaker encoder** (`campplus_cn_common.bin`) | `funasr/campplus` on Hugging Face, from `iic/speech_campplus_sv_zh-cn_16k-common` on ModelScope (3D-Speaker / Alibaba DAMO) | Apache-2.0 (Hugging Face card) | Upstream 3D-Speaker repository licence: **verify**. |
| **BigVGAN v2** (vocoder, `nvidia/bigvgan_v2_22khz_80band_256x`) | NVIDIA | MIT (model card and `NVIDIA/BigVGAN` repository) | |
| **Qwen emotion model** (`qwen0.6bemo4-merge`, shipped with IndexTTS-2.5) | IndexTeam, based on Qwen | **verify** (Qwen3-0.6B is Apache-2.0; the fine-tune falls under the IndexTTS-2.5 licence) | Downloaded with the checkpoints, **not loaded** by this app (the emotion follows the reference clip). |

The IndexTTS-2.5 checkpoint files themselves (`gpt.pth`, `s2mel.pth`, `codec.pth`, `feat1.pt`, `feat2.pt`,
`wav2vec2bert_stats.pt`, the tokenizer files) are under the bilibili Model Use License Agreement above.

## Voice conversion

### Applio (MIT)

- <https://github.com/IAHispano/Applio>. Copyright (c) 2026 AI Hispano. MIT License (the repository's `LICENSE`).
- Used for RVC dataset preprocessing, feature extraction, training and inference (as a separate worker process
  in its own Python environment). The Applio repository also has a `TERMS_OF_USE.md`: use only audio that you own or
  have permission to use, do not use it to harm, defame or deceive, and obey local law. This project's
  [DISCLAIMER.md](DISCLAIMER.md) is at least as strict.
- Applio's downloadable prerequisites (from `IAHispano/Applio` on Hugging Face, which lists MIT) are fetched by
  setup: the **HiFi-GAN pretrained generator and discriminator** weights (`f0G40k.pth`, `f0D40k.pth` and the other
  sample rates), the RefineGAN pretrained weights, the predictors (`rmvpe.pt`, `fcpe.pt`, `swift.onnx`), and
  `ffmpeg` / `ffprobe` executables on Windows. **Verify** the licence of each bundled binary and checkpoint (the FFmpeg
  builds are LGPL or GPL depending on the build; this project does not redistribute them).
- Python dependencies installed into the Applio environment include PyTorch (BSD-3-Clause), torchaudio, torchcrepe,
  torchfcpe, faiss-cpu (MIT), librosa (ISC), SciPy (BSD-3-Clause), NumPy (BSD-3-Clause), soundfile (BSD-3-Clause),
  transformers (Apache-2.0), onnxruntime (MIT) and others listed in Applio's `requirements-engine.txt`; each under its
  own licence (**verify** the exact versions with `pip-licenses` in the installed environment).

### Seed-VC (GPL-3.0; zero-shot native accent)

- <https://github.com/Plachtaa/seed-vc>, pinned to commit `51383efd921027683c89e5348211d93ff12ac2a8`. Copyright (c)
  Plachtaa and contributors. **GNU General Public License v3.0** (the repository's `LICENSE`). The model weights
  `Plachta/Seed-VC` on Hugging Face (`DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` and its config,
  revision `257283f9f41585055e8f858fba4fd044e5caed6e`) are also marked GPL-3.0.
- Used for the native accent of untrained voices: OmniVoice speaks the English from the native clip and Seed-VC gives
  it the timbre of the voice's reference clip (engine `zsvc.py`, worker `zsvc_worker.py`). **Not part of this
  repository and not shipped with it**: setup fetches the source checkout into `<data>/runtime/seed-vc` from its own
  repository and the weights into `<data>/runtime/models/seed-vc` on the user's computer, as an optional runtime
  component (`setup --no-zero-shot-vc` leaves it out; the app then offers the native accent to trained voices only).
  Mimvox's worker imports Seed-VC's modules at run time in a separate process. The GPL applies to whoever conveys
  Seed-VC (or a work based on it): anyone who distributes a bundle that contains Seed-VC must do so under the GPL-3.0,
  which also covers the parts of Mimvox combined with it in that bundle; this project's PolyForm Noncommercial licence
  is not compatible with that, so a Mimvox package must not include Seed-VC (or its weights). Downloading and running
  it on your own computer is private use, which the GPL permits without conditions.
- Weights it uses, downloaded by setup from their own repositories at pinned revisions: **BigVGAN v2** 22 kHz vocoder
  (`nvidia/bigvgan_v2_22khz_80band_256x`, MIT, revision `633ff708ed5b74903e86ff1298cf4a98e921c513`), the
  **Whisper-small** encoder (`openai/whisper-small`, Apache-2.0, revision `973afd24965f72e36ca33b3055d56a652f456b4d`),
  and **CAM++** (`campplus_cn_common.bin` from `funasr/campplus`, Apache-2.0, shipped inside the Seed-VC checkout).
- Python packages setup adds to the engine environment for it: `munch` 4.0.0 (MIT) and `einops` 0.8.0 (MIT).

### RVC (Retrieval-based Voice Conversion) project (MIT)

- <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI>, MIT License. The method, the model code that
  Applio builds on, and the **RMVPE** pitch extractor (`rmvpe.pt`, originally published through `lj1995/VoiceConversionWebUI`,
  MIT on Hugging Face) come from it.
- **ContentVec** embedder (`pytorch_model.bin`, the `contentvec` model; the Hugging Face copy `lengyue233/content-vec-best`
  is MIT). Original work: ContentVec by Qian et al. (ICML 2022), <https://github.com/auspicious3000/contentvec>
  (MIT, **verify**).

## Desktop app

### Rune (MIT)

- <https://github.com/anomalyco/rune>, pinned to revision `a7cea1692a8ce68b1b63c46688c493bb94a8456d`. Copyright (c)
  2026 Anomaly. MIT License. The UI framework the app is built on; its own dependencies (graphics, windowing, text
  layout and others, see its `Cargo.lock`) come with it under their own licences, mostly MIT and Apache-2.0
  (**verify** with `cargo about` or `cargo deny` before a release).

### Direct Rust dependencies (from `app/Cargo.toml`)

| Crate | Licence (crates.io) |
| --- | --- |
| `cpal` (microphone capture and playback) | Apache-2.0 |
| `hound` (WAV reading and writing) | Apache-2.0 |
| `serde`, `serde_json` | MIT OR Apache-2.0 |
| `anyhow` | MIT OR Apache-2.0 |
| `toml` | MIT OR Apache-2.0 |
| `dirs` | MIT OR Apache-2.0 |

All transitive Rust dependencies are listed in `app/Cargo.lock`. Generate the complete licence list of a release build
with `cargo about` (or `cargo license`) and check it before distributing binaries.

## Fonts and icons

The app icon is made for this project (`tools/release/`).

### Noto Sans (SIL Open Font License 1.1)

- Files: `NotoSans-Regular.ttf` and `NotoSans-Bold.ttf` in `app/assets/fonts/`, with the licence text in
  `app/assets/fonts/OFL.txt`. From the Noto project.
- Copyright 2022 The Noto Project Authors (<https://github.com/notofonts/latin-greek-cyrillic>).
- Licence: **SIL Open Font License, Version 1.1** (<https://scripts.sil.org/OFL>). The fonts are bundled unmodified
  with the app; the licence allows that, including in commercial products, as long as the fonts are not sold on their
  own and the copyright notice and licence text travel with them (hence `OFL.txt` next to the files). The Reserved Font
  Name rules apply if the fonts are ever modified.

## Where the full licence texts are

After setup, each downloaded component keeps its own licence files in its folder under `runtime/` (for example
`runtime/applio/LICENSE`, `runtime/models/omnivoice/audio_tokenizer/LICENSE`, and the package metadata in
`runtime/engine/.venv`) and next to its model files. For the Rust crates, see the
sources fetched by Cargo. If a notice here is wrong or missing, open an issue on the project's GitHub page.

### CMU ARCTIC speech database (demo samples, native-accent reference clips)

- Files: `docs/samples/reference.mp3` (five recordings of speaker `slt` joined into one clip, so modified) and the
  samples generated from it in `docs/samples/`.
- Files: `engine/src/mimvox_engine/assets/native/en-us-male.wav` (speaker `bdl`, US English male) and
  `en-us-female.wav` (speaker `slt`, US English female), each four recordings (`arctic_a0029`, `arctic_a0039`,
  `arctic_a0187`, `arctic_b0063`) trimmed, filtered, levelled, joined with short pauses and resampled to 24 kHz, so
  **modified**; the transcripts next to them are the corpus prompts. Shipped with the app as the native-accent
  references (made by `tools/release/make_native_refs.py`); the licence travels with them as
  `assets/native/CMU_ARCTIC_COPYING.txt`.
- Copyright (c) 2003 Carnegie Mellon University. Free for any use under the licence in
  `docs/samples/CMU_ARCTIC_COPYING.txt` (keep the notice, mark modifications, keep the authors' names).
  <http://festvox.org/cmu_arctic/>
