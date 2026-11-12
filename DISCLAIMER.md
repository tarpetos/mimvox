# Disclaimer and terms of use

> **This document is not legal advice.** See section 8.

Mimvox ("the app") is a desktop app that builds a synthetic copy of a human voice from a recording and
speaks text with it. A synthetic voice can be used to deceive people, so using the app comes with duties. Read this
document before you record anything.

## 1. Clone only your own voice, or a voice you have explicit permission for

- The app is made for cloning **your own voice**. Record only yourself.
- You may clone another person's voice only if that person has given you **explicit permission** for it, for the
  recording, for training a model from it, and for the way you will use the generated speech. Get it in writing where
  you can, and keep it. Permission can be withdrawn: if it is, stop using the voice and delete it.
- Never use audio you did not record for this purpose: not a call, a video, a podcast, a stream, a voice message or any
  other source of someone else's voice, whether or not it is public.
- Do not clone the voice of a child, or of a person who cannot give informed consent.
- Before the first recording the app shows this consent text and asks you to confirm that the voice you are about to
  record is your own, or that you have the speaker's explicit permission. Nothing is recorded before that. The
  acceptance is stored in your user data on your computer; when the consent text changes materially, the app asks again.

## 2. Prohibited use

You must not use the app, a voice made with it, or its output:

- to impersonate a real person, living or dead, or to make anyone believe that speech came from a person who did not
  say it;
- for fraud, scams, social engineering, phishing, fake voice messages or calls, fake endorsements, or to get around
  voice identification or authentication;
- to harass, threaten, defame, bully, exploit or deceive anyone, or to create content that sexualises a person or
  involves a minor;
- to spread false information, to influence elections, or to create content that is illegal where you or the listener
  live;
- to infringe copyright, personality or publicity rights, privacy or data-protection rights;
- for commercial purposes (see section 4);
- in high-risk settings that the model licence forbids (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), for
  example medical diagnosis, critical infrastructure, large-scale biometric surveillance or automated decisions about
  people;
- in any other way that breaks a law, a contract or a platform rule that applies to you.

If you publish or send generated speech, say clearly that it is synthetic wherever the law, the platform or common
decency expects it.

## 3. Your responsibility

- You decide whether, where and how you use the app. You are solely responsible for complying with every law,
  contract and policy that applies to you, to the person whose voice it is and to the people who hear the result,
  including laws on biometric data, voice and personality rights, deepfakes and synthetic media, privacy and data
  protection, consumer protection, and labelling of AI-generated content. These rules differ by country and by state
  and they change.
- A voice model and a reference recording are personal data, and in many places biometric data. You are the one who
  holds them. Keep them safe, do not share them without permission, and delete them when you no longer need them.
- The authors provide the app as it is, without any warranty, and accept no liability for how it is used or for
  what is made with it. Nothing in this document removes a responsibility that the law places on you, or a right that
  the law gives you and that cannot be excluded.

## 4. Non-commercial use only

- The app speaks with the OmniVoice model weights, licensed **CC BY-NC 4.0** (non-commercial, attribution), and uses
  other models with their own terms, among them the NLLB-200 translation model (CC BY-NC 4.0) and the Higgs Audio 2
  audio tokenizer inside OmniVoice (Boson Higgs Audio 2 Community License, with an acceptable-use policy). The app is
  therefore provided for **personal, non-commercial use**. Do not sell speech made with it, do not offer it as a
  service, and do not use it in a product.
- Those licences have more conditions than this summary; read them. The full list is in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and the licences control over this summary.
- Earlier versions used IndexTTS-2.5 (bilibili Model Use License Agreement). If an old `runtime/index-tts` folder is
  still on your computer, its licence and its authors' disclaimer still apply to it; setup offers to delete it.

## 5. Quality and limits

- Generated speech can contain mistakes: wrong words, odd intonation, skipped text, noise. It can sound very close to a
  real person. Check what you publish.
- A voice model made from poor recordings (noise, clipping, a different microphone for each part) sounds poor. The app
  checks every take and tells you, but it cannot guarantee a result.
- The measured similarity numbers in the documentation come from one speaker and one machine. Your results will differ.

## 6. Data

- The app runs on your computer. Your recordings, your voice models and the generated speech stay in the app's data
  folder on your computer and are **not uploaded** by the app. There is no account and no telemetry.
- Setup downloads the toolchains, the Python packages and the models from their public sources (for example GitHub,
  PyPI, Hugging Face and ModelScope). Those services see the download requests, under their own terms. After setup the
  app works offline.
- To remove a voice, delete it in the app (**Voices**), or delete its folder in the data folder (see
  `docs/architecture.md`). Delete the data folder to remove everything the app made.

## 7. Who provides the app

- Mimvox is made by [tarpetos](https://github.com/tarpetos), who holds all rights to it (see
  [LICENSE](LICENSE)). Questions about this document: open an issue on the project's GitHub page.
- The app is provided free of charge, as is, without warranty of any kind, and is not sold.
- You must be 18 or older, or the age of majority where you live, to use the app.

## 8. Not legal advice

This document explains how the app is meant to be used. It is not legal advice. If you are unsure whether you may
clone a voice, or use generated speech in a particular way, ask a lawyer who knows the laws that apply to you.
