#!/usr/bin/env python3
"""Rebuild the native-accent reference clips in engine/src/mimvox_engine/assets/native/ from CMU ARCTIC.

For each US English speaker (``bdl`` male, ``slt`` female) the same four sentences are read from the original 32 kHz
recordings (channel 0 of the WAVEGG archives; only those members are fetched, with HTTP range requests), trimmed,
high-passed at 50 Hz, levelled to the same loudness, joined with 0.25 s pauses, resampled to 24 kHz mono 16-bit and
written as ``en-us-<gender>.wav`` with the transcript in ``en-us-<gender>.txt``. ``voices.json`` gets each clip's
median F0, measured with the engine's own ``accent.median_f0``.

The joined clips are modifications of the CMU ARCTIC recordings (licence: CMU_ARCTIC_COPYING.txt next to them).

Usage (numpy, scipy, soundfile and librosa needed, for example):
    uv run --no-project --python 3.11 --with numpy,scipy,soundfile,librosa python tools/release/make_native_refs.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import urllib.request
import zipfile

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "engine", "src"))
from mimvox_engine import accent  # noqa: E402

BASE = "http://festvox.org/cmu_arctic/cmu_arctic"
SPEAKERS = {"male": "bdl", "female": "slt"}
SENTENCES = ["arctic_a0029", "arctic_a0039", "arctic_a0187", "arctic_b0063"]
SR_IN, SR_OUT = 32000, 24000
PAUSE_S, LEAD_S, TAIL_S = 0.25, 0.10, 0.15
TARGET_RMS_DB = -20.0   # of each sentence's speech
PEAK = 0.9


class HttpFile(io.RawIOBase):
    """A seekable read-only view of a remote file through HTTP range requests (for zipfile)."""

    def __init__(self, url: str) -> None:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.url, self.pos = url, 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b) -> int:  # type: ignore[no-untyped-def]
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def transcripts(spk: str) -> dict[str, str]:
    with urllib.request.urlopen(f"{BASE}/cmu_us_{spk}_arctic/etc/txt.done.data", timeout=60) as r:
        raw = r.read().decode("utf-8")
    return dict(re.findall(r'\( (\S+) "(.*)" \)', raw))


def sentence(z: zipfile.ZipFile, spk: str, utt: str) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(z.read(f"cmu_us_{spk}_arctic/orig/{utt}.wav")), dtype="float32", always_2d=True)
    assert sr == SR_IN, sr
    y = sosfiltfilt(butter(4, 50.0, "highpass", fs=sr, output="sos"), x[:, 0]).astype(np.float32)  # ch 1 is the EGG
    _, (a, b) = librosa.effects.trim(y, top_db=35, frame_length=1024, hop_length=256)
    y = y[max(0, a - int(0.03 * sr)):b + int(0.05 * sr)]
    frames = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    speech = frames[frames > frames.max() * 0.1]
    return y * (10 ** (TARGET_RMS_DB / 20.0) / float(np.sqrt(np.mean(speech ** 2))))


def main() -> int:
    out_dir = accent.NATIVE_DIR
    os.makedirs(out_dir, exist_ok=True)
    manifest: dict = {"corpus": "CMU ARCTIC 0.95 (http://festvox.org/cmu_arctic/), Copyright (c) 2003 Carnegie Mellon "
                                "University; modified: four recordings trimmed, levelled and joined",
                      "sentences": SENTENCES, "voices": {}}
    for gender, spk in SPEAKERS.items():
        texts = transcripts(spk)
        z = zipfile.ZipFile(io.BufferedReader(HttpFile(f"{BASE}/orig/cmu_us_{spk}_arctic-WAVEGG.zip"), 1 << 20))
        gap = np.zeros(int(PAUSE_S * SR_IN), dtype=np.float32)
        parts: list[np.ndarray] = [np.zeros(int(LEAD_S * SR_IN), dtype=np.float32)]
        for k, utt in enumerate(SENTENCES):
            parts += ([gap] if k else []) + [sentence(z, spk, utt)]
        parts.append(np.zeros(int(TAIL_S * SR_IN), dtype=np.float32))
        x = np.concatenate(parts)
        x *= min(1.0, PEAK / float(np.max(np.abs(x))))
        y = librosa.resample(x, orig_sr=SR_IN, target_sr=SR_OUT)
        name = f"en-us-{gender}"
        sf.write(os.path.join(out_dir, name + ".wav"), np.clip(y, -1, 1), SR_OUT, subtype="PCM_16")
        text = " ".join(texts[u] for u in SENTENCES)
        with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
        f0 = accent.median_f0(y, SR_OUT)
        manifest["voices"][gender] = {"file": name + ".wav", "text_file": name + ".txt", "speaker": spk,
                                      "f0_median": round(float(f0 or 0.0), 1), "duration_s": round(len(y) / SR_OUT, 2)}
        print(f"{name}: {len(y) / SR_OUT:.2f} s, median F0 {f0:.1f} Hz: {text}")
    with open(os.path.join(out_dir, "voices.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
