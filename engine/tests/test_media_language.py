import asyncio
import json

import numpy as np
import pytest
import soundfile as sf
from mimvox_engine import config, speech, voicebundle, voices
from mimvox_engine.dataset import media_ref
from mimvox_engine.registry import EngineError


class Ctx:
    def __init__(self, data_dir):
        self.data_dir = str(data_dir)
        self.runtime = {}


class FakeWorker:
    def __init__(self, answers=None, error=None):
        self.answers = answers or {}
        self.error = error
        self.calls = []

    async def detect_language(self, npy, windows, allow_cuda=None):
        self.calls.append({"npy": npy, "windows": windows, "samples": len(np.load(npy)), "allow_cuda": allow_cuda})
        if self.error is not None:
            raise self.error
        return {
            "device": "cpu",
            "windows": [{"start": a, "end": b, "probs": self.answers(k)} for k, (a, b) in enumerate(windows)],
        }


def probs(top, p=0.9, rest="en"):
    return {top: p, rest: round(1.0 - p, 4)}


@pytest.fixture
def data(tmp_path):
    d = config.DataDir(str(tmp_path / "data"))
    d.ensure()
    return d


@pytest.fixture
def whisper(data):
    path = speech.asr_path(data.root)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"model")
    return path


@pytest.fixture
def worker(monkeypatch):
    def install(answers=None, error=None):
        w = FakeWorker(answers, error)
        monkeypatch.setattr(speech, "get", lambda runtime, root: w)
        return w

    monkeypatch.setattr(media_ref, "_speakable", lambda code: code != "xx")
    return install


def pool(seconds):
    return np.zeros(int(seconds * media_ref.SR), dtype=np.int16)


def test_a_short_pool_is_one_window():
    assert media_ref.language_windows(12.0, 5) == [(0.0, 12.0)]


def test_windows_spread_from_start_to_end():
    wins = media_ref.language_windows(600.0, 5)
    assert len(wins) == 5
    assert wins[0] == (0.0, 30.0)
    assert wins[-1] == (570.0, 600.0)
    assert all(b - a == pytest.approx(30.0) for a, b in wins)


def test_no_more_windows_than_the_pool_fills():
    assert len(media_ref.language_windows(45.0, 5)) == 2
    assert media_ref.language_windows(600.0, 1) == [(285.0, 315.0)]


def test_votes_over_every_window_not_the_first():
    windows = [
        {"start": 0.0, "end": 30.0, "probs": probs("es", 0.7)},
        {"start": 30.0, "end": 60.0, "probs": probs("zh")},
        {"start": 60.0, "end": 90.0, "probs": probs("zh")},
    ]
    code, share = media_ref.vote_language(windows)
    assert code == "zh"
    assert share == pytest.approx(0.6)


def test_a_window_counts_by_the_seconds_it_heard():
    windows = [
        {"start": 0.0, "end": 30.0, "probs": probs("de")},
        {"start": 30.0, "end": 35.0, "probs": probs("fr", 1.0)},
    ]
    assert media_ref.vote_language(windows)[0] == "de"


def test_whisper_aliases_are_normalised():
    assert media_ref.vote_language([{"start": 0, "end": 10, "probs": {"cmn": 0.8, "ja": 0.2}}])[0] == "zh"


def test_no_language_when_unsure_or_unspoken():
    split = [
        {"start": 0.0, "end": 30.0, "probs": {"es": 0.4, "pt": 0.35, "it": 0.25}},
        {"start": 30.0, "end": 60.0, "probs": {"pt": 0.4, "es": 0.35, "it": 0.25}},
    ]
    assert media_ref.pick_language(split, lambda c: True) is None
    sure = [{"start": 0.0, "end": 30.0, "probs": probs("zh")}]
    assert media_ref.pick_language(sure, lambda c: True) == "zh"
    assert media_ref.pick_language(sure, lambda c: c != "zh") is None
    assert media_ref.pick_language([], lambda c: True) is None


def test_detects_over_several_windows_of_the_pool(data, whisper, worker):
    w = worker(lambda k: probs("es", 0.6) if k == 0 else probs("zh"))
    code = asyncio.run(media_ref.detect_language(Ctx(data.root), pool(300.0), True))
    assert code == "zh"
    call = w.calls[0]
    assert len(call["windows"]) == media_ref.LANG_WINDOWS_GPU
    assert call["samples"] == media_ref.LANG_WINDOWS_GPU * 30 * media_ref.ASR_SR
    assert call["allow_cuda"] is True
    assert not call["npy"].exists()


def test_fewer_windows_without_a_gpu(data, whisper, worker):
    w = worker(lambda k: probs("zh"))
    asyncio.run(media_ref.detect_language(Ctx(data.root), pool(300.0), False))
    assert len(w.calls[0]["windows"]) == media_ref.LANG_WINDOWS_CPU


def test_no_language_without_whisper_or_when_it_fails(data, worker):
    w = worker(lambda k: probs("zh"))
    assert asyncio.run(media_ref.detect_language(Ctx(data.root), pool(60.0), True)) is None
    assert w.calls == []
    path = speech.asr_path(data.root)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"model")
    w = worker(error=speech.SpeechError("internal", "boom"))
    assert asyncio.run(media_ref.detect_language(Ctx(data.root), pool(60.0), True)) is None
    assert not list(data.tmp.glob("language-*.npy"))


def new_voice(data, language="es"):
    return voices.voices_create(Ctx(data.root), {"name": "Song", "language": language})["voice"]["id"]


def test_an_empty_voice_takes_the_language_heard(data):
    vid = new_voice(data)
    assert voices.adopt_language(data, vid, "zh")
    assert voices.read_meta(data, vid)["language"] == "zh"
    assert voices.describe(data, vid)["language"] == "zh"


def test_a_voice_with_recordings_keeps_its_language(data):
    vid = new_voice(data)
    (data.voice(vid) / "dataset" / "part1.wav").write_bytes(b"")
    assert not voices.adopt_language(data, vid, "zh")
    assert voices.read_meta(data, vid)["language"] == "es"


def speech_like(path, seconds=8.0, sr=44100):
    rng = np.random.default_rng(1)
    t = np.arange(int(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140.0, 280.0, 420.0, 700.0), start=1))
    gate = (np.sin(2 * np.pi * 3.0 * t) > -0.2).astype(np.float32)
    y = 0.25 * tone * gate + 0.0005 * rng.standard_normal(len(t))
    sf.write(path, y.astype(np.float32), sr, subtype="PCM_16")
    return path


def combine(data, tmp_path, **params):
    files = [str(speech_like(tmp_path / "clip.wav"))]
    return asyncio.run(media_ref.combine(Ctx(data.root), {"paths": files, "remove_music": False, **params}))


def test_combine_sets_a_new_voices_language(data, tmp_path, whisper, worker):
    worker(lambda k: probs("zh"))
    vid = new_voice(data)
    res = combine(data, tmp_path, voice_id=vid, language="auto")
    assert (res["language"], res["language_set"]) == ("zh", True)
    assert voices.read_meta(data, vid)["language"] == "zh"


def test_combine_detects_only_when_asked(data, tmp_path, whisper, worker):
    w = worker(lambda k: probs("zh"))
    vid = new_voice(data)
    res = combine(data, tmp_path, voice_id=vid)
    assert (res["language"], res["language_set"]) == (None, False)
    assert w.calls == []
    assert voices.read_meta(data, vid)["language"] == "es"
    with pytest.raises(EngineError):
        combine(data, tmp_path, voice_id=vid, language="es")


def test_combine_leaves_a_voice_with_recordings_alone(data, tmp_path, whisper, worker):
    w = worker(lambda k: probs("zh"))
    vid = new_voice(data)
    (data.voice(vid) / "dataset" / "part1.wav").write_bytes(b"")
    res = combine(data, tmp_path, voice_id=vid, language="auto")
    assert (res["language"], res["language_set"]) == (None, False)
    assert w.calls == []


def test_a_bundle_keeps_a_language_heard_in_media():
    raw = json.dumps({"name": "Song", "language": "zh"}).encode()
    assert voicebundle._clean_meta(raw)["language"] == "zh"
    with pytest.raises(EngineError):
        voicebundle._clean_meta(json.dumps({"name": "Song", "language": "not a code"}).encode())
