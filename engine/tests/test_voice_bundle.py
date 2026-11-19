import hashlib
import io
import json
import os
import zipfile

import numpy as np
import pytest
import soundfile as sf
from mimvox_engine import config, reflang, voicebundle, voices
from mimvox_engine.registry import EngineError

SIG_NS = 1_700_000_000_000_000_000


class Ctx:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.runtime = {}

    def emit(self, event, data, job=None):
        pass


def wav(path, seconds=1.0, freq=200.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(int(16000 * seconds)) / 16000
    sf.write(path, (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32), 16000, subtype="PCM_16")
    os.utime(path, ns=(SIG_NS, SIG_NS))
    side = os.path.splitext(path)[0] + reflang.SIDECAR_SUFFIX
    with open(side, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "sig": [SIG_NS, os.path.getsize(path)], "start": 0, "end": seconds, "text": "hi",
                   "language": "en"}, fh)


def make_voice(root, vid="anna", rvc=True):
    data = config.DataDir(str(root))
    data.ensure()
    vdir = data.voice(vid)
    os.makedirs(os.path.join(vdir, "dataset"))
    meta = {"id": vid, "name": "Anna", "created": "2026-01-02T03:04:05Z", "preset": None, "language": "en",
            "reference": "reference.wav", "training_paused": {"epoch": 3, "total_epochs": 9}}
    if rvc:
        meta.update(trained=True, trained_epochs=300, trained_at="2026-01-03T10:00:00")
        os.makedirs(os.path.join(vdir, "rvc"))
        with open(os.path.join(vdir, "rvc", "model.pth"), "wb") as fh:
            fh.write(b"pth" * 1000)
        with open(os.path.join(vdir, "rvc", "model.index"), "wb") as fh:
            fh.write(b"idx" * 500)
    with open(os.path.join(vdir, "voice.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    wav(os.path.join(vdir, "reference.wav"))
    wav(os.path.join(vdir, "reference_uk.wav"), 0.5, 300.0)
    wav(os.path.join(vdir, "dataset", "part1.wav"), 2.0)
    with open(os.path.join(vdir, "dataset", "meta.json"), "w") as fh:
        fh.write("{}")
    wav(os.path.join(vdir, "reference_media.wav"), 2.0)
    return data


def export(data, vid="anna", dest=None):
    return voices.voices_export(Ctx(data.root), {"id": vid, **({"dest": dest} if dest else {})})


def import_bundle(data, path, **kw):
    return voices.voices_import(Ctx(data.root), {"bundle": path, **kw})["voice"]


def rewrite(src, dst, change):
    """Copies the zip, letting `change(name, bytes)` return new (name, bytes) or None to drop the member."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for info in zin.infolist():
            out = change(info.filename, zin.read(info))
            if out is not None:
                zout.writestr(zipfile.ZipInfo(out[0]), out[1])
    return str(dst)


def rebuild(src, dst, members):
    """Copies the zip with `members` (name -> bytes) replaced or added, and a manifest that matches them."""
    with zipfile.ZipFile(src) as zin:
        manifest = json.loads(zin.read("manifest.json"))
        body = {i.filename: zin.read(i) for i in zin.infolist() if i.filename != "manifest.json"}
    body.update(members)
    manifest["files"] = [{"path": n, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                         for n, b in sorted(body.items())]
    with zipfile.ZipFile(dst, "w") as zout:
        for n, b in [*sorted(body.items()), ("manifest.json", json.dumps(manifest).encode())]:
            zout.writestr(n, b)
    return str(dst)


def test_export_holds_what_speaks_and_nothing_else(tmp_path):
    data = make_voice(tmp_path / "a")
    res = export(data)
    assert os.path.dirname(res["path"]) == os.path.join(data.tmp, "export")
    assert os.path.basename(res["path"]) == f"anna-{res['sha256'][:12]}.mimvoice"
    with open(res["path"], "rb") as fh:
        body = fh.read()
    assert hashlib.sha256(body).hexdigest() == res["sha256"] and len(body) == res["bytes"]
    names = ["reference.omnivoice.json", "reference.wav", "reference_uk.omnivoice.json", "reference_uk.wav",
             "rvc/model.index", "rvc/model.pth", "voice.json"]
    assert [f["path"] for f in res["files"]] == names
    with zipfile.ZipFile(res["path"]) as zf:
        assert zf.namelist() == [*names, "manifest.json"]
        manifest = json.loads(zf.read("manifest.json"))
        meta = json.loads(zf.read("voice.json"))
    assert manifest["format"] == "mimvox-voice" and manifest["version"] == 1 and manifest["id"] == "anna"
    assert manifest["name"] == "Anna" and manifest["files"] == res["files"]
    assert set(manifest) == {"format", "version", "id", "name", "files"}
    assert "training_paused" not in meta


def test_two_exports_have_the_same_hash(tmp_path):
    data = make_voice(tmp_path / "a")
    first = export(data)
    os.utime(os.path.join(data.voice("anna"), "rvc", "model.pth"))
    second = export(data, dest=str(tmp_path / "out" / "copy.mimvoice"))
    assert second["path"] == str(tmp_path / "out" / "copy.mimvoice")
    assert first["sha256"] == second["sha256"] and first["bytes"] == second["bytes"]


def test_engine_version_does_not_change_the_hash(tmp_path, monkeypatch):
    data = make_voice(tmp_path / "a")
    monkeypatch.setattr(voicebundle, "__version__", "0.1.0")
    first = export(data)
    monkeypatch.setattr(voicebundle, "__version__", "9.9.9")
    second = export(data)
    assert first["engine"] == "0.1.0" and second["engine"] == "9.9.9"
    assert first["sha256"] == second["sha256"]


def test_round_trip(tmp_path):
    src = make_voice(tmp_path / "a")
    res = export(src)
    dst = config.DataDir(str(tmp_path / "b"))
    dst.ensure()
    upload = os.path.join(dst.root, "api", "uploads", "1234-anna.mimvoice")
    os.makedirs(os.path.dirname(upload))
    with open(res["path"], "rb") as fi, open(upload, "wb") as fo:
        fo.write(fi.read())
    voice = import_bundle(dst, upload)
    assert voice == {**voices.describe(src, "anna"), "dataset_minutes": 0.0, "parts": 0, "training_paused": None}
    assert voice["trained"] and voice["references"] == ["uk"]
    for f in res["files"]:
        if f["path"] != "voice.json":
            with open(os.path.join(dst.voice("anna"), *f["path"].split("/")), "rb") as fh:
                assert hashlib.sha256(fh.read()).hexdigest() == f["sha256"]
    assert reflang.read_sidecar(dst.reference("anna")) is not None
    assert not os.path.exists(os.path.join(dst.voice("anna"), "reference_media.wav"))
    assert os.listdir(os.path.join(dst.voice("anna"), "dataset")) == []
    assert export(dst)["sha256"] == res["sha256"]
    assert os.listdir(os.path.join(dst.tmp, "import")) == []


def test_taken_id_gets_a_new_one(tmp_path):
    data = make_voice(tmp_path / "a")
    res = export(data)
    voice = import_bundle(data, res["path"], name="Anna copy")
    assert voice["id"] == "anna-copy" and voice["name"] == "Anna copy"
    assert import_bundle(data, res["path"])["id"] == "anna-2"


def test_without_rvc_model(tmp_path):
    data = make_voice(tmp_path / "a", rvc=False)
    res = export(data)
    assert not any(f["path"].startswith("rvc/") for f in res["files"])
    dst = config.DataDir(str(tmp_path / "b"))
    voice = import_bundle(dst, res["path"])
    assert voice["trained"] is False and voice["trained_epochs"] is None
    assert not os.path.exists(os.path.join(dst.voice("anna"), "rvc"))


def assert_rejected(tmp_path, path):
    dst = config.DataDir(str(tmp_path / "b"))
    with pytest.raises(EngineError) as err:
        import_bundle(dst, path)
    assert err.value.code == "bad_request"
    assert voices.list_voice_ids(dst) == []
    assert not os.path.isdir(os.path.join(dst.tmp, "import")) or os.listdir(os.path.join(dst.tmp, "import")) == []
    return err.value.message


def test_tampered_member_is_rejected(tmp_path):
    res = export(make_voice(tmp_path / "a"))

    def flip(name, body):
        return (name, body[:-1] + bytes([body[-1] ^ 1])) if name == "rvc/model.pth" else (name, body)

    msg = assert_rejected(tmp_path, rewrite(res["path"], tmp_path / "bad.mimvoice", flip))
    assert "rvc/model.pth" in msg and "hash" in msg


def test_member_missing_from_manifest_is_rejected(tmp_path):
    res = export(make_voice(tmp_path / "a"))
    path = rewrite(res["path"], tmp_path / "bad.mimvoice", lambda n, b: None if n == "rvc/model.index" else (n, b))
    assert "manifest" in assert_rejected(tmp_path, path)


@pytest.mark.parametrize("name", [
    "../evil.wav", "/abs/reference.wav", "rvc/../../x.pth", "C:/x/reference.wav", "./reference.wav", "dataset/part1.wav",
    pytest.param("rvc\\model.pth", marks=pytest.mark.skipif(os.sep == "\\", reason="zipfile turns \\ into / there")),
])
def test_unsafe_or_unknown_names_are_rejected(tmp_path, name):
    res = export(make_voice(tmp_path / "a"))
    path = rewrite(res["path"], tmp_path / "bad.mimvoice", lambda n, b: (name, b) if n == "reference.wav" else (n, b))
    msg = assert_rejected(tmp_path, path)
    assert repr(name) in msg
    assert not os.path.exists(tmp_path / "evil.wav") and not os.path.exists(tmp_path / "x.pth")


def test_oversized_member_is_rejected(tmp_path, monkeypatch):
    res = export(make_voice(tmp_path / "a"))
    monkeypatch.setattr(voicebundle, "MAX_MEMBER", 2000)
    assert "too large" in assert_rejected(tmp_path, res["path"])


def test_not_a_zip_and_mixed_params(tmp_path):
    junk = tmp_path / "junk.mimvoice"
    junk.write_bytes(b"not a zip")
    assert_rejected(tmp_path, str(junk))
    with pytest.raises(EngineError) as err:
        voices.voices_import(Ctx(str(tmp_path / "b")), {"bundle": str(junk), "rvc_model": str(junk)})
    assert err.value.code == "bad_request"


def test_stale_sidecar_is_left_out(tmp_path):
    data = make_voice(tmp_path / "a")
    ref = data.reference("anna")
    with open(ref, "r+b") as fh:
        fh.seek(-2, os.SEEK_END)
        fh.write(b"\x01\x02")
    res = export(data)
    names = [f["path"] for f in res["files"]]
    assert "reference.wav" in names and "reference.omnivoice.json" not in names
    assert "reference_uk.omnivoice.json" in names
    dst = config.DataDir(str(tmp_path / "b"))
    import_bundle(dst, res["path"])
    assert not os.path.exists(os.path.join(dst.voice("anna"), "reference.omnivoice.json"))


def test_deeply_nested_voice_json_is_rejected(tmp_path):
    res = export(make_voice(tmp_path / "a"))
    path = rebuild(res["path"], tmp_path / "bad.mimvoice", {"voice.json": b"[" * 100_000 + b"]" * 100_000})
    assert "voice.json" in assert_rejected(tmp_path, path)
    path = rebuild(res["path"], tmp_path / "bad2.mimvoice", {})
    with zipfile.ZipFile(path) as zf:
        members = {i.filename: zf.read(i) for i in zf.infolist() if i.filename != "manifest.json"}
    with zipfile.ZipFile(path, "w") as zf:
        for n, b in members.items():
            zf.writestr(n, b)
        zf.writestr("manifest.json", b"[" * 100_000 + b"]" * 100_000)
    assert "manifest.json" in assert_rejected(tmp_path, path)


def test_impossible_sidecar_time_is_rejected(tmp_path):
    data = make_voice(tmp_path / "a")
    res = export(data)
    side = {"version": 1, "sig": [10**30, os.path.getsize(data.reference("anna"))], "text": "hi"}
    path = rebuild(res["path"], tmp_path / "bad.mimvoice", {"reference.omnivoice.json": json.dumps(side).encode()})
    assert "reference.wav" in assert_rejected(tmp_path, path)


def test_export_refuses_what_import_would(tmp_path, monkeypatch):
    data = make_voice(tmp_path / "a")
    monkeypatch.setattr(voicebundle, "MAX_MEMBER", 2000)
    with pytest.raises(EngineError) as err:
        export(data)
    assert err.value.code == "bad_request" and "reference.wav is too large" in err.value.message
    monkeypatch.setattr(voicebundle, "MAX_MEMBER", 2 << 30)
    monkeypatch.setattr(voicebundle, "MAX_TOTAL", 40_000)
    with pytest.raises(EngineError) as err:
        export(data)
    assert err.value.code == "bad_request" and "too large" in err.value.message
    assert not os.path.isdir(os.path.join(data.tmp, "export")) or os.listdir(os.path.join(data.tmp, "export")) == []


def test_export_errors(tmp_path):
    data = make_voice(tmp_path / "a")
    with pytest.raises(EngineError) as err:
        voices.voices_export(Ctx(data.root), {})
    assert err.value.code == "bad_request" and err.value.message == "'id' is required"
    with pytest.raises(EngineError) as err:
        export(data, dest="relative.mimvoice")
    assert err.value.code == "bad_request"
    with pytest.raises(EngineError) as err:
        export(data, vid="nobody")
    assert err.value.code == "not_found"
    os.remove(data.reference("anna"))
    with pytest.raises(EngineError) as err:
        export(data)
    assert err.value.code == "bad_request"


def test_zip_is_stored_with_fixed_times(tmp_path):
    res = export(make_voice(tmp_path / "a"))
    with zipfile.ZipFile(io.BytesIO(open(res["path"], "rb").read())) as zf:
        assert {(i.compress_type, i.date_time) for i in zf.infolist()} == {(zipfile.ZIP_STORED, (1980, 1, 1, 0, 0, 0))}
