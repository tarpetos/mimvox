import io
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from mimvox_engine import config, rvcfile, voices
from mimvox_engine.registry import EngineError

MODEL = {
    "weight": {"emb_g.weight": [0.0], "enc_p.emb_phone.weight": [0.0]},
    "config": [1, 2, 40000],
    "info": "300epoch",
    "sr": "40k",
    "f0": 1,
    "version": "v2",
}


class Ctx:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.runtime = {}

    def emit(self, event, data, job=None):
        pass


class Evil:
    def __reduce__(self):
        return (os.remove, ("never-run",))


def torch_zip(path, obj, member="model/data.pkl"):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, pickle.dumps(obj, protocol=2))
        z.writestr("model/version", "3\n")
    return str(path)


def legacy(path, obj):
    buf = io.BytesIO()
    for part in (0x1950A86A20F9469CFC6C, 1001, {"little_endian": True}, obj):
        pickle.dump(part, buf, protocol=2)
    path.write_bytes(buf.getvalue() + b"\x00" * 64)
    return str(path)


def reference(path):
    t = np.arange(16000) / 16000
    sf.write(path, (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), 16000, subtype="PCM_16")
    return str(path)


def test_rvc_models_pass(tmp_path):
    assert rvcfile.model_problem(torch_zip(tmp_path / "a.pth", MODEL)) is None
    assert rvcfile.model_problem(legacy(tmp_path / "b.pth", MODEL)) is None


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda p: (p.write_bytes(b"IwFl" + b"\x00" * 100), str(p))[1], rvcfile.NOT_TORCH),
        (lambda p: torch_zip(p, MODEL, member="other.pkl"), rvcfile.NOT_TORCH),
        (lambda p: (p.write_bytes(pickle.dumps(MODEL)), str(p))[1], rvcfile.NOT_TORCH),
        (
            lambda p: torch_zip(p, {"model": MODEL["weight"], "optimizer": {}, "iteration": 3, "learning_rate": 1e-4}),
            rvcfile.CHECKPOINT,
        ),
        (lambda p: torch_zip(p, {"state_dict": {"encoder.weight": [0.0]}}), rvcfile.NOT_RVC),
    ],
)
def test_other_files_are_named(tmp_path, make, message):
    assert rvcfile.model_problem(make(tmp_path / "x.pth")) == message


def test_a_pickle_is_read_not_run(tmp_path, monkeypatch):
    target = tmp_path / "never-run"
    target.write_text("kept")
    monkeypatch.chdir(tmp_path)
    assert rvcfile.model_problem(torch_zip(tmp_path / "evil.pth", {**MODEL, "x": Evil()})) is None
    assert target.read_text() == "kept"


def test_import_refuses_a_bad_model_before_making_a_voice(tmp_path):
    root = tmp_path / "data"
    bad = torch_zip(tmp_path / "G_2000.pth", {"model": {}, "optimizer": {}, "iteration": 2000})
    with pytest.raises(EngineError) as err:
        voices.voices_import(
            Ctx(str(root)), {"name": "Anna", "rvc_model": bad, "reference_path": reference(tmp_path / "ref.wav")}
        )
    assert err.value.code == "bad_request" and "training checkpoint" in err.value.message
    assert not config.DataDir(str(root)).voice("anna").is_dir()


def test_import_copies_a_model_as_trained(tmp_path):
    root = tmp_path / "data"
    pth = torch_zip(tmp_path / "anna_300e_9000s.pth", MODEL)
    idx = tmp_path / "anna.index"
    idx.write_bytes(b"IwFl" + b"\x00" * 100)
    voice = voices.voices_import(
        Ctx(str(root)),
        {
            "name": "Anna",
            "rvc_model": pth,
            "rvc_index": str(idx),
            "reference_path": reference(tmp_path / "ref.wav"),
            "language": "uk",
        },
    )["voice"]
    assert voice["trained"] and voice["trained_epochs"] == 300 and voice["language"] == "uk"
    model, index = config.DataDir(str(root)).rvc_model(voice["id"])
    assert model.read_bytes() == Path(pth).read_bytes() and index.is_file()
