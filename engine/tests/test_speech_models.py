import importlib.util
import sys
from pathlib import Path

import pytest
from mimvox_engine import backends, tts
from mimvox_engine.backends import clone
from mimvox_engine.registry import EngineError

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "setup"


def _setup_module():
    spec = importlib.util.spec_from_file_location("mimvox_setup_models", TOOLS / "setup.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mimvox_setup_models"] = module
    spec.loader.exec_module(module)
    return module


def install(tmp_path, *names):
    """A runtime.toml dict with the given clone models' interpreters and weights on disk."""
    runtime = {"python": {}, "paths": {}}
    for name in names:
        model = backends.by_name(name)
        py = tmp_path / name / "python"
        py.parent.mkdir(parents=True)
        py.write_bytes(b"")
        weights = tmp_path / "models" / name
        weights.mkdir(parents=True)
        for f in (*model.required, clone.STAMP):
            (weights / f).write_bytes(b"x")
        runtime["python"][name] = str(py)
        runtime["paths"][f"{name}_model"] = str(weights)
    return runtime


@pytest.fixture(autouse=True)
def fresh():
    for b in clone._ALL:
        b._status, b._runtime = "missing", {}
    yield
    for b in clone._ALL:
        b._status, b._runtime = "missing", {}


def probe(runtime):
    for b in clone._ALL:
        b.probe(runtime)


def test_a_named_model_speaks(tmp_path):
    probe(install(tmp_path, "voxcpm"))
    lang, backend = tts.speech_language({"language": "pl", "model": "voxcpm"})
    assert (lang, backend) == ("pl", clone.VOXCPM)
    tts.check_backend_ready(backend)  # "idle": it loads when it first speaks


def test_a_model_that_does_not_speak_the_language_is_refused(tmp_path):
    probe(install(tmp_path, "chatterbox"))
    with pytest.raises(EngineError) as e:
        tts.speech_language({"language": "uk", "model": "chatterbox"})
    assert e.value.code == "bad_request" and "Ukrainian" in e.value.message


def test_an_unknown_or_missing_model(tmp_path):
    probe(install(tmp_path))
    with pytest.raises(EngineError) as e:
        tts.speech_language({"language": "en", "model": "holos"})
    assert e.value.code == "bad_request"
    with pytest.raises(EngineError) as e:
        tts.speech_language({"language": "en", "model": "voxcpm"})
    assert e.value.code == "not_ready"


def test_auto_keeps_omnivoice_and_adds_the_languages_it_lacks(tmp_path):
    assert "he" not in backends.languages()
    with pytest.raises(EngineError):
        tts.speech_language({"language": "he"})
    probe(install(tmp_path, "voxcpm", "chatterbox"))
    assert tts.speech_language({"language": "en"})[1].name == "omnivoice"
    assert tts.speech_language({"language": "he"})[1] is clone.VOXCPM
    assert "he" in backends.languages()
    assert [b.name for b in backends.selectable()][-2:] == ["voxcpm", "chatterbox"]


def test_setup_installs_what_the_engine_looks_for():
    setup = _setup_module()
    for b in clone._ALL:
        spec = setup.CLONE_MODELS[b.name]
        assert set(spec["required"]) == set(b.required)
        assert setup.CLONE_STAMP == clone.STAMP
        assert b.name in setup.COMPONENTS
        assert spec["revision"] and len(spec["revision"]) == 40


@pytest.fixture
def no_omnivoice(monkeypatch):
    from mimvox_engine.backends.omnivoice import OMNIVOICE

    monkeypatch.setattr(OMNIVOICE, "_status", "missing")
    monkeypatch.setattr(OMNIVOICE, "_absent", True)


def test_without_omnivoice_another_model_speaks_its_own_languages(tmp_path, no_omnivoice):
    assert tts.speech_status() == "missing"
    probe(install(tmp_path, "chatterbox"))
    assert tts.speech_status() == "ready"
    assert backends.languages() == set(clone.CHATTERBOX.languages)
    assert tts.speech_language({"language": "en"}) == ("en", clone.CHATTERBOX)
    with pytest.raises(EngineError):
        tts.speech_language({"language": "uk"})


def test_the_last_voice_model_stays_on(tmp_path, no_omnivoice):
    import asyncio
    from types import SimpleNamespace

    from mimvox_engine import components

    probe(install(tmp_path, "chatterbox", "voxcpm"))
    ctx = SimpleNamespace(data_dir=tmp_path)
    asyncio.run(components.components_set(ctx, {"id": "voxcpm", "on": False}))
    with pytest.raises(EngineError) as e:
        asyncio.run(components.components_set(ctx, {"id": "chatterbox", "on": False}))
    assert e.value.code == "bad_request"
