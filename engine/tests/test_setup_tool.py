import importlib.util
import os
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "setup"
TORCH_CPU = '{"torch": "2.8.0+cpu", "cuda": false, "mps": false, "name": "", "hip": false}'


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # setup.run_move imports move_runtime by this name
    spec.loader.exec_module(module)
    return module


setup = _load("mimvox_setup", "setup.py")
move_runtime = _load("move_runtime", "move_runtime.py")


def make(data_dir, plat="linux", **kw):
    return setup.Setup(root=data_dir.parent / "repo", plat=plat, arch="x86_64", data_dir=data_dir, uv_path="uv", **kw)


def write_runtime(s, backend="cpu", components=("speech",)):
    s.data_dir.mkdir(parents=True, exist_ok=True)
    listed = ", ".join(f'"{c}"' for c in components)
    s.runtime_toml.write_text(f"[python]\nengine = {setup.toml_str(s.engine_python())}\n\n[device]\n"
                              f'backend = "{backend}"\n\n[setup]\ncomponents = [{listed}]\n', encoding="utf-8")


def fake_runtime(s, components=("speech",)):
    """The files a speech-only quick check looks at; the interpreters answer through a stubbed ``Setup.read``."""
    py = s.engine_python()
    if not py.exists():
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_bytes(b"")
    for name in setup.OMNIVOICE_REQUIRED:
        (s.omnivoice_model / name).parent.mkdir(parents=True, exist_ok=True)
        (s.omnivoice_model / name).write_bytes(b"x")
    s.ffmpeg_dir().mkdir(parents=True, exist_ok=True)
    for name in ("ffmpeg", "ffprobe"):
        (s.ffmpeg_dir() / (name + (".exe" if s.plat == "windows" else ""))).write_bytes(b"x")
    write_runtime(s, components=components)


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setattr(setup.Setup, "_detect_gpu", lambda self: setup.Gpu("cuda", "GeForce GTX 1650", 4.0, "nvidia"))


@pytest.fixture
def interpreters(monkeypatch):
    monkeypatch.setattr(setup.Setup, "read",
                        lambda self, command, cwd=None: TORCH_CPU if "import json, torch" in str(command[-1]) else "1.0\n")


def test_a_cpu_runtime_stays_on_the_cpu(tmp_path, nvidia, capsys):
    s = make(tmp_path / "data")
    write_runtime(s, "cpu")
    s.detect()
    out = capsys.readouterr().out
    assert s.gpu.backend == "cpu"
    assert "--gpu switches" in out
    assert "No supported GPU" not in out


def test_gpu_flag_leaves_the_cpu(tmp_path, nvidia):
    s = make(tmp_path / "data", force_gpu=True)
    write_runtime(s, "cpu")
    s.detect()
    assert (s.gpu.backend, s.gpu.vendor) == ("cuda", "nvidia")


def test_cpu_flag_without_the_no_gpu_warning(tmp_path, nvidia, capsys):
    s = make(tmp_path / "data", force_cpu=True)
    s.detect()
    assert s.gpu.backend == "cpu"
    assert "No supported GPU" not in capsys.readouterr().out


def test_a_new_install_uses_the_gpu(tmp_path, nvidia):
    s = make(tmp_path / "data")
    s.detect()
    assert s.gpu.backend == "cuda"


def test_cpu_and_gpu_flags_exclude_each_other():
    with pytest.raises(SystemExit):
        setup.parse_args(["--cpu", "--gpu"])


def test_quick_check_of_a_speech_only_runtime(tmp_path, interpreters):
    s = make(tmp_path / "data", components={"speech"})
    fake_runtime(s)
    assert s.quick_check() == []


def test_quick_check_wants_the_translation_model_only_with_translation(tmp_path, interpreters):
    s = make(tmp_path / "data", components={"speech", "translation"})
    fake_runtime(s, ("speech", "translation"))
    problems = s.quick_check()
    assert len(problems) == 1
    assert setup.NLLB_DIR in problems[0]


def test_uv_pythons_go_into_the_runtime(tmp_path, monkeypatch):
    monkeypatch.delenv("UV_PYTHON_INSTALL_DIR", raising=False)
    s = make(tmp_path / "data")
    assert s.env()["UV_PYTHON_INSTALL_DIR"] == str(s.runtime / "python")


def test_nltk_data_goes_into_the_engine_venv(tmp_path, monkeypatch):
    s = make(tmp_path / "data")
    s.engine_python().parent.mkdir(parents=True)
    s.engine_python().write_bytes(b"")
    calls = []
    monkeypatch.setattr(setup.Setup, "must", lambda self, command, what, cwd=None, extra_env=None: None)
    monkeypatch.setattr(setup.Setup, "run",
                        lambda self, command, cwd=None, extra_env=None: calls.append((command, extra_env)) or 0)
    s.engine_env()
    env = next(extra for command, extra in calls if "import g2p_en" in command)
    assert env["NLTK_DATA"] == str(s.engine_venv / "nltk_data")
    assert (s.engine_venv / "nltk_data").is_dir()


def _link(target, link):
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def test_move_of_a_speech_only_runtime_with_its_pythons(tmp_path, monkeypatch, interpreters):
    """uv links a Python's minor version to its patch release with an absolute link, which a rename leaves dangling."""
    monkeypatch.setattr(setup, "location_file", lambda plat: tmp_path / "config" / "location.toml")
    monkeypatch.setattr(move_runtime, "running_processes", lambda old, windows: [])
    old, new = (tmp_path / "old").resolve(), (tmp_path / "new").resolve()
    s = make(old, plat=setup.host_platform(), components={"speech"})
    pythons = s.runtime / "python"
    release = pythons / "cpython-3.11.15"
    (release / "bin").mkdir(parents=True)
    (release / "bin" / "python3.11").write_bytes(b"")
    _link(release, pythons / "cpython-3.11")
    if os.name != "nt":
        s.engine_python().parent.mkdir(parents=True)
        os.symlink(pythons / "cpython-3.11" / "bin" / "python3.11", s.engine_python())
    fake_runtime(s)
    assert setup.run_move(setup.parse_args(["--data-dir", str(old), "--move-to", str(new)])) == 0
    assert not old.exists()
    moved = make(new, plat=setup.host_platform())
    assert Path(os.path.realpath(new / "runtime" / "python" / "cpython-3.11")) == new / "runtime" / "python" / "cpython-3.11.15"
    assert Path(os.path.realpath(moved.engine_python())).is_relative_to(new)
    assert setup.toml_str(moved.engine_python()) in moved.runtime_toml.read_text(encoding="utf-8")
