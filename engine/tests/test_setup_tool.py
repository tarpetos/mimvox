import importlib.util
import os
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools" / "setup"
TORCH_CPU = '{"torch": "2.11.0+cpu", "cuda": false, "mps": false, "name": "", "hip": false}'


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
    s.runtime_toml.write_text(
        f"[python]\nengine = {setup.toml_str(s.engine_python())}\n\n[device]\n"
        f'backend = "{backend}"\n\n[setup]\ncomponents = [{listed}]\n',
        encoding="utf-8",
    )


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
    def read(self, command, cwd=None):
        return TORCH_CPU if "import json, torch" in str(command[-1]) else "1.0\n"

    monkeypatch.setattr(setup.Setup, "read", read)


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


class FakeUv:
    """``uv venv`` makes a venv on a Python in <runtime>/python; ``fail`` makes the commands it matches fail."""

    def __init__(self, s):
        self.s, self.calls, self.fail = s, [], lambda argv: False
        self.freeze = {}
        self.python = True

    def run(self, command, cwd=None, extra_env=None):
        argv = [str(part) for part in command]
        self.calls.append(argv)
        if self.fail(argv):
            return 1
        if argv[1:3] == ["python", "find"]:
            return 0 if self.python else 2
        if argv[1:3] == ["python", "install"]:
            self.python = True
        if argv[1:2] == ["venv"]:
            if not self.python and "--offline" in argv:
                return 2  # "No interpreter found for Python 3.14 in managed installations"
            venv = Path(argv[-1])
            venv_python = setup.venv_python(venv, self.s.plat)
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_bytes(b"")
            version = argv[argv.index("--python") + 1]
            home = self.s.python_dir / f"cpython-{version}-linux-x86_64-gnu" / "bin"
            (venv / "pyvenv.cfg").write_text(f"home = {home}\nversion_info = {version}.0\n", encoding="utf-8")
        return 0

    def read(self, command, cwd=None):
        return self.freeze.get(Path(str(command[command.index("--python") + 1])).parents[1].name, "")

    def venv_calls(self):
        return [argv for argv in self.calls if argv[1:2] == ["venv"]]


@pytest.fixture
def fake_uv(tmp_path, monkeypatch):
    s = make(tmp_path / "data")
    uv = FakeUv(s)
    monkeypatch.setattr(setup.Setup, "run", lambda self, command, cwd=None, extra_env=None: uv.run(command))
    monkeypatch.setattr(setup.Setup, "quiet", lambda self, command, cwd=None: uv.run(command))
    monkeypatch.setattr(setup.Setup, "read", lambda self, command, cwd=None: uv.read(command))
    s.engine_env()
    uv.calls.clear()
    return s, uv


def old_python(s, tmp_path):
    r"""The engine venv as setup made it before <runtime>/python: from uv's default folder (%APPDATA%\\uv\\python)."""
    home = tmp_path / "Roaming" / "uv" / "python" / "cpython-3.11-windows-x86_64-none"
    (s.engine_venv / "pyvenv.cfg").write_text(f"home = {home}\nversion_info = 3.11\n", encoding="utf-8")
    (s.engine_venv / "old-file").write_bytes(b"")
    return home


def test_which_venvs_are_on_a_python_in_the_runtime(tmp_path, fake_uv):
    s, _uv = fake_uv
    assert s.on_runtime_python(s.engine_venv)
    old_python(s, tmp_path)
    assert not s.on_runtime_python(s.engine_venv)
    assert s.on_runtime_python(tmp_path / "no-venv")


def test_a_ready_venv_in_the_runtime_is_left_alone(fake_uv, capsys):
    s, uv = fake_uv
    s.engine_env()
    assert uv.venv_calls() == []
    assert "Engine environment: ready." in capsys.readouterr().out


def test_a_venv_from_outside_the_runtime_is_made_again_with_its_packages(tmp_path, fake_uv, capsys):
    s, uv = fake_uv
    old_python(s, tmp_path)
    uv.freeze[".venv"] = "torch==2.8.0+cpu\nomnivoice==0.2.1\nipa-uk @ https://example.com/a.zip\nold-thing==1.0\n"
    uv.freeze[".venv.new"] = "torch==2.8.0+cpu\nomnivoice==0.2.1\n"
    s.engine_env()
    (venv_call,) = uv.venv_calls()
    assert {"--offline", "--relocatable", "--managed-python"} <= set(venv_call)
    install = next(argv for argv in uv.calls if argv[1:3] == ["pip", "install"])
    assert "--offline" in install
    pins = Path(install[install.index("-c") + 1]).read_text(encoding="utf-8").split()
    assert pins == ["torch==2.8.0+cpu", "omnivoice==0.2.1", "old-thing==1.0"]
    assert s.on_runtime_python(s.engine_venv)
    assert not (s.engine_venv / "old-file").exists()
    assert (s.engine_venv / ".mimvox-setup-ok").is_file()
    assert sorted(p.name for p in s.engine_venv.parent.iterdir()) == [".venv"]
    assert "no longer installed: old-thing==1.0" in capsys.readouterr().out
    uv.calls.clear()
    s.engine_env()
    assert uv.venv_calls() == []


def on_older_python(s, version="3.11"):
    """The engine venv as an earlier version of setup made it: in <runtime>/python, on an older Python."""
    home = s.python_dir / f"cpython-{version}-linux-x86_64-gnu" / "bin"
    (s.engine_venv / "pyvenv.cfg").write_text(f"home = {home}\nversion_info = {version}.15\n", encoding="utf-8")
    (s.engine_venv / ".mimvox-setup-ok").write_text("torch==2.8.0 torchaudio==2.8.0 cpu linux", encoding="utf-8")
    (s.engine_venv / "old-file").write_bytes(b"")


def test_a_venv_on_an_older_python_is_made_again_on_the_new_one(fake_uv, capsys):
    s, uv = fake_uv
    on_older_python(s)
    s.engine_env()
    (venv_call,) = uv.venv_calls()
    assert venv_call[venv_call.index("--python") + 1] == setup.ENGINE_PYTHON
    install = next(argv for argv in uv.calls if argv[1:3] == ["pip", "install"])
    assert "-c" not in install
    assert setup.venv_version(s.engine_venv) == setup.ENGINE_PYTHON
    assert not (s.engine_venv / "old-file").exists()
    assert sorted(p.name for p in s.engine_venv.parent.iterdir()) == [".venv"]
    assert f"on Python 3.11, needs {setup.ENGINE_PYTHON}" in capsys.readouterr().out
    uv.calls.clear()
    s.engine_env()
    assert uv.venv_calls() == []


def test_a_failed_python_upgrade_keeps_the_old_venv(fake_uv):
    s, uv = fake_uv
    on_older_python(s)
    uv.fail = lambda argv: argv[1:3] == ["pip", "install"]
    with pytest.raises(setup.SetupError):
        s.engine_env()
    assert setup.venv_version(s.engine_venv) == "3.11"
    assert (s.engine_venv / "old-file").exists()
    assert sorted(p.name for p in s.engine_venv.parent.iterdir()) == [".venv"]


def test_runtime_toml_names_an_installed_clone_model(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "location_file", lambda plat: tmp_path / "location.toml")
    s = make(tmp_path / "data")
    s.components = {"speech", "voxcpm", "chatterbox"}
    for py in (s.engine_python(), s.clone_python("voxcpm")):
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_bytes(b"")
    s.write_runtime_toml()
    text = s.runtime_toml.read_text(encoding="utf-8")
    assert f"voxcpm = {setup.toml_str(s.clone_python('voxcpm'))}" in text
    assert f"voxcpm_model = {setup.toml_str(s.clone_models_dir('voxcpm'))}" in text
    assert "chatterbox" not in text.split("[setup]")[0]
    assert s.component_paths("voxcpm") == [s.runtime / "voxcpm", s.runtime / "models" / "voxcpm"]


def test_a_clone_model_an_older_setup_left_out_is_kept(tmp_path):
    s = make(tmp_path / "data")
    write_runtime(s, components=("speech", "ukrainian"))
    (s.clone_models_dir("chatterbox")).mkdir(parents=True)
    (s.clone_models_dir("chatterbox") / setup.CLONE_STAMP).write_text("x", encoding="utf-8")
    (s.clone_models_dir("voxcpm")).mkdir(parents=True)
    s.resolve_components(setup.parse_args([]))
    assert s.components == {"speech", "ukrainian", "chatterbox"}


def test_pythons_no_environment_uses_are_removed(fake_uv, capsys):
    s, _uv = fake_uv
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    names = [
        "cpython-3.11.15-linux-x86_64-gnu",
        "cpython-3.11-linux-x86_64-gnu",
        f"cpython-{running}.1-linux-x86_64-gnu",
        f"cpython-{setup.ENGINE_PYTHON}.7-linux-x86_64-gnu",
        ".lock",
    ]
    for name in names:
        if name != "cpython-3.11-linux-x86_64-gnu":
            (s.python_dir / name).mkdir(parents=True, exist_ok=True)
    minor = s.python_dir / "cpython-3.11-linux-x86_64-gnu"
    if os.name == "nt":
        minor.mkdir()
    else:
        minor.symlink_to(s.python_dir / "cpython-3.11.15-linux-x86_64-gnu")
    s.remove_unused_pythons()
    left = {p.name for p in s.python_dir.iterdir()}
    kept = {f"cpython-{running}.1-linux-x86_64-gnu", f"cpython-{setup.ENGINE_PYTHON}.7-linux-x86_64-gnu", ".lock"}
    assert left >= kept
    assert not left & {"cpython-3.11.15-linux-x86_64-gnu", "cpython-3.11-linux-x86_64-gnu"}
    assert "Removing Python 3.11" in capsys.readouterr().out


def test_a_python_a_venv_still_uses_is_kept(fake_uv):
    s, _uv = fake_uv
    on_older_python(s)
    (s.python_dir / "cpython-3.11.15-linux-x86_64-gnu").mkdir(parents=True)
    s.remove_unused_pythons()
    assert (s.python_dir / "cpython-3.11.15-linux-x86_64-gnu").is_dir()


def test_a_rebuild_downloads_what_the_cache_lacks(tmp_path, fake_uv, capsys):
    s, uv = fake_uv
    old_python(s, tmp_path)
    uv.fail = lambda argv: "--offline" in argv and argv[1:3] == ["pip", "install"]
    s.engine_env()
    assert all("--offline" in argv for argv in uv.venv_calls())
    offline = ["--offline" in argv for argv in uv.calls if argv[1:3] == ["pip", "install"]]
    assert offline[0] and not offline[-1]
    assert s.on_runtime_python(s.engine_venv)
    out = capsys.readouterr().out
    assert "not every package is in the cache" in out and "downloading the missing ones" in out
    assert "downloading Python" not in out


def test_a_rebuild_fetches_the_python_first_and_the_packages_from_the_cache(tmp_path, fake_uv, capsys):
    s, uv = fake_uv
    old_python(s, tmp_path)
    uv.python = False
    s.engine_env()
    python_install = [i for i, argv in enumerate(uv.calls) if argv[1:3] == ["python", "install"]]
    assert len(python_install) == 1
    install = uv.calls[python_install[0]]
    assert install[-1] == setup.ENGINE_PYTHON and {"--no-bin", "--no-registry"} <= set(install)
    later = uv.calls[python_install[0] + 1 :]
    assert all("--offline" in argv for argv in later if argv[1:2] == ["venv"] or argv[1:3] == ["pip", "install"])
    assert len(uv.venv_calls()) == 1
    assert s.on_runtime_python(s.engine_venv)
    out = capsys.readouterr().out
    assert f"Engine environment: downloading Python {setup.ENGINE_PYTHON} into {s.python_dir}..." in out
    assert "not every package is in the cache" not in out


def test_without_the_python_the_old_venv_is_kept(tmp_path, fake_uv, capsys):
    s, uv = fake_uv
    home = old_python(s, tmp_path)
    uv.python = False
    uv.fail = lambda argv: argv[1:3] == ["python", "install"]
    s.engine_env()
    assert uv.venv_calls() == [] and setup.venv_home(s.engine_venv) == home
    out = capsys.readouterr().out
    assert "could not be downloaded" in out and "the old environment is kept" in out


def test_a_failed_rebuild_keeps_the_old_venv(tmp_path, fake_uv, capsys):
    s, uv = fake_uv
    home = old_python(s, tmp_path)
    uv.fail = lambda argv: argv[1:3] == ["pip", "install"]
    s.engine_env()
    assert setup.venv_home(s.engine_venv) == home
    assert (s.engine_venv / "old-file").exists()
    assert (s.engine_venv / ".mimvox-setup-ok").is_file()
    assert sorted(p.name for p in s.engine_venv.parent.iterdir()) == [".venv"]
    assert "the old environment is kept" in capsys.readouterr().out


def test_an_interrupted_swap_puts_the_old_venv_back(tmp_path, fake_uv):
    s, _uv = fake_uv
    old_python(s, tmp_path)
    s.engine_venv.rename(s.engine_venv.with_name(".venv.old"))
    s.engine_venv.with_name(".venv.new").mkdir()
    setup.restore_swapped(s.engine_venv)
    assert (s.engine_venv / "old-file").exists()
    assert sorted(p.name for p in s.engine_venv.parent.iterdir()) == [".venv"]


def test_nltk_data_goes_into_the_engine_venv(fake_uv):
    s, uv = fake_uv
    target = s.engine_venv / "nltk_data"
    s.engine_env()
    (download,) = [argv for argv in uv.calls if "download_dir" in " ".join(argv)]
    assert download[download.index("-c") + 2] == str(target)
    assert download[download.index("-c") + 3 :] == [name for _kind, name in setup.NLTK_PACKAGES]
    for kind, name in setup.NLTK_PACKAGES:
        (target / kind).mkdir(parents=True, exist_ok=True)
        (target / kind / f"{name}.zip").write_bytes(b"")
    uv.calls.clear()
    s.engine_env()
    assert not [argv for argv in uv.calls if "download_dir" in " ".join(argv)]


def _link(target, link):
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def test_move_of_a_speech_only_runtime_with_its_pythons(tmp_path, monkeypatch, interpreters):
    """Uv links a Python's minor version to its patch release with an absolute link, which a rename leaves dangling."""
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
        s.engine_python().symlink_to(pythons / "cpython-3.11" / "bin" / "python3.11")
    fake_runtime(s)
    assert setup.run_move(setup.parse_args(["--data-dir", str(old), "--move-to", str(new)])) == 0
    assert not old.exists()
    moved = make(new, plat=setup.host_platform())
    pythons = new / "runtime" / "python"
    assert (pythons / "cpython-3.11").resolve() == pythons / "cpython-3.11.15"
    assert moved.engine_python().resolve().is_relative_to(new)
    assert setup.toml_str(moved.engine_python()) in moved.runtime_toml.read_text(encoding="utf-8")
