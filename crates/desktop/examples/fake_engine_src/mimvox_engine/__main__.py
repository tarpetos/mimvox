import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).absolute().parent / ".." / ".." / "fake_engine.py"), run_name="__main__")
