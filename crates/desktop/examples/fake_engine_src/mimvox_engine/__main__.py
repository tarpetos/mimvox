import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fake_engine.py"), run_name="__main__")
