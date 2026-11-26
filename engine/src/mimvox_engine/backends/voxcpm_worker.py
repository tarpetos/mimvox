"""VoxCPM2 worker (Apache-2.0): 30 languages, 48 kHz.

    python -I voxcpm_worker.py --model-dir <data>/runtime/models/voxcpm [--device cuda|cpu|mps]

Ops: see ``clone_worker.py``.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("mimvox_clone_worker", os.path.join(HERE, "clone_worker.py"))
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
INP, OUT = common.kit.open_channels()  # before torch: it prints, also from C code
args = common.parse_args()

STEPS = {"best": 10, "fast": 6}


def load(model_dir, device):
    from voxcpm import VoxCPM

    # optimize=False: torch.compile took ~60 s at every start for a ~8 % faster generation (RTX 4070).
    return VoxCPM.from_pretrained(model_dir, load_denoiser=False, optimize=False, local_files_only=True,
                                  device=device)


def speak(model, text, lang, prompt, req):
    extra = {"prompt_wav_path": prompt["wav"], "prompt_text": prompt["text"]} if prompt["text"] else {}
    wav = model.generate(text=text, reference_wav_path=prompt["wav"], cfg_value=2.0,
                         inference_timesteps=STEPS.get(req.get("quality"), STEPS["best"]), **extra)
    return wav, model.tts_model.sample_rate


common.run("voxcpm", INP, OUT, args, load, speak)
os._exit(0)
