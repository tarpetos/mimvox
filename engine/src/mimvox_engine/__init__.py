import os as _os

# Caps OpenBLAS's per-thread buffers (numpy, scipy, faiss) before numpy loads; workers inherit it.
# Measured: RVC worker 6.8 GB -> 3.1 GB, engine 1.1 GB -> 0.2 GB.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

__version__ = "0.1.0"
