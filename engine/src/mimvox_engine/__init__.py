"""Mimvox engine: OmniVoice + RVC behind a JSON-lines protocol (see contract/engine-protocol.md)."""

import os as _os

# Before numpy is imported anywhere (the engine and, through the inherited environment, every worker): OpenBLAS
# allocates a buffer per CPU thread when it loads, in numpy, scipy and faiss each. On a 16-thread CPU that is 0.5 GB of
# commit per copy that nothing here uses (measured: the RVC worker 6.8 GB -> 3.1 GB, the engine 1.1 GB -> 0.2 GB). The
# heavy math runs in PyTorch, which has its own threads.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

__version__ = "0.1.0"
