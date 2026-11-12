"""RVC training: orchestrates Applio's CLI (preprocess -> extract -> train -> index) for one voice.

Importing this package registers ``train.start``, ``train.pause``, ``train.resume`` and ``train.cancel`` with the method registry.
``recommended_epochs`` is public so the dataset module can reuse it for ``dataset.summary``.
"""
from . import handlers  # noqa: F401  (registers the methods)
from .plan import pick_batch_size, recommended_epochs

__all__ = ["pick_batch_size", "recommended_epochs"]
