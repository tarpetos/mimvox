"""Dataset tooling: presets, reading-text analysis, recording checks and the per-voice dataset store.

Importing this package registers the handlers ``presets.list``, ``text.analyze``, ``recording.check`` and
``dataset.add`` / ``dataset.remove`` / ``dataset.summary``, ``voices.reference_candidates`` / ``voices.set_reference`` /
``voices.add_reference`` / ``voices.remove_reference`` and ``voices.reference_from_media`` / ``voices.media_cancel`` /
``voices.clear_media`` with the engine registry.
"""
from . import quality, store, text  # noqa: F401  (imports register the handlers)
from . import media_ref, reference  # noqa: F401
from . import presets  # noqa: F401
from .levels import level_for, recommended_epochs  # noqa: F401
