"""Redub: drop an audio/video file, then voice-convert it or transcribe it and re-speak it in a chosen voice.

Importing this package registers ``redub.probe``, ``redub.prepare``, ``redub.transcribe``, ``redub.dub``,
``redub.render`` and ``redub.cancel``.
"""
from . import handlers  # noqa: F401  (the import registers the methods)
