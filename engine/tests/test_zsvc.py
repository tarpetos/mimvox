import asyncio
import logging

from mimvox_engine import zsvc
from mimvox_engine.workerproc import WorkerDied, WorkerError


class FakeClient:
    def __init__(self, exc):
        self.exc = exc
        self.on_release = []

    def alive(self):
        return True

    async def call(self, *args, **kwargs):
        raise self.exc


def release(exc):
    asyncio.run(zsvc.ZsvcWorker("code", "models", FakeClient(exc)).release_gpu())


def test_a_converter_whose_process_ended_has_nothing_to_release(caplog):
    with caplog.at_level(logging.ERROR):
        release(WorkerDied("the omnivoice worker exited (code 0)"))
    assert "could not release" not in caplog.text


def test_a_failed_release_is_logged(caplog):
    with caplog.at_level(logging.ERROR):
        release(WorkerError("internal", "out of memory"))
    assert "could not release" in caplog.text
