import types

import pytest
from mimvox_engine import textimport


@pytest.fixture
def ctx(tmp_path):
    return types.SimpleNamespace(data_dir=str(tmp_path / "data"), runtime={})


@pytest.fixture
def import_book(ctx):
    def run(path, **params):
        return textimport.text_import(ctx, {"path": str(path), **params})

    return run
