"""Atomic writes for model artifacts and JSON reports."""

import json
import os
from pathlib import Path
import tempfile


def atomic_write(path, data):
    """Replace one result file only after the new contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    try:
        with handle:
            handle.write(data)
        os.replace(handle.name, path)
    finally:
        Path(handle.name).unlink(missing_ok=True)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
