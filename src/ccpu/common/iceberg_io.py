"""PyIceberg local-file compatibility helpers for Windows."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def local_file_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


class WindowsSafePyArrowFileIO:
    """Factory-compatible PyArrowFileIO with standard Windows file URI parsing."""

    def __new__(cls, properties: dict[str, Any] | None = None) -> Any:
        from pyiceberg.io.pyarrow import PyArrowFileIO

        class _FileIO(PyArrowFileIO):
            @staticmethod
            def parse_location(
                location: str, properties: dict[str, Any] | None = None
            ) -> tuple[str, str, str]:
                if os.name == "nt" and re.match(r"^file:///[A-Za-z]:/", location):
                    return "file", "", location[len("file:///") :]
                return PyArrowFileIO.parse_location(location, properties or {})

        return _FileIO(properties or {})


def local_path_from_uri(location: str) -> Path:
    if os.name == "nt" and re.match(r"^file:///[A-Za-z]:/", location):
        return Path(location[len("file:///") :])
    if location.startswith("file://"):
        return Path(location[len("file://") :])
    return Path(location)
