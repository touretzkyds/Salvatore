"""Snapshot filename generation and persistence utilities for camera captures."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtGui import QImage


_RAW_SUFFIX = "_snap"
_ANNOTATED_SUFFIX = "_asnap"


class SnapshotService:
    """Generate sequential snapshot paths and persist ``QImage`` frames to disk."""

    def __init__(self, base_name: str = "robot", output_dir: str | Path = "snapshots") -> None:
        self._base_name = base_name
        self._output_dir = Path(output_dir)
        self._counter = 0
        self._write_enabled = True

    def enable_disk_write(self, enabled: bool) -> None:
        """Toggle on-disk persistence without disturbing the filename sequence."""

        self._write_enabled = bool(enabled)

    def next_path_raw(self) -> Path:
        return self._next_path(_RAW_SUFFIX)

    def next_path_annotated(self) -> Path:
        return self._next_path(_ANNOTATED_SUFFIX)

    def capture(self, image: Optional[QImage], *, annotated: bool = False) -> Optional[Path]:
        """Persist ``image`` and return the resolved output path.

        If ``image`` is ``None`` or null, ``None`` is returned and the counter is not
        advanced. When disk writes are disabled the computed path is still returned,
        allowing callers to surface the intended filename to users.
        """

        if image is None or image.isNull():
            return None

        path = self.next_path_annotated() if annotated else self.next_path_raw()
        if not self._write_enabled:
            return path

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        if not image.save(str(path), "PNG"):
            return None
        return path

    def _next_path(self, suffix: str) -> Path:
        path = self._output_dir / f"{self._base_name}{suffix}{self._counter}.png"
        self._counter += 1
        return path


__all__ = ["SnapshotService"]
