from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)


class _HasToJsonDict(Protocol):
    def to_json_dict(self) -> dict: ...


class JSONLResultWriter:
    """Appends results one at a time (never batched) so a crash or a killed
    sbatch job never loses more than the single in-flight record. Also tracks
    which ids are already present, so a re-submitted job skips finished work
    instead of re-generating (and re-paying for) it.

    2026-08-14: made domain-agnostic (was previously typed to
    ExtractionResult specifically) — reused as-is for the jury's JudgeScore
    results too. Anything with a to_json_dict() method and an "id" field
    in that dict works here.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def already_processed_ids(self) -> set[str]:
        """A truncated last line is possible if a previous run was killed
        mid-write (disk full, SIGKILL between write() and flush()) — skip
        that one line with a warning instead of crashing the whole resume,
        which would otherwise force a full re-run over already-paid-for work.
        """
        if not self._path.exists():
            return set()
        ids = set()
        with self._path.open("r", encoding="utf-8") as handle:
            for line_num, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError) as error:
                    logger.warning("Skipping corrupted line %d in %s: %s", line_num, self._path, error)
        return ids

    def append(self, result: _HasToJsonDict) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_json_dict(), ensure_ascii=False) + "\n")
