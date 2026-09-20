from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class PileNERRecord:
    id: str
    span: str
    gold: dict[str, list[str]]


class PileNERLoader:
    """Loads records from filippogreppi/Pile-NER-type-merged-CoT.

    Unlike the original Universal-NER/Pile-NER-type (ShareGPT-style
    conversations, needing regex parsing to recover the span and gold
    entities — see git history), this dataset already ships flat `text` and
    structured `entities` columns, so loading is direct.

    Note: despite iter_records() being a generator, the underlying rows are
    fully materialized in memory on first access (cached after that) — this
    is not a true streaming reader. Fine for this dataset's size (45,889
    rows), would need revisiting for something much larger.
    """

    def __init__(self, source: str = "filippogreppi/Pile-NER-type-merged-CoT", split: str = "train") -> None:
        self._source = source
        self._split = split
        self._rows_cache: list[dict] | None = None

    def iter_records(self, start_idx: int = 0, end_idx: int | None = None) -> Iterator[PileNERRecord]:
        rows = self._load_rows()
        end_idx = len(rows) if end_idx is None else min(end_idx, len(rows))
        for i in range(start_idx, end_idx):
            row = rows[i]
            yield PileNERRecord(
                id=row["id"],
                span=row["text"],
                gold=self._group_by_label(row["entities"]),
            )

    def iter_records_by_ids(self, ids: set[str]) -> Iterator[PileNERRecord]:
        """Yields only the records whose id is in `ids`, in dataset order —
        used for subset-file-based sharding (build_subsets.py), which
        splits by exact remaining ids rather than a contiguous index range.
        Unlike start_idx/end_idx slicing, this correctly picks up rows that
        failed earlier and were never actually completed, wherever they
        fall positionally — a contiguous-range shard boundary would
        otherwise silently skip them forever once a later shard's range
        starts past that position."""
        rows = self._load_rows()
        for row in rows:
            if row["id"] in ids:
                yield PileNERRecord(
                    id=row["id"],
                    span=row["text"],
                    gold=self._group_by_label(row["entities"]),
                )

    def all_ids(self) -> list[str]:
        """All ids in the dataset, in order — used by build_subsets.py to
        compute the full id universe before subtracting already-done ones."""
        return [row["id"] for row in self._load_rows()]

    def __len__(self) -> int:
        return len(self._load_rows())

    def _load_rows(self) -> list[dict]:
        if self._rows_cache is not None:
            return self._rows_cache
        if Path(self._source).exists():
            self._rows_cache = [json.loads(line) for line in Path(self._source).read_text().splitlines() if line.strip()]
        else:
            from datasets import load_dataset

            self._rows_cache = list(load_dataset(self._source, split=self._split))
        return self._rows_cache

    @staticmethod
    def _group_by_label(entities: list[dict]) -> dict[str, list[str]]:
        """Reference gold for the jury's completeness check later — kept as a
        flat text list per label; duplicate mentions of the same span are
        expected and harmless here (the check is "does this span appear
        anywhere", not a count)."""
        gold: dict[str, list[str]] = {}
        for e in entities:
            gold.setdefault(e["label"], []).append(e["text"])
        return gold
