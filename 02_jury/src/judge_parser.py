from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class JudgeScore:
    id: str
    judge_model: str
    completeness_score: float
    completeness_notes: str
    correctness_score: float
    correctness_notes: str
    raw_response: str

    def to_json_dict(self) -> dict:
        return asdict(self)


def parse_judge_response(record_id: str, judge_model: str, raw: str) -> Optional[JudgeScore]:
    """Same layered fallback as the production extractor's parser: try a
    fenced ```json block first, then a bare-JSON parse of the whole text,
    then a balanced-brace scan — a judge model can have the same
    formatting quirks as the extractor model did.

    Only the text after the LAST </think> is the answer. DeepSeek-R1-Distill's
    chat template already puts the opening <think> in the prompt, so its
    output has a closing tag with no opening one (99.9% of real DeepSeek
    scores) — searching for a full <think>...</think> pair never matched, and
    the JSON was then looked for inside the reasoning too. An opening tag with
    no closing one means the reasoning was cut off: no answer at all."""
    if "</think>" in raw:
        visible = raw.rsplit("</think>", 1)[1].strip()
    elif "<think>" in raw:
        return None
    else:
        visible = raw.strip()

    payload = None
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", visible, re.DOTALL)
    if fenced:
        try:
            payload = json.loads(fenced.group(1))
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        try:
            payload = json.loads(visible)
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        start = visible.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(visible)):
                if visible[i] == "{":
                    depth += 1
                elif visible[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            payload = json.loads(visible[start: i + 1])
                        except json.JSONDecodeError:
                            payload = None
                        break
    if payload is None:
        return None

    try:
        return JudgeScore(
            id=record_id,
            judge_model=judge_model,
            completeness_score=float(payload["completeness_score"]),
            completeness_notes=str(payload.get("completeness_notes", "")),
            correctness_score=float(payload["correctness_score"]),
            correctness_notes=str(payload.get("correctness_notes", "")),
            raw_response=raw,
        )
    except (KeyError, ValueError, TypeError):
        return None
