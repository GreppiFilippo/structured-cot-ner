"""Parsing and scoring helpers shared by evaluate.py, rescore.py and
sanity_check.py. Pure Python + numpy/scipy only (no torch/transformers),
so existing predictions.jsonl files can be re-scored on any machine,
without a GPU and without regenerating anything.

METRIC_VERSION history:
  1 — original metric from bioner-evaluation-fix.ipynb: token IoU on
      whitespace-split spans, Hungarian matching per type.
  2 — (2026-09-24) three fixes, applied identically to all three models:
      * span tokens are word characters only (\\w+), so punctuation
        spacing no longer matters: gold spans are built as
        " ".join(tokens) ("5 , 6 - dihydro", "( CD25 )") while the text
        shown to the model is Moses-detokenized ("5, 6 - dihydro",
        "(CD25)") — a model copying the span verbatim from the text got
        only partial IoU under v1;
      * spans are de-duplicated within each type (and across types for
        the type-agnostic metric): the output format is a list of
        strings per type, so repeating the same surface form carries no
        extra information, and the CoT prompt explicitly asks to report
        each entity once;
      * reasoning blocks (<think>...</think>, MedGemma's
        <unused94>thought...<unused95>) are removed before parsing, so a
        JSON draft written while reasoning is never taken as the answer.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import numpy as np
from scipy import optimize as sp_optimize

METRIC_VERSION = 2


# --- Span matching ---


_WORD_RE = re.compile(r"\w+")


def _tokenize(span):
    return set(_WORD_RE.findall(span.lower()))


def iou(a, b):
    sa, sb = _tokenize(a), _tokenize(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def dedupe_spans(spans):
    """Keeps the first occurrence of each span, compared on its word
    tokens (case and punctuation ignored, same normalization as iou())."""
    seen, unique = set(), []
    for s in spans:
        key = tuple(_WORD_RE.findall(s.lower()))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def compute_entity_counts(gt_spans, pred_spans):
    gc_, pc = len(gt_spans), len(pred_spans)
    if gc_ == 0 or pc == 0:
        return 0.0, gc_, pc
    M = np.array([[iou(g, p) for p in pred_spans] for g in gt_spans])
    r, c = sp_optimize.linear_sum_assignment(-M)
    return float(M[r, c].sum()), gc_, pc


def f1_from_counts(tp, gt, pred):
    d = gt + pred
    return (2.0 * tp / d) if d else 1.0


# --- Reply cleaning ---


_THOUGHT_BLOCK_RE = re.compile(r"<think>.*?</think>|<unused94>.*?<unused95>", re.DOTALL)
# an opening marker with no closing one: generation was cut off while still
# reasoning, so there is no answer after it
_UNCLOSED_THOUGHT_RE = re.compile(r"(?:<think>|<unused94>).*\Z", re.DOTALL)
_SPECIAL_TOKEN_RE = re.compile(r"<(?:pad|eos|bos|start_of_turn)>")


def clean_reply(text):
    """Returns only the model's final answer: stops at <end_of_turn>,
    drops reasoning blocks (including a closing tag whose opening tag was
    part of the prompt, as with DeepSeek-R1-style templates), drops an
    unterminated reasoning block entirely, and removes leftover special
    tokens."""
    text = text.split("<end_of_turn>")[0]
    text = _THOUGHT_BLOCK_RE.sub("", text)
    for closing in ("</think>", "<unused95>"):
        if closing in text:
            text = text.rsplit(closing, 1)[1]
    text = _UNCLOSED_THOUGHT_RE.sub("", text)
    text = _SPECIAL_TOKEN_RE.sub("", text)
    return text.strip()


# --- Parsing ---


class ParseStatus:
    OK = "ok"
    FENCE_STRIPPED = "fence_stripped"
    EXTRACTED_OBJECT = "extracted_object"
    WRONG_TOP_LEVEL = "wrong_top_level"
    PARSE_ERROR = "parse_error"


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def iter_json_objects(text):
    """Yields every top-level balanced {...} substring, in order. String
    literals are tracked so braces inside them are ignored."""
    depth, start, in_string, escape = 0, None, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def _loads_or_none(candidate):
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


class PredictionParser:
    """Closed-schema output: {TYPE: [span, ...]} (baseline, label-only)."""

    @classmethod
    def parse(cls, raw_text):
        text = clean_reply(raw_text)
        for candidate, status in [
            (text, ParseStatus.OK),
            (_FENCE_RE.sub("", text).strip(), ParseStatus.FENCE_STRIPPED),
        ]:
            obj = _loads_or_none(candidate) if candidate else None
            if obj is None:
                continue
            if not isinstance(obj, dict):
                return {}, ParseStatus.WRONG_TOP_LEVEL
            return cls.coerce_values(obj), status
        for candidate in iter_json_objects(text):
            obj = _loads_or_none(candidate)
            if isinstance(obj, dict):
                return cls.coerce_values(obj), ParseStatus.EXTRACTED_OBJECT
        return {}, ParseStatus.PARSE_ERROR

    @staticmethod
    def coerce_values(obj):
        cleaned = {}
        for k, v in obj.items():
            if isinstance(v, list):
                cleaned[k] = [str(x) for x in v if isinstance(x, (str, int, float))]
            elif isinstance(v, str):
                cleaned[k] = [v]
            else:
                cleaned[k] = []
        return cleaned


class CoTPredictionParser:
    """Open-schema CoT output: reasoning prose followed by
    {"schema": [...], "entities": [{"text":..., "label":..., "reason":...}],
    "excluded": [...]}. Returns the same {TYPE: [span, ...]} shape as
    PredictionParser, grouping entities by their free-form label.

    Takes the LAST JSON object that has an "entities" key: the answer
    block always comes after the reasoning, so an earlier object can only
    be something quoted or drafted inside the reasoning."""

    @classmethod
    def parse(cls, raw_text):
        text = clean_reply(raw_text)
        for candidate, status in [
            (text, ParseStatus.OK),
            (_FENCE_RE.sub("", text).strip(), ParseStatus.FENCE_STRIPPED),
        ]:
            obj = _loads_or_none(candidate) if candidate else None
            if isinstance(obj, dict) and "entities" in obj:
                return cls._group_by_label(obj), status
        found = None
        for candidate in iter_json_objects(text):
            obj = _loads_or_none(candidate)
            if isinstance(obj, dict) and "entities" in obj:
                found = obj
        if found is not None:
            return cls._group_by_label(found), ParseStatus.EXTRACTED_OBJECT
        return {}, ParseStatus.PARSE_ERROR

    @staticmethod
    def _group_by_label(obj):
        grouped = defaultdict(list)
        entities = obj.get("entities", [])
        if not isinstance(entities, list):
            return {}
        for e in entities:
            if isinstance(e, dict) and isinstance(e.get("text"), str) and e.get("text"):
                label = e.get("label") if isinstance(e.get("label"), str) and e.get("label") else "UNLABELED"
                grouped[label].append(e["text"])
        return dict(grouped)


def parse_gold(gold_json):
    try:
        return PredictionParser.coerce_values(json.loads(gold_json))
    except Exception:
        return {}


def select_gold(row, model_type):
    """Returns (gold, gold_source) for one subset row.

    baseline / label-only are asked about a specific list of entity types
    (real ones plus sampled distractors), so their gold is exactly the
    assistant turn of that closed-schema prompt.

    The CoT model is never told which types to look for and extracts
    every entity in the text. Scoring it against the closed-schema gold
    would count every real entity of a type that was not sampled into
    that row's prompt as a false positive (and, on rows where no real
    type was sampled at all, the gold is empty). It is scored against
    "all_gold" instead — every annotated span of the row, all types —
    when the subset has that column (build_eval_subset.py, 2026-09-24+).
    """
    closed = parse_gold(row["messages"][-1]["content"])
    if model_type != "cot":
        return closed, "prompt_types"
    if row.get("all_gold") is not None:
        return parse_gold(row["all_gold"]), "all_gold"
    return closed, "prompt_types_only_MISSING_all_gold"


# --- Scoring ---


class Scorer:
    """Accumulates type-exact and type-agnostic counts over rows.

    type-exact: IoU + Hungarian matching within each type, types must
    match by name. type-agnostic: same matching over all spans of the row
    pooled together, type ignored — the only metric that is comparable
    across the closed-schema models and the open-schema CoT model, which
    names its own types."""

    def __init__(self):
        self.tp = self.gt = self.pred = 0.0
        self.tp_ta = self.gt_ta = self.pred_ta = 0.0
        self.per_entity = defaultdict(lambda: [0.0, 0, 0])
        self.parse_counts = Counter()
        self.n = 0

    def add(self, gold, pred, status):
        """Scores one row; returns (row_f1, row_f1_type_agnostic)."""
        self.n += 1
        self.parse_counts[status] += 1

        tp_s = gt_s = pred_s = 0.0
        for etype in set(gold) | set(pred):
            tp, gcount, pcount = compute_entity_counts(
                dedupe_spans(gold.get(etype, [])), dedupe_spans(pred.get(etype, [])))
            tp_s += tp
            gt_s += gcount
            pred_s += pcount
            acc = self.per_entity[etype]
            acc[0] += tp
            acc[1] += gcount
            acc[2] += pcount
        self.tp += tp_s
        self.gt += gt_s
        self.pred += pred_s

        gold_flat = dedupe_spans([s for spans in gold.values() for s in spans])
        pred_flat = dedupe_spans([s for spans in pred.values() for s in spans])
        tp_ta, gt_ta, pred_ta = compute_entity_counts(gold_flat, pred_flat)
        self.tp_ta += tp_ta
        self.gt_ta += gt_ta
        self.pred_ta += pred_ta

        return f1_from_counts(tp_s, gt_s, pred_s), f1_from_counts(tp_ta, gt_ta, pred_ta)

    def metrics(self):
        return {
            "metric_version": METRIC_VERSION,
            "n_samples": self.n,
            "F1_micro": f1_from_counts(self.tp, self.gt, self.pred),
            "precision": (self.tp / self.pred) if self.pred else 0.0,
            "recall": (self.tp / self.gt) if self.gt else 0.0,
            "F1_micro_type_agnostic": f1_from_counts(self.tp_ta, self.gt_ta, self.pred_ta),
            "precision_type_agnostic": (self.tp_ta / self.pred_ta) if self.pred_ta else 0.0,
            "recall_type_agnostic": (self.tp_ta / self.gt_ta) if self.gt_ta else 0.0,
            "TP": float(self.tp), "GT": int(self.gt), "PRED": int(self.pred),
            "TP_type_agnostic": float(self.tp_ta), "GT_type_agnostic": int(self.gt_ta),
            "PRED_type_agnostic": int(self.pred_ta),
            "parse_status_counts": dict(self.parse_counts),
        }

    def per_entity_metrics(self):
        per_ent = {}
        for etype, (tp, gcount, pcount) in self.per_entity.items():
            if gcount == 0 and pcount == 0:
                continue
            per_ent[etype] = {
                "precision": (tp / pcount) if pcount else 0.0,
                "recall": (tp / gcount) if gcount else 0.0,
                "f1": f1_from_counts(tp, gcount, pcount),
                "gt_count": int(gcount), "pred_count": int(pcount),
            }
        return dict(sorted(per_ent.items(), key=lambda kv: kv[1]["f1"]))


def write_results(save_folder, scorer, records, extra_metrics=None, worst_k=10):
    """Writes metrics.json, per_entity_metrics.json, predictions.jsonl and
    worst_examples.md into save_folder (a pathlib.Path)."""
    metrics = scorer.metrics()
    metrics.update(extra_metrics or {})
    with open(save_folder / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(save_folder / "per_entity_metrics.json", "w") as f:
        json.dump(scorer.per_entity_metrics(), f, indent=2)
    with open(save_folder / "predictions.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # worst rows by the metric that is meaningful for every model type
    worst = sorted(records, key=lambda r: r["f1_type_agnostic"])[:worst_k]
    with open(save_folder / "worst_examples.md", "w") as f:
        for r in worst:
            f.write(f"## idx {r['idx']}  F1_type_agnostic={r['f1_type_agnostic']:.3f}  F1={r['f1']:.3f}\n")
            f.write(f"gt: {json.dumps(r['gt'], ensure_ascii=False)}\n")
            f.write(f"pred: {json.dumps(r['pred_parsed'], ensure_ascii=False)}\n")
            f.write(f"raw: {r['pred_raw'][-300:]}\n---\n")
    return metrics
