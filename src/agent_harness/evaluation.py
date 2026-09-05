from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .cases import CaseAtlas, CaseCard, MECHANISMS
from .frames import AnalogyFrame

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def lexical_score(source_situation: str, card: CaseCard) -> float:
    """Transparent surface-only baseline used as an experimental control.

    This intentionally ignores Echo's relational mechanism labels. It is not a product
    feature; it exists to test whether the structural scorer behaves differently from
    a simple lexical matcher on surface-trap stimuli.
    """
    query = _tokens(source_situation)
    card_text = " ".join([card.title, " ".join(card.surface_terms)])
    return _jaccard(query, _tokens(card_text))


def lexical_ranking(source_situation: str, atlas: CaseAtlas) -> list[tuple[str, float]]:
    rows = [(card.id, lexical_score(source_situation, card)) for card in atlas.all()]
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows


@dataclass
class RetrievalMetrics:
    n_scored: int
    recall_at_5: float
    mrr: float
    near_recall_at_5: float
    far_recall_at_5: float
    far_transfer_gap: float
    structural_over_surface_accuracy: float
    lexical_over_surface_accuracy: float
    far_structural_over_surface_accuracy: float
    far_lexical_over_surface_accuracy: float
    unrelated_false_match_rate: float
    no_match_threshold: float
    failures: list[dict]

    def as_dict(self) -> dict:
        return {
            "n_scored": self.n_scored,
            "recall_at_5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
            "near_recall_at_5": round(self.near_recall_at_5, 4),
            "far_recall_at_5": round(self.far_recall_at_5, 4),
            "far_transfer_gap": round(self.far_transfer_gap, 4),
            "structural_over_surface_accuracy": round(self.structural_over_surface_accuracy, 4),
            "lexical_over_surface_accuracy": round(self.lexical_over_surface_accuracy, 4),
            "far_structural_over_surface_accuracy": round(self.far_structural_over_surface_accuracy, 4),
            "far_lexical_over_surface_accuracy": round(self.far_lexical_over_surface_accuracy, 4),
            "unrelated_false_match_rate": round(self.unrelated_false_match_rate, 4),
            "no_match_threshold": self.no_match_threshold,
            "failures": self.failures,
        }


def _safe_rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else math.nan


def evaluate_retrieval(path: str | Path, atlas: CaseAtlas | None = None) -> RetrievalMetrics:
    atlas = atlas or CaseAtlas()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload["items"]
    threshold = float(payload["no_match_threshold"])

    recalls: list[bool] = []
    reciprocal_ranks: list[float] = []
    near_recalls: list[bool] = []
    far_recalls: list[bool] = []
    structural_pairwise: list[bool] = []
    lexical_pairwise: list[bool] = []
    far_structural_pairwise: list[bool] = []
    far_lexical_pairwise: list[bool] = []
    unrelated_false_matches: list[bool] = []
    failures: list[dict] = []

    for item in items:
        frame = item["frame"]
        structural = atlas.search(**frame, domains=[], top_k=len(atlas.all()), diversify_domains=False)
        structural_ids = [row["case_id"] for row in structural]
        structural_scores = {row["case_id"]: row["structural_score"] for row in structural}
        lexical = lexical_ranking(item["source_situation"], atlas)
        lexical_ids = [case_id for case_id, _ in lexical]
        gold = item.get("gold_case_id")
        trap = item.get("surface_trap_case_id")
        condition = item["condition"]
        if gold:
            rank = structural_ids.index(gold) + 1
            hit = rank <= 5
            recalls.append(hit)
            reciprocal_ranks.append(1.0 / rank)
            if condition == "near": near_recalls.append(hit)
            elif condition == "far": far_recalls.append(hit)
            if not hit:
                failures.append({"id": item["id"], "type": "recall_at_5", "gold": gold, "rank": rank, "top5": structural_ids[:5]})
            if trap:
                structural_ok = structural_ids.index(gold) < structural_ids.index(trap)
                lexical_ok = lexical_ids.index(gold) < lexical_ids.index(trap)
                structural_pairwise.append(structural_ok)
                lexical_pairwise.append(lexical_ok)
                if condition == "far":
                    far_structural_pairwise.append(structural_ok)
                    far_lexical_pairwise.append(lexical_ok)
                if not structural_ok:
                    failures.append({"id": item["id"], "type": "surface_trap", "gold": gold, "trap": trap, "structural_gold_score": structural_scores[gold], "structural_trap_score": structural_scores[trap]})
        else:
            top = structural[0]
            is_false_match = top["structural_score"] >= threshold
            unrelated_false_matches.append(is_false_match)
            if is_false_match:
                failures.append({"id": item["id"], "type": "unrelated_overmatch", "top_case": top["case_id"], "top_score": top["structural_score"], "threshold": threshold})

    near = _safe_rate(near_recalls)
    far = _safe_rate(far_recalls)
    return RetrievalMetrics(len(recalls), _safe_rate(recalls), sum(reciprocal_ranks)/len(reciprocal_ranks), near, far, near-far, _safe_rate(structural_pairwise), _safe_rate(lexical_pairwise), _safe_rate(far_structural_pairwise), _safe_rate(far_lexical_pairwise), _safe_rate(unrelated_false_matches), threshold, failures)


def build_selection_trials(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    trials: list[dict] = []
    for item in payload["items"]:
        trials.append({"id": f"{item['id']}__far_vs_surface", "family": item["family"], "kind": "far_vs_surface", "source": item["source"], "a": item["candidates"]["far"], "b": item["candidates"]["surface_trap"], "expected": "A"})
        trials.append({"id": f"{item['id']}__near_vs_unrelated", "family": item["family"], "kind": "near_vs_unrelated", "source": item["source"], "a": item["candidates"]["near"], "b": item["candidates"]["unrelated"], "expected": "A"})
    for item in payload.get("no_match_controls", []):
        trials.append({"id": item["id"], "family": "no_match", "kind": "no_match", "source": item["source"], "candidates": item["candidates"], "expected": "NONE"})
    return trials


def build_adversarial_selection_trials(path: str | Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{"id": i["id"], "family": i["kind"], "kind": i["kind"], "source": i["source"], "a": i["a"], "b": i["b"], "expected": i.get("expected", "A"), "control": i.get("control", "")} for i in payload["trials"]]


def selection_prompt(trial: dict) -> str:
    if trial["kind"] == "no_match":
        options = "\n".join(f"{chr(65+i)}. {text}" for i, text in enumerate(trial["candidates"]))
        return f"You are evaluating analogical similarity, not topical similarity.\n\nSOURCE:\n{trial['source']}\n\nCANDIDATES:\n{options}\n\nIf none shares a meaningful higher-order relational/causal structure with the source, answer NONE. Otherwise answer only the single letter of the strongest analogy. Do not explain."
    return f"You are evaluating analogical similarity, not topical or lexical similarity. Prefer a candidate that preserves the higher-order relational/causal structure even if its domain is different.\n\nSOURCE:\n{trial['source']}\n\nA. {trial['a']}\n\nB. {trial['b']}\n\nAnswer only A or B. Do not explain."


@dataclass
class FrameExtractionMetrics:
    n_items: int
    valid_frame_rate: float
    critical_mechanism_hit_rate: float
    mechanism_macro_f1: float
    forbidden_surface_mechanism_rate: float
    downstream_recall_at_5: float
    downstream_structural_over_surface_accuracy: float
    by_condition: dict[str, dict[str, float]]
    failures: list[dict]

    def as_dict(self) -> dict:
        return {"n_items": self.n_items, "valid_frame_rate": round(self.valid_frame_rate,4), "critical_mechanism_hit_rate": round(self.critical_mechanism_hit_rate,4), "mechanism_macro_f1": round(self.mechanism_macro_f1,4), "forbidden_surface_mechanism_rate": round(self.forbidden_surface_mechanism_rate,4), "downstream_recall_at_5": round(self.downstream_recall_at_5,4), "downstream_structural_over_surface_accuracy": round(self.downstream_structural_over_surface_accuracy,4), "by_condition": {c:{k:round(v,4) for k,v in m.items()} for c,m in self.by_condition.items()}, "failures": self.failures}


def _set_f1(expected: set[str], predicted: set[str]) -> float:
    if not expected and not predicted: return 1.0
    if not predicted: return 0.0
    tp = len(expected & predicted); precision = tp/len(predicted); recall = tp/len(expected) if expected else 1.0
    return 0.0 if precision+recall == 0 else 2*precision*recall/(precision+recall)


def frame_extraction_prompt(source_situation: str) -> str:
    mechanisms = ", ".join(MECHANISMS)
    return f"Extract one compact relational AnalogyFrame from the situation below.\nTreat surface nouns, domain, and outcome as weak evidence unless they participate in the causal structure.\nRepresent role direction explicitly. Choose 1-5 mechanism labels only from the controlled vocabulary.\nDo not search for an analogy and do not name a historical case.\n\nCONTROLLED MECHANISMS:\n{mechanisms}\n\nSITUATION:\n{source_situation}\n\nReturn exactly one JSON object with these keys and no markdown:\nroles (array of strings), goal (string), strategy (string), mechanisms (array), turning_point (string), outcome (string)."


def parse_frame_json(text: str) -> dict:
    raw=text.strip()
    if raw.startswith("```"):
        lines=raw.splitlines(); lines=lines[1:] if lines and lines[0].startswith("```") else lines; lines=lines[:-1] if lines and lines[-1].strip()=="```" else lines; raw="\n".join(lines).strip()
    start,end=raw.find("{"),raw.rfind("}")
    if start<0 or end<start: raise ValueError("no_json_object")
    payload=json.loads(raw[start:end+1])
    if not isinstance(payload,dict): raise ValueError("frame_not_object")
    return payload


def evaluate_frame_predictions(path: str | Path, predictions: dict[str, dict], atlas: CaseAtlas | None = None) -> FrameExtractionMetrics:
    atlas=atlas or CaseAtlas(); items=json.loads(Path(path).read_text(encoding="utf-8"))["items"]
    valid=[]; critical_hits=[]; f1s=[]; forbidden_hits=[]; recalls=[]; pairwise=[]; failures=[]; condition_rows={}
    for item in items:
        condition=item["condition"]; bucket=condition_rows.setdefault(condition,{"valid":[],"critical":[],"recall5":[],"pairwise":[]}); raw=predictions.get(item["id"])
        if raw is None:
            valid.append(False); critical_hits.append(False); f1s.append(0.0); forbidden_hits.append(False); recalls.append(False); pairwise.append(False); [bucket[k].append(False) for k in bucket]; failures.append({"id":item["id"],"type":"missing_prediction"}); continue
        try: frame=AnalogyFrame.from_dict(raw)
        except Exception as exc:
            valid.append(False); critical_hits.append(False); f1s.append(0.0); forbidden_hits.append(False); recalls.append(False); pairwise.append(False); [bucket[k].append(False) for k in bucket]; failures.append({"id":item["id"],"type":"invalid_frame","error":type(exc).__name__}); continue
        valid.append(True); bucket["valid"].append(True); predicted=set(frame.mechanisms); expected=set(item["expected_frame"]["mechanisms"]); critical=set(item["critical_mechanisms"]); forbidden=set(item.get("forbidden_surface_mechanisms",[])); critical_ok=critical<=predicted; forbidden_hit=bool(forbidden & predicted); f1=_set_f1(expected,predicted); critical_hits.append(critical_ok); forbidden_hits.append(forbidden_hit); f1s.append(f1); bucket["critical"].append(critical_ok)
        ranking=atlas.search(**frame.as_dict(),domains=[],top_k=len(atlas.all()),diversify_domains=False); ids=[r["case_id"] for r in ranking]; gold=item["gold_case_id"]; trap=item["surface_trap_case_id"]; recall5=gold in ids[:5]; structural_ok=ids.index(gold)<ids.index(trap); recalls.append(recall5); pairwise.append(structural_ok); bucket["recall5"].append(recall5); bucket["pairwise"].append(structural_ok)
        if not critical_ok: failures.append({"id":item["id"],"type":"critical_mechanism_miss","required":sorted(critical),"predicted":sorted(predicted)})
        if forbidden_hit: failures.append({"id":item["id"],"type":"surface_mechanism_intrusion","forbidden_hit":sorted(forbidden & predicted)})
        if not recall5: failures.append({"id":item["id"],"type":"downstream_recall_at_5","gold":gold,"top5":ids[:5]})
        if not structural_ok: failures.append({"id":item["id"],"type":"downstream_surface_trap","gold":gold,"trap":trap,"gold_rank":ids.index(gold)+1,"trap_rank":ids.index(trap)+1})
    by_condition={c:{"valid_frame_rate":_safe_rate(r["valid"]),"critical_mechanism_hit_rate":_safe_rate(r["critical"]),"downstream_recall_at_5":_safe_rate(r["recall5"]),"downstream_structural_over_surface_accuracy":_safe_rate(r["pairwise"])} for c,r in condition_rows.items()}
    return FrameExtractionMetrics(len(items),_safe_rate(valid),_safe_rate(critical_hits),sum(f1s)/len(f1s) if f1s else math.nan,_safe_rate(forbidden_hits),_safe_rate(recalls),_safe_rate(pairwise),by_condition,failures)
