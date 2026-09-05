from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MECHANISMS = (
    "adaptation",
    "adaptation_failure",
    "collective_action",
    "commitment",
    "coordination",
    "credibility_decay",
    "dependency",
    "escalation_control",
    "feedback_failure",
    "feedback_loop",
    "information_asymmetry",
    "local_global_tradeoff",
    "model_world_mismatch",
    "network_effect",
    "overconfidence",
    "path_dependence",
    "perverse_incentive",
    "proxy_failure",
    "resource_depletion",
    "resource_matching",
    "second_order_effect",
    "selection_pressure",
    "self_defeating_strategy",
    "short_term_long_term_tradeoff",
    "signaling",
    "strategic_deception",
    "strategic_indirection",
)

DOMAINS = (
    "business",
    "chinese_history",
    "fable",
    "institutions",
    "myth_history",
    "science_technology",
    "strategy",
    "world_history",
)

MECHANISM_ALIASES = {
    "alert_fatigue": "credibility_decay",
    "alarm_fatigue": "credibility_decay",
    "trust_erosion": "credibility_decay",
    "signal_dilution": "credibility_decay",
    "warning_fatigue": "credibility_decay",
    "habituation": "adaptation",
    "information_overload": "feedback_failure",
    "goal_displacement": "proxy_failure",
    "metric_gaming": "proxy_failure",
    "lock_in": "path_dependence",
    "role_lock_in": "path_dependence",
    "backfire": "self_defeating_strategy",
    "unintended_consequence": "second_order_effect",
    "unintended_consequences": "second_order_effect",
}

DOMAIN_ALIASES = {
    "engineering": "science_technology",
    "technology": "science_technology",
    "science": "science_technology",
    "medicine": "science_technology",
    "medical": "science_technology",
    "cybersecurity": "science_technology",
    "software": "science_technology",
    "finance": "business",
    "economics": "business",
    "commerce": "business",
    "startup": "business",
    "corporate": "business",
    "organization": "institutions",
    "organizations": "institutions",
    "institution": "institutions",
    "politics": "world_history",
    "political_history": "world_history",
    "diplomacy": "world_history",
    "military_history": "world_history",
}


def _canonical_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def normalize_mechanisms(values: Iterable[str]) -> tuple[list[str], dict[str, str], list[str]]:
    """Map open semantic labels onto the Core Atlas vocabulary without crashing.

    Provider-facing schemas deliberately accept unseen semantic labels. The curated atlas,
    however, has a finite vocabulary. Known labels pass through, selected common synonyms
    are mapped deterministically, and genuinely unknown labels are reported back to the
    caller instead of becoming provider-level validation failures.
    """
    normalized: list[str] = []
    mapped: dict[str, str] = {}
    unknown: list[str] = []
    for raw in values:
        label = _canonical_label(raw)
        if label in MECHANISMS:
            canonical = label
        else:
            canonical = MECHANISM_ALIASES.get(label)
        if canonical:
            if canonical not in normalized:
                normalized.append(canonical)
            if canonical != label:
                mapped[raw] = canonical
        else:
            unknown.append(raw)
    return normalized, mapped, unknown


def normalize_domains(values: Iterable[str]) -> tuple[list[str], dict[str, str], list[str]]:
    """Normalize optional *target-case* domain filters for the finite Core Atlas.

    Unknown filters are ignored rather than treated as malformed calls. This is important
    for cross-domain analogy: a user's source domain (e.g. cybersecurity) should not make
    a far-domain fable impossible to retrieve. Callers receive the ignored labels as an
    observation and can revise if the user explicitly intended a domain restriction.
    """
    normalized: list[str] = []
    mapped: dict[str, str] = {}
    unknown: list[str] = []
    for raw in values:
        label = _canonical_label(raw)
        if label in DOMAINS:
            canonical = label
        else:
            canonical = DOMAIN_ALIASES.get(label)
        if canonical:
            if canonical not in normalized:
                normalized.append(canonical)
            if canonical != label:
                mapped[raw] = canonical
        else:
            unknown.append(raw)
    return normalized, mapped, unknown


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass(frozen=True)
class CaseCard:
    id: str
    title: str
    domain: str
    era: str
    case_type: str
    summary: str
    roles: list[str]
    goal: str
    strategy: str
    mechanisms: list[str]
    turning_point: str
    outcome: str
    principle: str
    surface_terms: list[str]
    source_note: str

    @classmethod
    def from_dict(cls, raw: dict) -> "CaseCard":
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown CaseCard fields for {raw.get('id')}: {sorted(unknown)}")
        card = cls(**raw)
        bad_mechanisms = sorted(set(card.mechanisms) - set(MECHANISMS))
        if bad_mechanisms:
            raise ValueError(f"Unknown mechanisms in {card.id}: {bad_mechanisms}")
        if card.domain not in DOMAINS:
            raise ValueError(f"Unknown domain in {card.id}: {card.domain}")
        return card

    def compact(self) -> dict:
        return {
            "case_id": self.id,
            "title": self.title,
            "domain": self.domain,
            "era": self.era,
            "case_type": self.case_type,
            "summary": self.summary,
            "matched_via": "core_atlas",
            "provider": "core_atlas",
            "candidate_only": True,
        }

    def full(self) -> dict:
        return {
            "case_id": self.id,
            "title": self.title,
            "domain": self.domain,
            "era": self.era,
            "case_type": self.case_type,
            "summary": self.summary,
            "roles": self.roles,
            "goal": self.goal,
            "strategy": self.strategy,
            "mechanisms": self.mechanisms,
            "turning_point": self.turning_point,
            "outcome": self.outcome,
            "principle": self.principle,
            "source_note": self.source_note,
            "provider": "core_atlas",
        }


class CaseAtlas:
    """Small, curated, deterministic case source used for retrieval and evaluation.

    The atlas is intentionally not exhaustive. Its job is to make structural retrieval
    inspectable and testable before adding broad external sources.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        source = Path(path) if path else Path(__file__).with_name("data") / "cases.json"
        raw = json.loads(source.read_text(encoding="utf-8"))
        self._cards = {item["id"]: CaseCard.from_dict(item) for item in raw}
        if len(self._cards) != len(raw):
            raise ValueError("Duplicate case id in Core Atlas")

    def get(self, case_id: str) -> CaseCard:
        try:
            return self._cards[case_id]
        except KeyError as exc:
            raise ValueError(f"case_not_found: {case_id}") from exc

    def all(self) -> list[CaseCard]:
        return list(self._cards.values())

    def search(
        self,
        *,
        roles: list[str],
        goal: str,
        strategy: str,
        mechanisms: list[str],
        turning_point: str,
        outcome: str,
        domains: list[str],
        top_k: int,
        diversify_domains: bool,
    ) -> list[dict]:
        # Search receives already-normalized semantic labels from SearchTool. Keep this
        # method deterministic and finite-vocabulary, but fail only on programmer misuse.
        bad = sorted(set(mechanisms) - set(MECHANISMS))
        if bad:
            raise ValueError(f"noncanonical_mechanism_reached_atlas: {', '.join(bad)}")
        unknown_domains = sorted(set(domains) - set(DOMAINS))
        if unknown_domains:
            raise ValueError(f"noncanonical_domain_reached_atlas: {', '.join(unknown_domains)}")

        query_role_tokens = _tokens(" ".join(roles))
        query_goal = _tokens(goal)
        query_strategy = _tokens(strategy)
        query_turn = _tokens(turning_point)
        query_outcome = _tokens(outcome)
        query_mechanisms = set(mechanisms)

        scored: list[tuple[float, CaseCard, dict]] = []
        for card in self._cards.values():
            if domains and card.domain not in domains:
                continue

            mechanism_score = _jaccard(query_mechanisms, card.mechanisms)
            role_score = _jaccard(query_role_tokens, _tokens(" ".join(card.roles)))
            goal_score = _jaccard(query_goal, _tokens(card.goal))
            strategy_score = _jaccard(query_strategy, _tokens(card.strategy))
            turn_score = _jaccard(query_turn, _tokens(card.turning_point))
            outcome_score = _jaccard(query_outcome, _tokens(card.outcome))

            # Deliberately make relational mechanism the dominant signal. Text fields help
            # break ties but cannot overwhelm a strong mechanism match.
            score = (
                0.60 * mechanism_score
                + 0.08 * role_score
                + 0.08 * goal_score
                + 0.10 * strategy_score
                + 0.07 * turn_score
                + 0.07 * outcome_score
            )
            matched_mechanisms = sorted(query_mechanisms & set(card.mechanisms))
            evidence = {
                "mechanism_overlap": round(mechanism_score, 4),
                "matched_mechanisms": matched_mechanisms,
                "role_overlap": round(role_score, 4),
                "strategy_overlap": round(strategy_score, 4),
            }
            scored.append((score, card, evidence))

        scored.sort(key=lambda row: (-row[0], row[1].id))

        selected: list[tuple[float, CaseCard, dict]] = []
        if diversify_domains:
            used_domains: set[str] = set()
            for row in scored:
                if row[1].domain not in used_domains:
                    selected.append(row)
                    used_domains.add(row[1].domain)
                    if len(selected) == top_k:
                        break
            if len(selected) < top_k:
                selected_ids = {row[1].id for row in selected}
                for row in scored:
                    if row[1].id not in selected_ids:
                        selected.append(row)
                        if len(selected) == top_k:
                            break
        else:
            selected = scored[:top_k]

        results: list[dict] = []
        for rank, (score, card, evidence) in enumerate(selected, start=1):
            item = card.compact()
            item.update({
                "rank": rank,
                "structural_score": round(score, 4),
                "evidence": evidence,
            })
            results.append(item)
        return results
