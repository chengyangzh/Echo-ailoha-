from __future__ import annotations

from typing import Any

from ..cases import CaseAtlas, normalize_domains, normalize_mechanisms
from ..external import WikipediaAdapter
from ..frames import analogy_search_properties
from ..tooling import Tool, ToolSpec


CORE_COVERAGE_THRESHOLD = 0.48
CORE_MECHANISM_COVERAGE_THRESHOLD = 0.67


class SearchTool(Tool):
    """Candidate generation over either Echo's Core Atlas or Wikipedia.

    The tool deliberately does not decide whether an external result is a good analogy.
    It only returns compact candidates that the model may inspect with read_case.
    """

    spec = ToolSpec(
        name="search",
        description=(
            "Search for candidate analogies. Use source='core' for deterministic structural "
            "retrieval over Echo's curated Core Atlas. Use source='wikipedia' only when the Core "
            "Atlas is insufficient and broad historical/political/business/technology coverage "
            "is useful. Wikipedia results are lightweight candidates, not validated CaseCards; "
            "call read_case before relying on one."
        ),
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": ["core", "wikipedia"]},
                "query": {
                    "type": "string",
                    "description": (
                        "Mechanism-oriented lexical probe for Wikipedia. Use an empty string for "
                        "Core Atlas search."
                    ),
                },
                **analogy_search_properties(),
                "domains": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": 8,
                    "description": (
                        "Optional TARGET-case domain filters. Leave [] for normal cross-domain "
                        "analogy search. Only add a domain when the USER explicitly asks to "
                        "restrict the analogies to that domain; do not infer filters from the "
                        "source situation's topic (e.g. cybersecurity). Open labels are accepted "
                        "and normalized/ignored safely by the tool."
                    ),
                },
                "top_k": {"type": "integer", "minimum": 1, "maximum": 8},
                "diversify_domains": {"type": "boolean"},
            },
            "required": [],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        atlas: CaseAtlas | None = None,
        wikipedia: WikipediaAdapter | None = None,
    ) -> None:
        self.atlas = atlas or CaseAtlas()
        self.wikipedia = wikipedia or WikipediaAdapter()

    async def execute(
        self,
        *,
        user_id: str,
        session_id: str,
        source: str = "core",
        query: str = "",
        roles: list[str] | None = None,
        goal: str = "",
        strategy: str = "",
        mechanisms: list[str] | None = None,
        turning_point: str = "",
        outcome: str = "",
        domains: list[str] | None = None,
        top_k: int = 5,
        diversify_domains: bool = True,
        **_: Any,
    ) -> Any:
        roles = list(roles or [])
        mechanisms = list(mechanisms or [])
        domains = list(domains or [])

        if source == "core":
            canonical_mechanisms, mechanism_mapped, mechanism_unknown = normalize_mechanisms(mechanisms)
            canonical_domains, domain_mapped, domain_unknown = normalize_domains(domains)

            effective_domains = canonical_domains
            domain_filter_relaxed = bool(domain_mapped or domain_unknown)
            if domain_filter_relaxed:
                effective_domains = []

            normalization = {
                "mechanisms": {
                    "input": mechanisms,
                    "effective": canonical_mechanisms,
                    "mapped": mechanism_mapped,
                    "ignored_unknown": mechanism_unknown,
                },
                "domains": {
                    "input": domains,
                    "effective_filters": effective_domains,
                    "mapped": domain_mapped,
                    "ignored_unknown": domain_unknown,
                    "filter_relaxed_for_cross_domain_recall": domain_filter_relaxed,
                },
            }
            notices = [
                "Candidates are the best matches found in Echo's curated Core Atlas, not proof "
                "that no better analogy exists outside the accessible search space."
            ]
            missing_frame_fields = [
                name for name, value in {
                    "roles": roles, "goal": goal, "strategy": strategy,
                    "mechanisms": mechanisms, "turning_point": turning_point, "outcome": outcome,
                }.items() if not value
            ]
            if missing_frame_fields:
                notices.append(
                    "Search used a partial AnalogyFrame; missing fields were treated as neutral rather "
                    "than invented: " + ", ".join(missing_frame_fields) + "."
                )
            if mechanism_unknown:
                notices.append(
                    "Some open mechanism labels are outside the Core Atlas vocabulary and were "
                    "ignored for deterministic mechanism scoring; revise the frame if they are "
                    "causally important."
                )
            if domain_filter_relaxed:
                notices.append(
                    "Non-canonical/inferred domain labels were not used as hard filters so that "
                    "surface-domain guesses do not block far analogies. Use canonical target "
                    "domains only when the user explicitly requests a domain restriction."
                )

            candidates = self.atlas.search(
                roles=roles,
                goal=goal,
                strategy=strategy,
                mechanisms=canonical_mechanisms,
                turning_point=turning_point,
                outcome=outcome,
                domains=effective_domains,
                top_k=top_k,
                diversify_domains=diversify_domains,
            )

            top = candidates[0] if candidates else None
            top_score = float(top.get("structural_score", 0.0)) if top else 0.0
            matched_mechanisms = (
                list((top.get("evidence") or {}).get("matched_mechanisms") or [])
                if top else []
            )
            mechanism_coverage = (
                len(set(matched_mechanisms)) / len(set(canonical_mechanisms))
                if canonical_mechanisms else 0.0
            )
            weak_reasons: list[str] = []
            if not candidates:
                weak_reasons.append("no_candidates")
            if top_score < CORE_COVERAGE_THRESHOLD:
                weak_reasons.append("low_structural_score")
            if canonical_mechanisms and not matched_mechanisms:
                weak_reasons.append("no_core_mechanism_match")
            elif canonical_mechanisms and mechanism_coverage < CORE_MECHANISM_COVERAGE_THRESHOLD:
                weak_reasons.append("partial_mechanism_coverage")
            if mechanism_unknown:
                weak_reasons.append("unmapped_causal_mechanism")

            coverage = {
                "status": "weak" if weak_reasons else "strong",
                "top_structural_score": round(top_score, 4),
                "threshold": CORE_COVERAGE_THRESHOLD,
                "matched_mechanisms": matched_mechanisms,
                "mechanism_coverage": round(mechanism_coverage, 4),
                "mechanism_coverage_threshold": CORE_MECHANISM_COVERAGE_THRESHOLD,
                "unmapped_mechanisms": mechanism_unknown,
                "reasons": weak_reasons,
            }
            if weak_reasons:
                notices.append(
                    "Core Atlas coverage is weak under the deterministic coverage check; "
                    "widen candidate discovery to Wikipedia before concluding, or use a clearly labeled constructed analogy if no retrieved case survives semantic review."
                )

            return {
                "scope": "core_atlas",
                "exhaustive": False,
                "notice": " ".join(notices),
                "normalization": normalization,
                "coverage": coverage,
                "candidates": candidates,
            }

        if source == "wikipedia":
            candidates = await self.wikipedia.search(query, limit=top_k)
            return {
                "scope": "wikipedia",
                "exhaustive": False,
                "notice": (
                    "Wikipedia search is candidate discovery only. Results may be lexically related "
                    "without sharing the target causal structure. Inspect promising pages with "
                    "read_case and let the LLM judge the analogy."
                ),
                "query": query,
                "candidates": candidates,
            }

        raise ValueError(f"unsupported_search_source: {source}")
