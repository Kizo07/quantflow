"""Fixed-budget research context packet builder for the Research Knowledge Plane.

Turns fused retrieval output into the agent-facing ``RESEARCH CONTEXT``
packet from ``knowledge_base.md`` § "Return a research context packet, not
raw search results" — consensus, prior experiments, failures, conflicts,
assumptions, skills, and open questions — instead of raw search results.
The packet fits a fixed context budget (:class:`PacketBudget` caps both
estimated tokens and UTF-8 bytes); sections fill greedily in priority
order with per-section share caps, and anything that does not fit is
reported via ``truncated`` flags and warnings rather than silently
dropped.

Tiering follows the MemGPT/OpenViking L0/L1/L2 model the KB adopts:

* **L0/L1** (topic abstracts, project dossiers) are derived summaries owned
  by Phase 4 consolidation. Until then :func:`build_context_packet` emits
  explicit :class:`SummaryPlaceholder` rows with
  ``status == "pending_phase4"`` — marked placeholders, never invented
  summaries.
* **L2** is the evidence itself. Packet items carry L2 *pointers*
  (experiment / artifact / finding ids) so agents open evidence on demand
  under the citation-lock rule (a finding may be cited only after one of
  its supporting evidence records was opened during the run).

Storage boundary: none — this module renders packets from in-memory
:class:`~deerflow.knowledge.retrieval.fusion.FusedCandidate` records plus
caller-supplied conflict views and open questions. It performs no I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from deerflow.knowledge.retrieval.fusion import FusedCandidate, FusionResult
from deerflow.knowledge.write_api import KnowledgeValidationError

__all__ = [
    "SECTION_CONSENSUS",
    "SECTION_EXPERIMENTS",
    "SECTION_FAILURES",
    "SECTION_CONFLICTS",
    "SECTION_ASSUMPTIONS",
    "SECTION_SKILLS",
    "SECTION_OPEN_QUESTIONS",
    "SECTION_ORDER",
    "SECTION_TITLES",
    "DEFAULT_SECTION_SHARES",
    "L0_LEVEL",
    "L1_LEVEL",
    "PLACEHOLDER_PENDING_PHASE4",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_SNIPPET_CHARS",
    "PacketBudget",
    "PacketItem",
    "PacketSection",
    "SummaryPlaceholder",
    "ConflictView",
    "ContextPacket",
    "estimate_tokens",
    "utf8_bytes",
    "build_context_packet",
]

#: Findings the run can treat as current consensus (validated first by score).
SECTION_CONSENSUS = "consensus"
#: Closest prior experiments (non-failure runs).
SECTION_EXPERIMENTS = "prior-experiments"
#: Relevant failures / negative evidence (dedicated channel output).
SECTION_FAILURES = "failures"
#: Known contradictions: disputed findings + conflict sets.
SECTION_CONFLICTS = "conflicts"
#: Important assumptions behind the retrieved research.
SECTION_ASSUMPTIONS = "assumptions"
#: Reusable priors / skills.
SECTION_SKILLS = "skills"
#: Open questions the retrieval surfaced but did not answer.
SECTION_OPEN_QUESTIONS = "open-questions"

#: Sections in fill/render priority order.
SECTION_ORDER = (
    SECTION_CONSENSUS,
    SECTION_EXPERIMENTS,
    SECTION_FAILURES,
    SECTION_CONFLICTS,
    SECTION_ASSUMPTIONS,
    SECTION_SKILLS,
    SECTION_OPEN_QUESTIONS,
)

#: Human-readable titles used by :meth:`ContextPacket.to_text`.
SECTION_TITLES = {
    SECTION_CONSENSUS: "Current consensus",
    SECTION_EXPERIMENTS: "Closest prior experiments",
    SECTION_FAILURES: "Relevant failures / negative evidence",
    SECTION_CONFLICTS: "Known contradictions",
    SECTION_ASSUMPTIONS: "Important assumptions",
    SECTION_SKILLS: "Reusable priors / skills",
    SECTION_OPEN_QUESTIONS: "Open questions",
}

#: Default per-section budget shares (fractions of the packet budget; sum to 1.0).
DEFAULT_SECTION_SHARES = {
    SECTION_CONSENSUS: 0.25,
    SECTION_EXPERIMENTS: 0.20,
    SECTION_FAILURES: 0.20,
    SECTION_CONFLICTS: 0.10,
    SECTION_ASSUMPTIONS: 0.10,
    SECTION_SKILLS: 0.10,
    SECTION_OPEN_QUESTIONS: 0.05,
}

#: L0 tier: single topic/project abstract.
L0_LEVEL = "L0"
#: L1 tier: per-topic dossiers.
L1_LEVEL = "L1"
#: Placeholder status marking summaries owned by Phase 4 consolidation.
PLACEHOLDER_PENDING_PHASE4 = "pending_phase4"

#: Default packet token budget (estimated tokens, see :func:`estimate_tokens`).
DEFAULT_MAX_TOKENS = 4000
#: Default packet byte budget (UTF-8 bytes of the rendered packet).
DEFAULT_MAX_BYTES = 16384
#: Default per-item snippet cap (characters; longer text truncates with an ellipsis marker).
DEFAULT_MAX_SNIPPET_CHARS = 600


def estimate_tokens(text: str) -> int:
    """Estimate token cost with a deterministic heuristic (``ceil(len / 4)``, minimum 1).

    This is a budgeting approximation (≈4 chars/token for English prose),
    not a tokenizer: it keeps packet sizing deterministic and dependency-free
    until a measured tokenizer is wired in (Phase 6 specialization).
    """
    if not isinstance(text, str):
        raise KnowledgeValidationError(f"text must be a string, got {type(text).__name__}.")
    return max(1, (len(text) + 3) // 4)


def utf8_bytes(text: str) -> int:
    """Return the UTF-8 byte length of a string."""
    if not isinstance(text, str):
        raise KnowledgeValidationError(f"text must be a string, got {type(text).__name__}.")
    return len(text.encode("utf-8"))


@dataclass(frozen=True)
class PacketBudget:
    """Fixed packet budget: token cap + byte cap + per-item snippet cap."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    max_bytes: int = DEFAULT_MAX_BYTES
    max_snippet_chars: int = DEFAULT_MAX_SNIPPET_CHARS

    def __post_init__(self) -> None:
        """Validate that every budget bound is a positive int."""
        for name in ("max_tokens", "max_bytes", "max_snippet_chars"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise KnowledgeValidationError(f"{name} must be an int >= 1, got {value!r}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the budget."""
        return {"max_tokens": self.max_tokens, "max_bytes": self.max_bytes, "max_snippet_chars": self.max_snippet_chars}


@dataclass(frozen=True)
class PacketItem:
    """One packet row: pointer-rich summary of a retrieved candidate.

    ``evidence`` carries L2 pointers (experiment / artifact / finding ids)
    the agent opens on demand; ``flags`` carries display markers such as
    ``validated``, ``disputed``, ``superseded``, ``failure``,
    ``replicated``, ``candidate``, ``summary``.
    """

    id: str = ""
    kind: str = ""
    title: str = ""
    snippet: str = ""
    score: float = 0.0
    flags: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the item fields."""
        if not isinstance(self.id, str) or not self.id.strip():
            raise KnowledgeValidationError(f"id must be a non-empty string, got {self.id!r}.")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise KnowledgeValidationError(f"kind must be a non-empty string, got {self.kind!r}.")
        if not isinstance(self.title, str):
            raise KnowledgeValidationError(f"title must be a string, got {type(self.title).__name__}.")
        if not isinstance(self.snippet, str):
            raise KnowledgeValidationError(f"snippet must be a string, got {type(self.snippet).__name__}.")
        if not isinstance(self.score, (int, float)) or isinstance(self.score, bool):
            raise KnowledgeValidationError(f"score must be a number, got {self.score!r}.")
        for name in ("flags", "evidence"):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)) or not isinstance(values, tuple):
                raise KnowledgeValidationError(f"{name} must be a tuple of strings, got {values!r}.")
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    raise KnowledgeValidationError(f"{name} entries must be non-empty strings, got {value!r}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the packet item."""
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "snippet": self.snippet,
            "score": self.score,
            "flags": list(self.flags),
            "evidence": list(self.evidence),
        }

    def to_text(self) -> str:
        """Render the item as one packet line (``- <id> [flags]: <title> — <snippet>``)."""
        markers = f" [{', '.join(self.flags)}]" if self.flags else ""
        pointers = f" (evidence: {', '.join(self.evidence)})" if self.evidence else ""
        body = self.title.strip()
        if self.snippet.strip():
            body = f"{body} — {self.snippet.strip()}" if body else self.snippet.strip()
        return f"- {self.id}{markers}: {body}{pointers}"


@dataclass(frozen=True)
class PacketSection:
    """One named packet section with its fitted items and cost accounting."""

    name: str = ""
    items: tuple[PacketItem, ...] = ()
    truncated: bool = False
    tokens_used: int = 0
    bytes_used: int = 0

    def __post_init__(self) -> None:
        """Validate the section fields."""
        if self.name not in SECTION_ORDER:
            raise KnowledgeValidationError(f"name must be one of {list(SECTION_ORDER)}, got {self.name!r}.")
        if not isinstance(self.items, tuple):
            raise KnowledgeValidationError(f"items must be a tuple of PacketItem, got {type(self.items).__name__}.")
        for item in self.items:
            if not isinstance(item, PacketItem):
                raise KnowledgeValidationError(f"items must be PacketItem records, got {type(item).__name__}.")
        if not isinstance(self.truncated, bool):
            raise KnowledgeValidationError(f"truncated must be a bool, got {type(self.truncated).__name__}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the packet section."""
        return {
            "name": self.name,
            "title": SECTION_TITLES[self.name],
            "items": [item.to_dict() for item in self.items],
            "truncated": self.truncated,
            "tokens_used": self.tokens_used,
            "bytes_used": self.bytes_used,
        }


@dataclass(frozen=True)
class SummaryPlaceholder:
    """Explicit L0/L1 placeholder marking summaries owned by Phase 4 consolidation."""

    level: str = L0_LEVEL
    topic_key: str = ""
    status: str = PLACEHOLDER_PENDING_PHASE4
    note: str = "Phase 4 consolidation has not run yet; open L2 evidence directly."

    def __post_init__(self) -> None:
        """Validate the placeholder fields."""
        if self.level not in (L0_LEVEL, L1_LEVEL):
            raise KnowledgeValidationError(f"level must be {L0_LEVEL!r} or {L1_LEVEL!r}, got {self.level!r}.")
        if not isinstance(self.topic_key, str):
            raise KnowledgeValidationError(f"topic_key must be a string, got {type(self.topic_key).__name__}.")
        if self.status != PLACEHOLDER_PENDING_PHASE4:
            raise KnowledgeValidationError(f"status must be {PLACEHOLDER_PENDING_PHASE4!r} until Phase 4 lands, got {self.status!r}.")
        if not isinstance(self.note, str) or not self.note.strip():
            raise KnowledgeValidationError("note must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the placeholder."""
        return {"level": self.level, "topic_key": self.topic_key, "status": self.status, "note": self.note}

    def to_text(self) -> str:
        """Render the placeholder as one packet line."""
        topic = f" [{self.topic_key}]" if self.topic_key else ""
        return f"- {self.level}{topic} ({self.status}): {self.note}"


@dataclass(frozen=True)
class ConflictView:
    """One conflict set rendered into the packet (Phase 3 sources these from ``conflict_set``)."""

    id: str = ""
    status: str = "open"
    member_ids: tuple[str, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        """Validate the conflict view fields."""
        if not isinstance(self.id, str) or not self.id.strip():
            raise KnowledgeValidationError(f"id must be a non-empty string, got {self.id!r}.")
        if not isinstance(self.status, str) or not self.status.strip():
            raise KnowledgeValidationError(f"status must be a non-empty string, got {self.status!r}.")
        if isinstance(self.member_ids, (str, bytes)) or not isinstance(self.member_ids, tuple):
            raise KnowledgeValidationError(f"member_ids must be a tuple of strings, got {self.member_ids!r}.")
        for member_id in self.member_ids:
            if not isinstance(member_id, str) or not member_id.strip():
                raise KnowledgeValidationError(f"member_ids entries must be non-empty strings, got {member_id!r}.")
        if not isinstance(self.summary, str):
            raise KnowledgeValidationError(f"summary must be a string, got {type(self.summary).__name__}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the conflict view."""
        return {"id": self.id, "status": self.status, "member_ids": list(self.member_ids), "summary": self.summary}

    def to_text(self) -> str:
        """Render the conflict set as one packet line."""
        members = f" (members: {', '.join(self.member_ids)})" if self.member_ids else ""
        body = f": {self.summary.strip()}" if self.summary.strip() else ""
        return f"- {self.id} [conflict-set, {self.status}]{body}{members}"


@dataclass(frozen=True)
class ContextPacket:
    """A fixed-budget research context packet (agent-facing bootstrap output)."""

    sections: tuple[PacketSection, ...] = ()
    l0: SummaryPlaceholder = field(default_factory=SummaryPlaceholder)
    l1: tuple[SummaryPlaceholder, ...] = ()
    budget: PacketBudget = field(default_factory=PacketBudget)
    tokens_used: int = 0
    bytes_used: int = 0
    within_budget: bool = True
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the context packet."""
        return {
            "sections": [section.to_dict() for section in self.sections],
            "l0": self.l0.to_dict(),
            "l1": [placeholder.to_dict() for placeholder in self.l1],
            "budget": self.budget.to_dict(),
            "tokens_used": self.tokens_used,
            "bytes_used": self.bytes_used,
            "within_budget": self.within_budget,
            "warnings": list(self.warnings),
        }

    def section(self, name: str) -> PacketSection:
        """Return the section with this name (all seven are always present)."""
        for candidate in self.sections:
            if candidate.name == name:
                return candidate
        raise KnowledgeValidationError(f"unknown packet section {name!r}.")

    def to_text(self) -> str:
        """Render the packet in the KB § packet format (headers + items + L0/L1 + budget line)."""
        lines = ["RESEARCH CONTEXT", ""]
        for part in self.sections:
            lines.append(SECTION_TITLES[part.name])
            if part.items:
                lines.extend(item.to_text() for item in part.items)
            else:
                lines.append("- (none retrieved)")
            if part.truncated:
                lines.append("- ... truncated to budget (narrow the intent or raise the budget)")
            lines.append("")
        lines.append("Derived summaries (L0/L1)")
        lines.append(f"  {self.l0.to_text()}")
        for placeholder in self.l1:
            lines.append(f"  {placeholder.to_text()}")
        lines.append("")
        lines.append(f"[budget: {self.tokens_used}/{self.budget.max_tokens} tokens, {self.bytes_used}/{self.budget.max_bytes} bytes]")
        return "\n".join(lines)


def _clip_snippet(text: str, max_chars: int) -> str:
    """Clip snippet text to the per-item cap with an explicit ellipsis marker."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def _flags_for(item: FusedCandidate) -> tuple[str, ...]:
    """Derive display flags from a fused candidate (deterministic order)."""
    candidate = item.candidate
    flags: list[str] = []
    if candidate.status == "validated":
        flags.append("validated")
    elif candidate.status == "reviewed":
        flags.append("reviewed")
    elif candidate.status == "candidate":
        flags.append("candidate")
    elif candidate.status == "superseded":
        flags.append("superseded")
    if item.disputed:
        flags.append("disputed")
    if item.failure:
        flags.append("failure")
    metadata = candidate.metadata if isinstance(candidate.metadata, Mapping) else {}
    replications = metadata.get("replication_count")
    if (isinstance(replications, int) and not isinstance(replications, bool) and replications >= 1) or metadata.get("replicates") is True or metadata.get("replicated_experiment_id"):
        flags.append("replicated")
    if candidate.kind == "summary":
        flags.append("summary")
    return tuple(flags)


def _route(item: FusedCandidate) -> str:
    """Route a fused candidate to exactly one packet section."""
    candidate = item.candidate
    if item.failure:
        return SECTION_FAILURES
    if item.disputed or candidate.kind == "conflict":
        return SECTION_CONFLICTS
    if candidate.kind == "assumption":
        return SECTION_ASSUMPTIONS
    if candidate.kind in ("skill", "prior"):
        return SECTION_SKILLS
    if candidate.kind == "experiment":
        return SECTION_EXPERIMENTS
    return SECTION_CONSENSUS


def build_context_packet(
    fused: FusionResult | Sequence[FusedCandidate],
    *,
    conflicts: Sequence[ConflictView] = (),
    open_questions: Sequence[str] = (),
    budget: PacketBudget | None = None,
    topic_key: str = "",
    l1_topics: Sequence[str] = (),
    section_shares: Mapping[str, float] | None = None,
) -> ContextPacket:
    """Build a fixed-budget :class:`ContextPacket` from fused retrieval output.

    Routing: failures → failures; disputed findings / conflict rows →
    conflicts; assumptions → assumptions; skills/priors → skills;
    experiments → prior-experiments; findings/summaries → consensus.
    Explicit ``conflicts`` (Phase 3 ``conflict_set`` rows) render first in
    the conflicts section; ``open_questions`` fill the open-questions
    section verbatim. Sections fill greedily in :data:`SECTION_ORDER` with
    per-section share caps; anything beyond a cap (or the global budget)
    sets ``truncated`` and a warning instead of dropping silently.

    Args:
        fused: A :class:`FusionResult` or an already-ranked fused sequence.
        conflicts: Explicit conflict sets to surface (Phase 3 seam).
        open_questions: Open-question strings for the final section.
        budget: Packet budget (defaults to :class:`PacketBudget`).
        topic_key: Topic label for the L0 placeholder ("" when unknown).
        l1_topics: Topic labels pre-registering L1 placeholders.
        section_shares: Optional per-section budget fractions (must cover
            every section and sum to ~1.0; defaults to
            :data:`DEFAULT_SECTION_SHARES`).

    Returns:
        A :class:`ContextPacket` whose rendered text fits the budget.

    Raises:
        KnowledgeValidationError: On invalid conflicts, questions, shares, or topics.
    """
    if isinstance(fused, FusionResult):
        ranked = list(fused.candidates)
    elif isinstance(fused, (str, bytes)) or not isinstance(fused, Sequence):
        raise KnowledgeValidationError(f"fused must be a FusionResult or a sequence of FusedCandidate, got {type(fused).__name__}.")
    else:
        ranked = list(fused)
    for item in ranked:
        if not isinstance(item, FusedCandidate):
            raise KnowledgeValidationError(f"fused entries must be FusedCandidate records, got {type(item).__name__}.")
    if isinstance(conflicts, (str, bytes)) or not isinstance(conflicts, Sequence):
        raise KnowledgeValidationError(f"conflicts must be a sequence of ConflictView, got {type(conflicts).__name__}.")
    for view in conflicts:
        if not isinstance(view, ConflictView):
            raise KnowledgeValidationError(f"conflicts entries must be ConflictView records, got {type(view).__name__}.")
    if isinstance(open_questions, (str, bytes)) or not isinstance(open_questions, Sequence):
        raise KnowledgeValidationError(f"open_questions must be a sequence of strings, got {type(open_questions).__name__}.")
    questions: list[str] = []
    for index, question in enumerate(open_questions):
        if not isinstance(question, str) or not question.strip():
            raise KnowledgeValidationError(f"open_questions[{index}] must be a non-empty string, got {question!r}.")
        questions.append(" ".join(question.split()))
    effective_budget = budget or PacketBudget()
    if not isinstance(effective_budget, PacketBudget):
        raise KnowledgeValidationError(f"budget must be a PacketBudget, got {type(effective_budget).__name__}.")
    if not isinstance(topic_key, str):
        raise KnowledgeValidationError(f"topic_key must be a string, got {type(topic_key).__name__}.")
    if isinstance(l1_topics, (str, bytes)) or not isinstance(l1_topics, Sequence):
        raise KnowledgeValidationError(f"l1_topics must be a sequence of strings, got {type(l1_topics).__name__}.")
    for index, label in enumerate(l1_topics):
        if not isinstance(label, str) or not label.strip():
            raise KnowledgeValidationError(f"l1_topics[{index}] must be a non-empty string, got {label!r}.")
    shares = dict(DEFAULT_SECTION_SHARES) if section_shares is None else dict(section_shares)
    if set(shares) != set(SECTION_ORDER):
        raise KnowledgeValidationError(f"section_shares must cover exactly {list(SECTION_ORDER)}, got {sorted(shares)}.")
    for name, share in shares.items():
        if not isinstance(share, (int, float)) or isinstance(share, bool) or share < 0:
            raise KnowledgeValidationError(f"section_shares[{name!r}] must be a non-negative number, got {share!r}.")
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise KnowledgeValidationError(f"section_shares must sum to 1.0, got {sum(shares.values())!r}.")

    routed: dict[str, list[PacketItem]] = {name: [] for name in SECTION_ORDER}
    for view in conflicts:
        routed[SECTION_CONFLICTS].append(
            PacketItem(
                id=view.id,
                kind="conflict",
                title=f"Conflict {view.id} [{view.status}]",
                snippet=_clip_snippet(view.summary, effective_budget.max_snippet_chars),
                evidence=tuple(view.member_ids),
            )
        )
    for item in ranked:
        candidate = item.candidate
        snippet_source = candidate.text.strip() or candidate.title.strip()
        routed[_route(item)].append(
            PacketItem(
                id=candidate.id,
                kind=candidate.kind,
                title=candidate.title.strip() or candidate.id,
                snippet=_clip_snippet(snippet_source, effective_budget.max_snippet_chars),
                score=item.score,
                flags=_flags_for(item),
                evidence=tuple(candidate.evidence),
            )
        )
    for index, question in enumerate(questions):
        routed[SECTION_OPEN_QUESTIONS].append(
            PacketItem(
                id=f"q{index + 1}",
                kind="open_question",
                title=question,
                snippet="",
            )
        )

    warnings: list[str] = []
    fitted: list[PacketSection] = []
    total_tokens = 0
    total_bytes = 0
    budget_exhausted = False
    for name in SECTION_ORDER:
        section_token_cap = int(shares[name] * effective_budget.max_tokens)
        section_byte_cap = int(shares[name] * effective_budget.max_bytes)
        header = SECTION_TITLES[name]
        header_tokens = estimate_tokens(header)
        header_bytes = utf8_bytes(header)
        section_tokens = header_tokens
        section_bytes = header_bytes
        kept: list[PacketItem] = []
        truncated = False
        for candidate_item in routed[name]:
            line = candidate_item.to_text()
            cost_tokens = estimate_tokens(line)
            cost_bytes = utf8_bytes(line)
            if (
                section_tokens + cost_tokens > section_token_cap
                or section_bytes + cost_bytes > section_byte_cap
                or total_tokens + section_tokens + cost_tokens > effective_budget.max_tokens
                or total_bytes + section_bytes + cost_bytes > effective_budget.max_bytes
            ):
                truncated = True
                budget_exhausted = budget_exhausted or (total_tokens + section_tokens + cost_tokens > effective_budget.max_tokens or total_bytes + section_bytes + cost_bytes > effective_budget.max_bytes)
                continue
            kept.append(candidate_item)
            section_tokens += cost_tokens
            section_bytes += cost_bytes
        if truncated:
            warnings.append(f"section {name!r} truncated to budget ({len(kept)}/{len(routed[name])} items kept)")
        fitted.append(PacketSection(name=name, items=tuple(kept), truncated=truncated, tokens_used=section_tokens, bytes_used=section_bytes))
        total_tokens += section_tokens
        total_bytes += section_bytes

    l0 = SummaryPlaceholder(level=L0_LEVEL, topic_key=topic_key.strip())
    l1 = tuple(SummaryPlaceholder(level=L1_LEVEL, topic_key=label.strip()) for label in l1_topics)
    if budget_exhausted:
        warnings.append("global budget exhausted while fitting sections")
    measured_tokens = 0
    measured_bytes = 0
    for _ in range(3):
        provisional = ContextPacket(
            sections=tuple(fitted),
            l0=l0,
            l1=l1,
            budget=effective_budget,
            tokens_used=measured_tokens,
            bytes_used=measured_bytes,
            warnings=tuple(warnings),
        )
        rendered = provisional.to_text()
        # The budget footer echoes the measured cost, so re-measure until the
        # digit widths stabilize (converges after at most two passes).
        stabilized_tokens = estimate_tokens(rendered)
        stabilized_bytes = utf8_bytes(rendered)
        if stabilized_tokens == measured_tokens and stabilized_bytes == measured_bytes:
            break
        measured_tokens, measured_bytes = stabilized_tokens, stabilized_bytes
    within = measured_tokens <= effective_budget.max_tokens and measured_bytes <= effective_budget.max_bytes
    final_warnings = list(warnings)
    if not within:
        final_warnings.append("packet exceeds budget even before item text; raise the budget")
    return ContextPacket(
        sections=tuple(fitted),
        l0=l0,
        l1=l1,
        budget=effective_budget,
        tokens_used=measured_tokens,
        bytes_used=measured_bytes,
        within_budget=within,
        warnings=tuple(final_warnings),
    )
