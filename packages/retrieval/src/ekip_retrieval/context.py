"""Turn ranked hits into a bounded, de-duplicated, citable context for the model.

Why not just paste the top-k? Real corpora repeat themselves (the same paragraph in a PDF and its
HTML export, boilerplate in every revision), one long document can crowd out all others, and the
model's window is finite and billed per token. So we: drop repeats, cap passages per document, fit
a token budget, and number what is left so the answer can cite it.
"""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ekip_contracts.context import Citation, ContextBlock, ContextBundle
from ekip_contracts.retrieval import RetrievedChunk

TokenCounter = Callable[[str], int]
_BLOCK_OVERHEAD_TOKENS = 20  # the <source ...> wrapper around each passage
_WS = re.compile(r"\s+")


def approx_tokens(text: str) -> int:
    """~4 characters per token. A deliberate over-estimate for English; inject a real tokenizer
    for the chosen chat model when exact budgets matter (Phase 6)."""
    return math.ceil(len(text) / 4)


@dataclass(frozen=True)
class ContextConfig:
    budget_tokens: int = 3000
    max_per_document: int = 3
    min_block_tokens: int = 40  # do not squeeze a passage into a useless sliver of budget

    def __post_init__(self) -> None:
        if self.budget_tokens < 1 or self.max_per_document < 1 or self.min_block_tokens < 1:
            raise ValueError("budget_tokens, max_per_document and min_block_tokens must be >= 1")


class ContextAssembler:
    def __init__(
        self,
        config: ContextConfig = ContextConfig(),  # noqa: B008 - frozen dataclass
        *,
        count_tokens: TokenCounter = approx_tokens,
    ) -> None:
        self._config = config
        self._count = count_tokens

    def assemble(self, hits: Sequence[RetrievedChunk]) -> ContextBundle:
        cfg = self._config
        seen_text: set[str] = set()
        per_document: Counter[str] = Counter()
        blocks: list[ContextBlock] = []
        used = 0
        duplicates = diversity = budget = 0

        for hit in hits:  # best first: earlier hits win every conflict
            key = hit.content_hash or _fingerprint(hit.text)
            fingerprint = _fingerprint(hit.text)
            if key in seen_text or fingerprint in seen_text:
                duplicates += 1
                continue
            if per_document[hit.document_id] >= cfg.max_per_document:
                diversity += 1
                continue
            text, tokens, truncated = hit.text, self._count(hit.text), False
            remaining = cfg.budget_tokens - used - _BLOCK_OVERHEAD_TOKENS
            if tokens > remaining:
                if remaining < cfg.min_block_tokens:
                    budget += 1
                    continue
                text, tokens = self._truncate(hit.text, remaining)
                truncated = True
            number = len(blocks) + 1
            blocks.append(
                ContextBlock(
                    citation=Citation(
                        number=number,
                        chunk_id=hit.chunk_id,
                        document_id=hit.document_id,
                        document_version=hit.document_version,
                        title=hit.title,
                        source_uri=hit.source_uri,
                        section_path=hit.section_path,
                        page_number=hit.page_number,
                    ),
                    text=text,
                    tokens=tokens,
                    truncated=truncated,
                )
            )
            used += tokens + _BLOCK_OVERHEAD_TOKENS
            seen_text.update({key, fingerprint})
            per_document[hit.document_id] += 1

        return ContextBundle(
            blocks=tuple(blocks),
            dropped_duplicates=duplicates,
            dropped_diversity=diversity,
            dropped_budget=budget,
            total_tokens=used,
        )

    def _truncate(self, text: str, max_tokens: int) -> tuple[str, int]:
        """Cut at the last whole word that fits (binary search: the counter may not be linear)."""
        words = text.split(" ")
        lo, hi = 1, len(words)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._count(" ".join(words[:mid]) + " …") <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        cut = " ".join(words[:lo]) + " …"
        return cut, self._count(cut)


def _fingerprint(text: str) -> str:
    """Whitespace- and case-insensitive identity of a passage."""
    return hashlib.sha256(_WS.sub(" ", text).strip().lower().encode()).hexdigest()


# ---- citations in the model's answer -------------------------------------------------------

_MARKER = re.compile(r"\[(\d{1,4}(?:\s*,\s*\d{1,4})*)\]")


@dataclass(frozen=True)
class CitationReport:
    cited: tuple[int, ...]  # numbers that map to a block in the bundle
    unknown: tuple[int, ...]  # numbers the model invented (no such block)

    @property
    def ok(self) -> bool:
        return not self.unknown

    @property
    def has_citations(self) -> bool:
        return bool(self.cited)


def check_citations(answer: str, bundle: ContextBundle) -> CitationReport:
    """Find `[1]`, `[2, 3]` markers in an answer and check each number against the bundle."""
    valid = {b.citation.number for b in bundle.blocks}
    found: list[int] = []
    for match in _MARKER.finditer(answer):
        found.extend(int(n) for n in match.group(1).replace(" ", "").split(","))
    return CitationReport(
        cited=tuple(sorted({n for n in found if n in valid})),
        unknown=tuple(sorted({n for n in found if n not in valid})),
    )
