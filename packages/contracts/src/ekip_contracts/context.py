"""What the answer-generating model is allowed to read, and how answers point back to it.

A `ContextBundle` is built ONLY from chunks the retriever returned for this caller, so everything
in it is already access-checked. Each block has a citation number the model may use as `[n]`;
`citation(n)` maps a number back to a real chunk, document and version. Numbers that do not map
are hallucinated citations and are rejected downstream.
"""

import html
from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    number: int
    chunk_id: str
    document_id: str
    document_version: int | None
    title: str
    source_uri: str
    section_path: tuple[str, ...] = ()
    page_number: int | None = None


@dataclass(frozen=True)
class ContextBlock:
    citation: Citation
    text: str
    tokens: int
    truncated: bool = False


@dataclass(frozen=True)
class ContextBundle:
    blocks: tuple[ContextBlock, ...] = ()
    dropped_duplicates: int = 0  # same text already included (e.g. one paragraph in two files)
    dropped_diversity: int = 0  # over the per-document cap
    dropped_budget: int = 0  # did not fit the token budget
    total_tokens: int = 0

    @property
    def citations(self) -> tuple[Citation, ...]:
        return tuple(b.citation for b in self.blocks)

    def citation(self, number: int) -> Citation | None:
        return next((b.citation for b in self.blocks if b.citation.number == number), None)

    def render(self) -> str:
        """Prompt-ready text. Every passage sits inside its own <source> element with the markup
        characters of its title and text escaped, so retrieved text can never close its element
        or open a fake one (it is DATA, not instructions - invariant I-3)."""
        parts: list[str] = []
        for block in self.blocks:
            c = block.citation
            where = " > ".join(c.section_path)
            attrs = f'id="{c.number}" title="{html.escape(c.title, quote=True)}"'
            if where:
                attrs += f' section="{html.escape(where, quote=True)}"'
            if c.page_number is not None:
                attrs += f' page="{c.page_number}"'
            parts.append(f"<source {attrs}>\n{html.escape(block.text, quote=False)}\n</source>")
        return "\n".join(parts)
