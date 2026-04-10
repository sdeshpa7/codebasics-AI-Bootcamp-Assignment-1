"""
app/chunking.py
Splits ingested documents into chunks with rich metadata, ready for embedding.

Two chunking strategies:
  - Docling-native  : uses HierarchicalChunker on the DoclingDocument object.
                      Provides accurate page numbers and heading hierarchy from
                      the document's internal structure.
  - Markdown-native : for files that are already .md (no DoclingDocument).
                      Splits on heading boundaries, estimates page numbers from
                      cumulative word count, and detects tables / code blocks.
"""

import re
import uuid
from dataclasses import dataclass, field

from docling_core.transforms.chunker import HierarchicalChunker

from app.config import settings

# Approximate words per printed A4 page — used to estimate page numbers
# for documents that have no provenance metadata (already-markdown files).
_WORDS_PER_PAGE = 350

# A single shared chunker instance
_chunker = HierarchicalChunker()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str                          # UUID — unique chunk identifier
    text: str                        # chunk content
    chunk_type: str                  # "text" | "table" | "code"
    page_number: int | None          # 1-indexed; None if unavailable
    section_title: str | None       # most-specific heading above this chunk
    parent_chunk_id: str | None      # chunk_id of the containing section chunk
    source_file: str                 # original filename
    collection: str                  # data subfolder (e.g. "finance")
    access_roles: list[str] = field(default_factory=list)
    # access_roles format: ["C-Level", "General", <collection-specific role>]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_access_roles(collection: str) -> list[str]:
    """
    Return the access_roles list for a given collection.
    Always starts with the common roles, then appends the collection-specific
    role if it is not already present.
    """
    roles: list[str] = list(settings.chunk_common_roles)   # e.g. ["C-Level"]
    specific = settings.folder_access_role.get(collection)
    if specific and specific not in roles:
        roles.append(specific)
    return roles


def _detect_chunk_type(text: str) -> str:
    """Heuristic classification of a text block."""
    stripped = text.strip()
    if stripped.startswith("```"):
        return "code"
    lines = [l for l in stripped.splitlines() if l.strip()]
    if lines:
        table_lines = sum(1 for l in lines if l.lstrip().startswith("|"))
        if table_lines / len(lines) > 0.5:
            return "table"
    return "text"


def _estimate_page_number(word_offset: int) -> int:
    """Return a 1-indexed estimated page number from a cumulative word offset."""
    return (word_offset // _WORDS_PER_PAGE) + 1


# ---------------------------------------------------------------------------
# Strategy 1: Docling-native chunking
# ---------------------------------------------------------------------------

def chunk_docling_document(doc_obj: object, source_file: str, collection: str) -> list[Chunk]:
    """
    Chunk a DoclingDocument using docling's HierarchicalChunker.

    Page numbers are read directly from each chunk's provenance metadata.
    Heading hierarchy is used to establish parent→child relationships.
    If a chunk has no page provenance, we fall back to position-based estimation.

    parent_chunk_id is resolved via a heading-level registry: when we see a
    chunk at heading depth N, its parent is the most recent chunk registered
    at depth N-1.  This avoids missing lookups caused by heading-text mismatches.
    """
    access_roles = _build_access_roles(collection)
    chunks: list[Chunk] = []

    # level (1-indexed depth) → chunk_id of the most recent chunk at that depth
    level_chunk_id: dict[int, str] = {}
    cumulative_words = 0

    for raw in _chunker.chunk(doc_obj):
        text = raw.text.strip()
        if not text:
            continue

        chunk_id = str(uuid.uuid4())

        # -- Page number from provenance --
        page_number: int | None = None
        doc_items = getattr(raw.meta, "doc_items", None) or []
        for item in doc_items:
            provs = getattr(item, "prov", None) or []
            if provs:
                page_number = provs[0].page_no
                break

        if page_number is None:
            page_number = _estimate_page_number(cumulative_words)
        cumulative_words += len(text.split())

        # -- Headings (outermost → innermost) --
        headings: list[str] = getattr(raw.meta, "headings", None) or []
        current_heading = headings[-1] if headings else None
        current_depth   = len(headings)          # 0 = no heading, 1 = H1, 2 = H2 …

        # Parent = most recent chunk at any shallower heading depth
        parent_chunk_id: str | None = None
        for d in range(current_depth - 1, 0, -1):
            if d in level_chunk_id:
                parent_chunk_id = level_chunk_id[d]
                break

        chunk_type = _detect_chunk_type(text)

        # Register this chunk at its heading depth for future children
        if current_depth > 0:
            level_chunk_id[current_depth] = chunk_id
            # Invalidate all deeper levels — they belong to a different branch now
            for d in list(level_chunk_id):
                if d > current_depth:
                    del level_chunk_id[d]

        chunks.append(Chunk(
            id=chunk_id,
            text=text,
            chunk_type=chunk_type,
            page_number=page_number,
            section_title=current_heading,
            parent_chunk_id=parent_chunk_id,
            source_file=source_file,
            collection=collection,
            access_roles=access_roles,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Markdown-native chunking
# ---------------------------------------------------------------------------

_HEADING_RE  = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_CODE_RE     = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
_TABLE_RE    = re.compile(r"((?:^\|.+(?:\n|$))+)", re.MULTILINE)


def _split_typed_blocks(text: str) -> list[tuple[str, str]]:
    """
    Split a markdown body (between headings) into (content, type) tuples.
    Recognises fenced code blocks (``` ```) and pipe tables first, then text.
    """
    blocks: list[tuple[str, str]] = []
    remaining = text

    while remaining:
        code_m  = _CODE_RE.search(remaining)
        table_m = _TABLE_RE.search(remaining)

        # Pick whichever special block comes first
        first, ftype = None, None
        if code_m and (table_m is None or code_m.start() <= table_m.start()):
            first, ftype = code_m, "code"
        elif table_m:
            first, ftype = table_m, "table"

        if first:
            before = remaining[:first.start()].strip()
            if before:
                blocks.append((before, "text"))
            blocks.append((first.group().strip(), ftype))
            remaining = remaining[first.end():]
        else:
            if remaining.strip():
                blocks.append((remaining.strip(), "text"))
            break

    return blocks


def chunk_markdown_text(markdown: str, source_file: str, collection: str) -> list[Chunk]:
    """
    Chunk a plain markdown string by heading boundaries.

    For every heading encountered, a lightweight section-header Chunk is
    created and added to the list first.  Body content chunks then reference
    this real Chunk via parent_chunk_id — eliminating phantom UUID references.

    Page numbers are estimated from cumulative word count.
    """
    access_roles = _build_access_roles(collection)
    chunks: list[Chunk] = []

    # Stack entries: (heading_level, heading_text, section_chunk_id)
    # section_chunk_id is the id of the REAL header Chunk already appended.
    heading_stack: list[tuple[int, str, str]] = []
    cumulative_words = 0

    # Split the document at heading boundaries, keeping the delimiters
    parts = _HEADING_RE.split(markdown)
    # _HEADING_RE has 2 groups → split gives: [before, hashes, text, body, hashes, text, body, …]

    # Reassemble into (heading_level, heading_text, body) triples
    segments: list[tuple[int | None, str | None, str]] = []
    if parts[0].strip():
        segments.append((None, None, parts[0]))

    i = 1
    while i + 2 <= len(parts):
        level_str    = parts[i]
        heading_text = parts[i + 1].strip()
        body         = parts[i + 2] if i + 2 < len(parts) else ""
        segments.append((len(level_str), heading_text, body))
        i += 3

    for level, heading_text, body in segments:
        # -- Maintain heading stack and create a real section-header chunk --
        if level is not None:
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            # Parent of this heading = top of the (now trimmed) stack
            heading_parent_id = heading_stack[-1][2] if heading_stack else None

            section_chunk_id = str(uuid.uuid4())
            page_number = _estimate_page_number(cumulative_words)
            cumulative_words += len(heading_text.split())

            # Emit a real section-header chunk so children can reference it
            chunks.append(Chunk(
                id=section_chunk_id,
                text=heading_text,
                chunk_type="text",
                page_number=page_number,
                section_title=heading_text,
                parent_chunk_id=heading_parent_id,
                source_file=source_file,
                collection=collection,
                access_roles=access_roles,
            ))

            heading_stack.append((level, heading_text, section_chunk_id))

        # Chunks in the body are children of the current heading chunk
        current_heading   = heading_stack[-1][1] if heading_stack else None
        current_parent_id = heading_stack[-1][2] if heading_stack else None

        for block_text, block_type in _split_typed_blocks(body):
            chunk_id    = str(uuid.uuid4())
            page_number = _estimate_page_number(cumulative_words)
            cumulative_words += len(block_text.split())

            chunks.append(Chunk(
                id=chunk_id,
                text=block_text,
                chunk_type=block_type,
                page_number=page_number,
                section_title=current_heading,
                parent_chunk_id=current_parent_id,   # always a real chunk id
                source_file=source_file,
                collection=collection,
                access_roles=access_roles,
            ))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_all(documents: dict) -> list[Chunk]:
    """
    Chunk every document returned by ingestion.ingest_all().

    Args:
        documents: {
            filename: {
                "markdown": str,
                "doc_obj":  DoclingDocument | None,
                "collection": str,
            }
        }

    Returns:
        Flat list of Chunk objects across all documents, with metadata.
    """
    all_chunks: list[Chunk] = []

    for filename, doc_data in documents.items():
        collection = doc_data.get("collection", "general")
        doc_obj    = doc_data.get("doc_obj")
        markdown   = doc_data.get("markdown", "")

        if doc_obj is not None:
            print(f"Chunking (docling) [{collection}] -> {filename}")
            file_chunks = chunk_docling_document(doc_obj, filename, collection)
        else:
            print(f"Chunking (markdown) [{collection}] -> {filename}")
            file_chunks = chunk_markdown_text(markdown, filename, collection)

        print(f"  → {len(file_chunks)} chunks  |  access_roles: {file_chunks[0].access_roles if file_chunks else '—'}")
        all_chunks.extend(file_chunks)

    return all_chunks
