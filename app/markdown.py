"""
app/markdown.py
Handles document-to-markdown conversion using docling.
"""

from docling.document_converter import DocumentConverter

# Initialised once and reused across all conversions
_converter = DocumentConverter()


def convert_document(file_path: str) -> tuple[str, object]:
    """
    Convert a document to markdown using docling.

    Returns:
        (markdown_str, docling_document_object)

    The document object is passed to the chunker so it can extract
    accurate page numbers and heading hierarchy without re-parsing.
    """
    result = _converter.convert(file_path)
    return result.document.export_to_markdown(), result.document


def parse_document(file_path: str) -> str:
    """Convert a document to markdown only (convenience wrapper)."""
    markdown, _ = convert_document(file_path)
    return markdown
