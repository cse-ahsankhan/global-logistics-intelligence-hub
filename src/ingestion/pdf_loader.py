"""PDF document loader with table-aware parsing.

Uses pdfplumber for structured extraction and falls back to
unstructured for complex layouts. Tables are converted to
Markdown for downstream chunking.
"""

from pathlib import Path
from typing import Optional

import pdfplumber
from langchain_core.documents import Document


class PDFLoader:
    """Load and parse PDF documents with table awareness."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.file_path}")

    def load(self) -> list[Document]:
        """Load PDF and return list of Documents (one per page)."""
        documents = []

        with pdfplumber.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text_content = self._extract_page_content(page)
                if text_content.strip():
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "source": str(self.file_path),
                            "page": page_num,
                            "total_pages": len(pdf.pages),
                            "content_type": "pdf",
                        },
                    )
                    documents.append(doc)

        return documents

    def _extract_page_content(self, page: pdfplumber.page.Page) -> str:
        """Extract text and tables from a single page."""
        parts: list[str] = []

        # Extract raw text
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text.strip())

        # Extract tables and convert to Markdown
        tables = page.extract_tables()
        for table in tables:
            markdown_table = self._table_to_markdown(table)
            if markdown_table:
                parts.append(f"\n{markdown_table}\n")

        return "\n\n".join(parts)

    @staticmethod
    def _table_to_markdown(table: list[list[Optional[str]]]) -> str:
        """Convert a pdfplumber table to Markdown format."""
        if not table or len(table) < 2:
            return ""

        clean_table = []
        for row in table:
            clean_row = [(cell or "").strip().replace("\n", " ") for cell in row]
            clean_table.append(clean_row)

        headers = clean_table[0]
        col_count = len(headers)

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * col_count) + " |",
        ]

        for row in clean_table[1:]:
            # Pad or truncate row to match header count
            padded = (row + [""] * col_count)[:col_count]
            lines.append("| " + " | ".join(padded) + " |")

        return "\n".join(lines)
