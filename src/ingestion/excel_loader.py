"""Excel/CSV document loader for structured supply chain data.

Converts spreadsheet data into Markdown-formatted documents
suitable for embedding and retrieval.
"""

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document


class ExcelLoader:
    """Load Excel/CSV files and convert to Document objects."""

    SUPPORTED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".tsv"}

    def __init__(self, file_path: str | Path, sheet_name: str | int | None = None):
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if self.file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported format: {self.file_path.suffix}")

    def load(self) -> list[Document]:
        """Load spreadsheet and return Documents (one per sheet/file)."""
        suffix = self.file_path.suffix.lower()

        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(self.file_path, sep=sep)
            return [self._dataframe_to_document(df, sheet_name="Sheet1")]

        sheets = pd.read_excel(
            self.file_path,
            sheet_name=self.sheet_name,
            engine="openpyxl",
        )

        if isinstance(sheets, pd.DataFrame):
            return [self._dataframe_to_document(sheets, sheet_name=str(self.sheet_name or "Sheet1"))]

        documents = []
        for name, df in sheets.items():
            documents.append(self._dataframe_to_document(df, sheet_name=str(name)))
        return documents

    def _dataframe_to_document(self, df: pd.DataFrame, sheet_name: str) -> Document:
        """Convert a DataFrame to a Markdown-formatted Document."""
        # Drop fully empty rows/columns
        df = df.dropna(how="all").dropna(axis=1, how="all")

        markdown = self._dataframe_to_markdown(df)
        summary = self._generate_summary(df, sheet_name)

        return Document(
            page_content=f"{summary}\n\n{markdown}",
            metadata={
                "source": str(self.file_path),
                "sheet_name": sheet_name,
                "row_count": len(df),
                "columns": list(df.columns),
                "content_type": "spreadsheet",
            },
        )

    @staticmethod
    def _dataframe_to_markdown(df: pd.DataFrame) -> str:
        """Convert DataFrame to Markdown table string."""
        return df.to_markdown(index=False) if not df.empty else ""

    @staticmethod
    def _generate_summary(df: pd.DataFrame, sheet_name: str) -> str:
        """Generate a natural-language summary of the spreadsheet."""
        lines = [
            f"**Sheet: {sheet_name}**",
            f"- Rows: {len(df)}",
            f"- Columns: {', '.join(str(c) for c in df.columns)}",
        ]

        # Add basic stats for numeric columns
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols[:5]:
            lines.append(f"- {col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")

        return "\n".join(lines)
