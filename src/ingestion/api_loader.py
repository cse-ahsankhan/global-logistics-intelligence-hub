"""REST API data loader for external supply chain systems.

Supports fetching data from logistics APIs (TMS, WMS, carrier APIs)
and converting JSON responses into Document objects.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from langchain_core.documents import Document


class APILoader:
    """Load data from REST APIs and convert to Documents."""

    def __init__(
        self,
        base_url: str,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout = timeout

    def load(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        source_label: Optional[str] = None,
    ) -> list[Document]:
        """Fetch data from an API endpoint and return Documents."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, headers=self.headers, params=params)
            response.raise_for_status()

        data = response.json()
        source = source_label or endpoint

        if isinstance(data, list):
            return [self._record_to_document(record, source, idx) for idx, record in enumerate(data)]

        return [self._record_to_document(data, source, 0)]

    def _record_to_document(
        self, record: dict[str, Any], source: str, index: int
    ) -> Document:
        """Convert a single API record to a Document."""
        content = self._flatten_to_text(record)
        record_id = self._generate_record_id(record, index)

        return Document(
            page_content=content,
            metadata={
                "source": f"api://{self.base_url}/{source}",
                "record_id": record_id,
                "record_index": index,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "content_type": "api_response",
            },
        )

    @staticmethod
    def _flatten_to_text(record: dict[str, Any], prefix: str = "") -> str:
        """Flatten a nested dict into human-readable key-value text."""
        lines = []
        for key, value in record.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                lines.append(APILoader._flatten_to_text(value, full_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        lines.append(APILoader._flatten_to_text(item, f"{full_key}[{i}]"))
                    else:
                        lines.append(f"{full_key}[{i}]: {item}")
            else:
                lines.append(f"{full_key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _generate_record_id(record: dict[str, Any], index: int) -> str:
        """Generate a deterministic ID for a record."""
        # Use 'id' field if present, otherwise hash the content
        if "id" in record:
            return str(record["id"])
        content = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(f"{content}:{index}".encode()).hexdigest()[:16]
