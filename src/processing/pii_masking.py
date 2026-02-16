"""PII masking for supply chain documents using Microsoft Presidio.

Detects and redacts personally identifiable information while
preserving a reversible mapping for audit and compliance purposes.
Includes custom recognizers for logistics-specific identifiers
(container numbers, customs references, bill of lading numbers).
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


@dataclass
class MaskingResult:
    """Result of a PII masking operation."""

    masked_text: str
    entity_mapping: dict[str, str] = field(default_factory=dict)
    entities_found: list[dict] = field(default_factory=list)


# --- Custom recognizers for logistics-specific identifiers ---

_CONTAINER_ID_PATTERN = Pattern(
    name="container_id",
    regex=r"\b[A-Z]{4}\d{7}\b",
    score=0.85,
)
CONTAINER_ID_RECOGNIZER = PatternRecognizer(
    supported_entity="CONTAINER_ID",
    name="Container ID Recognizer",
    patterns=[_CONTAINER_ID_PATTERN],
    supported_language="en",
)

_BOL_PATTERN = Pattern(
    name="bill_of_lading",
    regex=r"\b(?:BOL|B/L|BL)[-#]?\s*\d{6,12}\b",
    score=0.80,
)
BOL_RECOGNIZER = PatternRecognizer(
    supported_entity="BILL_OF_LADING",
    name="Bill of Lading Recognizer",
    patterns=[_BOL_PATTERN],
    supported_language="en",
)

_CUSTOMS_REF_PATTERN = Pattern(
    name="customs_reference",
    regex=r"\b(?:CUS|CUSTOMS|MRN)[-/]?\d{8,18}\b",
    score=0.80,
)
CUSTOMS_REF_RECOGNIZER = PatternRecognizer(
    supported_entity="CUSTOMS_REFERENCE",
    name="Customs Reference Recognizer",
    patterns=[_CUSTOMS_REF_PATTERN],
    supported_language="en",
)

_HS_CODE_PATTERN = Pattern(
    name="hs_code",
    regex=r"\bHS\s*\d{4}(?:\.\d{2}(?:\.\d{2,4})?)?\b",
    score=0.75,
)
HS_CODE_RECOGNIZER = PatternRecognizer(
    supported_entity="HS_CODE",
    name="HS Code Recognizer",
    patterns=[_HS_CODE_PATTERN],
    supported_language="en",
)


class PIIMasker:
    """Detect and mask PII in supply chain documents.

    Uses Presidio for standard PII entities (names, emails, phone
    numbers, etc.) and adds custom recognizers for logistics-specific
    identifiers.

    Parameters
    ----------
    entities_to_mask : list[str] | None
        Specific entity types to mask. If None, all detected entities
        are masked.
    mask_logistics_ids : bool
        Whether to also mask logistics-specific identifiers
        (container IDs, customs references, etc.).
    """

    DEFAULT_ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IBAN_CODE",
        "IP_ADDRESS",
        "LOCATION",
    ]

    LOGISTICS_ENTITIES = [
        "CONTAINER_ID",
        "BILL_OF_LADING",
        "CUSTOMS_REFERENCE",
        "HS_CODE",
    ]

    def __init__(
        self,
        entities_to_mask: Optional[list[str]] = None,
        mask_logistics_ids: bool = True,
    ):
        self.analyzer = AnalyzerEngine()

        # Register custom recognizers
        registry = self.analyzer.registry
        registry.add_recognizer(CONTAINER_ID_RECOGNIZER)
        registry.add_recognizer(BOL_RECOGNIZER)
        registry.add_recognizer(CUSTOMS_REF_RECOGNIZER)
        registry.add_recognizer(HS_CODE_RECOGNIZER)

        self.anonymizer = AnonymizerEngine()

        self.entities_to_mask = entities_to_mask or self.DEFAULT_ENTITIES
        if mask_logistics_ids:
            self.entities_to_mask.extend(self.LOGISTICS_ENTITIES)

        self._counter: dict[str, int] = {}

    def mask(self, text: str, language: str = "en") -> MaskingResult:
        """Mask PII in the given text.

        Returns a ``MaskingResult`` with the masked text, a mapping
        from placeholder to original value, and a list of detected
        entities.
        """
        # Reset counter for fresh mapping
        self._counter = {}

        # Analyze
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities_to_mask,
            language=language,
        )

        if not results:
            return MaskingResult(masked_text=text)

        # Build entity mapping and prepare operators
        entity_mapping: dict[str, str] = {}
        operators: dict[str, OperatorConfig] = {}

        # Sort by start position (descending) to replace from end to start
        sorted_results = sorted(results, key=lambda r: r.start, reverse=True)

        masked_text = text
        for result in sorted_results:
            original = text[result.start:result.end]
            placeholder = self._get_placeholder(result.entity_type)
            entity_mapping[placeholder] = original
            masked_text = masked_text[:result.start] + placeholder + masked_text[result.end:]

        entities_found = [
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(r.score, 3),
            }
            for r in results
        ]

        return MaskingResult(
            masked_text=masked_text,
            entity_mapping=entity_mapping,
            entities_found=entities_found,
        )

    def unmask(self, masked_text: str, entity_mapping: dict[str, str]) -> str:
        """Restore original values from a masked text using the mapping."""
        text = masked_text
        for placeholder, original in entity_mapping.items():
            text = text.replace(placeholder, original)
        return text

    def _get_placeholder(self, entity_type: str) -> str:
        """Generate a unique placeholder like [PERSON_1], [PERSON_2], etc."""
        count = self._counter.get(entity_type, 0) + 1
        self._counter[entity_type] = count
        return f"[{entity_type}_{count}]"
