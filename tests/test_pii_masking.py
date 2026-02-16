"""Tests for the PII masking module."""

import pytest

from src.processing.pii_masking import PIIMasker, MaskingResult


@pytest.fixture
def masker():
    return PIIMasker()


@pytest.fixture
def masker_no_logistics():
    return PIIMasker(mask_logistics_ids=False)


class TestPIIMasker:
    def test_mask_email(self, masker):
        text = "Contact john.doe@example.com for shipment details."
        result = masker.mask(text)
        assert isinstance(result, MaskingResult)
        assert "john.doe@example.com" not in result.masked_text
        assert len(result.entity_mapping) > 0

    def test_mask_phone_number(self, masker):
        text = "Call the warehouse at +1-555-123-4567 for pickup."
        result = masker.mask(text)
        assert "+1-555-123-4567" not in result.masked_text

    def test_mask_person_name(self, masker):
        text = "Shipment approved by John Smith on behalf of the logistics team."
        result = masker.mask(text)
        assert "John Smith" not in result.masked_text

    def test_mask_container_id(self, masker):
        text = "Container MSCU1234567 loaded at berth 7."
        result = masker.mask(text)
        assert "MSCU1234567" not in result.masked_text

    def test_mask_bill_of_lading(self, masker):
        text = "Reference BOL-123456789 for the ocean shipment."
        result = masker.mask(text)
        assert "BOL-123456789" not in result.masked_text

    def test_mask_customs_reference(self, masker):
        text = "Customs declaration MRN12345678901234 filed."
        result = masker.mask(text)
        assert "MRN12345678901234" not in result.masked_text

    def test_no_pii_returns_original(self, masker):
        text = "Ocean freight rates increased by 15% in Q3."
        result = masker.mask(text)
        assert result.masked_text == text
        assert len(result.entity_mapping) == 0

    def test_unmask_roundtrip(self, masker):
        text = "Contact john.doe@example.com about container MSCU1234567."
        result = masker.mask(text)
        restored = masker.unmask(result.masked_text, result.entity_mapping)
        assert "john.doe@example.com" in restored
        assert "MSCU1234567" in restored

    def test_entities_found_list(self, masker):
        text = "Email: test@example.com, Container: ABCD1234567"
        result = masker.mask(text)
        assert len(result.entities_found) > 0
        entity_types = {e["entity_type"] for e in result.entities_found}
        assert "EMAIL_ADDRESS" in entity_types or "CONTAINER_ID" in entity_types

    def test_placeholder_format(self, masker):
        text = "Contact john.doe@example.com for details."
        result = masker.mask(text)
        # Placeholders should follow the [TYPE_N] pattern
        for placeholder in result.entity_mapping:
            assert placeholder.startswith("[")
            assert placeholder.endswith("]")

    def test_logistics_ids_not_masked_when_disabled(self, masker_no_logistics):
        text = "Container MSCU1234567 loaded at berth 7."
        result = masker_no_logistics.mask(text)
        # Without logistics masking, container ID should remain
        assert "MSCU1234567" in result.masked_text
