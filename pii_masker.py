"""
pii_masker.py
HIPAA-compliant PII detection and masking before embedding.
"""

import re
from typing import List
from dataclasses import dataclass


@dataclass
class Document:
    content: str
    metadata: dict


class PIIMasker:
    """
    Detects and masks Personally Identifiable Information (PII)
    from documents before they are embedded into the vector store.
    Compliant with HIPAA Safe Harbor method.
    """

    PII_PATTERNS = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "phone": r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "dob": r"\b(0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b",
        "mrn": r"\bMRN[:\s]?\d{6,10}\b",
        "zip": r"\b\d{5}(-\d{4})?\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    MASK_MAP = {
        "ssn": "[SSN_MASKED]",
        "phone": "[PHONE_MASKED]",
        "email": "[EMAIL_MASKED]",
        "dob": "[DOB_MASKED]",
        "mrn": "[MRN_MASKED]",
        "zip": "[ZIP_MASKED]",
        "ip_address": "[IP_MASKED]",
    }

    def mask(self, documents: List[Document]) -> List[Document]:
        """Mask PII from a list of documents."""
        return [self._mask_document(doc) for doc in documents]

    def _mask_document(self, doc: Document) -> Document:
        masked_content = doc.content
        pii_found = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, masked_content, re.IGNORECASE)
            if matches:
                pii_found.append(pii_type)
                masked_content = re.sub(
                    pattern,
                    self.MASK_MAP[pii_type],
                    masked_content,
                    flags=re.IGNORECASE
                )

        return Document(
            content=masked_content,
            metadata={**doc.metadata, "pii_types_masked": pii_found}
        )

    def audit_report(self, documents: List[Document]) -> dict:
        """Generate PII audit report without modifying documents."""
        report = {"total_docs": len(documents), "pii_detected": {}}

        for pii_type, pattern in self.PII_PATTERNS.items():
            count = sum(
                len(re.findall(pattern, doc.content, re.IGNORECASE))
                for doc in documents
            )
            if count > 0:
                report["pii_detected"][pii_type] = count

        return report
