"""
Evidence-Backed Research Notebook Service for ProductMind X.

Allows researchers and analysts to:
1. Save analytical findings with full provenance:
   - Query & Timestamp
   - Resolved Intent & Product
   - Finding Summary
   - Verified Evidence IDs ([Review #ID], [Metadata #ASIN])
   - Deterministic Backend Calculations
   - Counter-Evidence & Contradictions
   - Active Filter Scope
   - Pinned Model & Classifier Architecture
2. List, retrieve, and delete saved notebook entries.
3. Export findings to structured Markdown reports and JSON data files.
"""

import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from config import CACHE_DIR
from utils.logger import get_logger

logger = get_logger("research_notebook")

NOTEBOOK_DIR = CACHE_DIR / "research_notebook"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_STORE_PATH = NOTEBOOK_DIR / "saved_findings.json"


class ResearchNotebookService:
    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or NOTEBOOK_STORE_PATH
        self._entries: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load research notebook: {e}")
        return []

    def _save(self):
        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save research notebook: {e}")

    def save_finding(
        self,
        query: str,
        finding_summary: str,
        intent: str,
        product_title: Optional[str] = None,
        product_asin: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        backend_metrics: Optional[Dict[str, Any]] = None,
        counter_evidence_summary: Optional[Dict[str, Any]] = None,
        active_filters: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Save a new research finding with immutable provenance."""
        entry_id = f"NOTE-{uuid.uuid4().hex[:8].upper()}"
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "entry_id": entry_id,
            "created_at": now_str,
            "query": query,
            "intent": intent,
            "product_title": product_title or "Catalog Wide",
            "product_asin": product_asin or "ALL",
            "finding_summary": finding_summary,
            "evidence_ids": evidence_ids or [],
            "backend_metrics": backend_metrics or {},
            "counter_evidence": counter_evidence_summary or {},
            "active_filters": active_filters or {},
            "model_name": model_name or "groq/openai/gpt-oss-120b",
            "tags": tags or [intent]
        }

        self._entries.insert(0, entry)
        self._save()
        logger.info(f"Saved research notebook finding: {entry_id}")
        return entry

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Return all saved findings chronologically (newest first)."""
        return list(self._entries)

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single entry by ID."""
        for e in self._entries:
            if e["entry_id"] == entry_id:
                return e
        return None

    def delete_entry(self, entry_id: str) -> bool:
        """Remove a saved finding."""
        initial_len = len(self._entries)
        self._entries = [e for e in self._entries if e["entry_id"] != entry_id]
        if len(self._entries) < initial_len:
            self._save()
            return True
        return False

    def export_to_markdown(self) -> str:
        """Export all findings to a consolidated Markdown research dossier."""
        lines = [
            "# ProductMind X — Evidence-Backed Consumer Intelligence Dossier",
            f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Total Saved Findings: {len(self._entries)}*",
            "\n---\n"
        ]

        for idx, e in enumerate(self._entries, 1):
            lines.append(f"## {idx}. [{e['entry_id']}] {e['product_title']}")
            lines.append(f"**Research Question:** {e['query']}")
            lines.append(f"**Resolved Intent:** `{e['intent']}` | **Date:** {e['created_at']}")
            lines.append(f"**Model:** `{e['model_name']}`\n")
            lines.append(f"### Finding Summary\n{e['finding_summary']}\n")

            if e.get("evidence_ids"):
                lines.append(f"**Verified Evidence Citations:** {', '.join(e['evidence_ids'])}")

            if e.get("backend_metrics"):
                bm = e["backend_metrics"]
                lines.append(
                    f"**Calculations:** Average Rating: {bm.get('average_rating', 'N/A')}★ | "
                    f"Verified Purchase Ratio: {bm.get('verified_ratio_pct', 'N/A')}% | "
                    f"Total Reviews: {bm.get('total_reviews', 'N/A')}"
                )

            ce = e.get("counter_evidence", {})
            if ce.get("has_counter_evidence"):
                lines.append(f"**Counter-Evidence Audit:** {ce.get('claim_re_evaluation_note', 'Opposing reviews detected.')}")

            lines.append("\n---\n")

        return "\n".join(lines)


# Global singleton
research_notebook = ResearchNotebookService()
