"""
Feedback-Driven Error Analysis Service for ProductMind X.

Allows users to audit and flag:
1. Incorrect sentiment labels (e.g. model predicted Positive for negative complaint)
2. Irrelevant retrieval chunks (e.g. chunk did not match query intent)
3. Unsupported claims / hallucination flags
4. Misleading explanation / XAI attributions

Stores feedback separately from verified ground truth in a reviewable audit log.
Does NOT automatically retrain model on unverified user submissions.
"""

import os
import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from config import CACHE_DIR
from utils.logger import get_logger

logger = get_logger("feedback_service")

FEEDBACK_DIR = CACHE_DIR / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_STORE_PATH = FEEDBACK_DIR / "user_feedback.json"

FEEDBACK_TYPES = [
    "incorrect_sentiment",
    "irrelevant_retrieval",
    "unsupported_claim",
    "misleading_explanation",
    "other_error"
]


class FeedbackService:
    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or FEEDBACK_STORE_PATH
        self._feedback_list: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load user feedback: {e}")
        return []

    def _save(self):
        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(self._feedback_list, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user feedback: {e}")

    def log_feedback(
        self,
        feedback_type: str,
        query: str,
        system_output: str,
        user_correction: str,
        context_id: Optional[str] = None,
        flagged_tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Record a user audit flag with operational metadata."""
        feedback_id = f"FB-{uuid.uuid4().hex[:8].upper()}"
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "feedback_id": feedback_id,
            "created_at": now_str,
            "feedback_type": feedback_type if feedback_type in FEEDBACK_TYPES else "other_error",
            "query": query,
            "system_output": system_output,
            "user_correction": user_correction,
            "context_id": context_id or "NONE",
            "flagged_tokens": flagged_tokens or [],
            "status": "UNDER_REVIEW",           # 'UNDER_REVIEW', 'VERIFIED_BUG', 'DISMISSED'
            "notes": "Logged for offline scientific error analysis."
        }

        self._feedback_list.insert(0, entry)
        self._save()
        logger.info(f"Recorded error analysis feedback: {feedback_id} ({feedback_type})")
        return entry

    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """Return all logged feedback entries."""
        return list(self._feedback_list)

    def update_feedback_status(self, feedback_id: str, new_status: str, audit_notes: str) -> bool:
        """Update audit review status of a feedback entry."""
        for f in self._feedback_list:
            if f["feedback_id"] == feedback_id:
                f["status"] = new_status
                f["audit_notes"] = audit_notes
                f["reviewed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._save()
                return True
        return False


# Global singleton
feedback_service = FeedbackService()
