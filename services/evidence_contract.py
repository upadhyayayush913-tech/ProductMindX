"""
Evidence and Execution Trace Contracts for ProductMind X.

Provides standardized, immutable data contracts for:
1. EvidenceRecord: Canonical representation of a single piece of retrieved/referenced evidence.
2. ExecutionTraceRecord: Transparent audit log for every analytical and retrieval operation executed.
3. AnalysisPlan: Compact cognitive execution plan listing required analytical steps.
4. CounterEvidenceSummary: Assessment of contradictory evidence, unsupported claims, and quotation fidelity.
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import time
import datetime
import uuid

@dataclass
class EvidenceRecord:
    review_id: str                              # Verifiable ID, e.g. 'REV-B09XS7-U1001-9600'
    product_asin: str                           # Parent ASIN or ASIN
    product_title: str                          # Product name
    category: str                               # Catalog category
    aspect: Optional[str] = None                # Referenced product aspect (e.g. 'battery life')
    excerpt: str = ""                           # Exact excerpt or review body text
    rating: float = 3.0                         # Star rating (1.0 to 5.0)
    timestamp: int = 0                          # Epoch timestamp
    verified_purchase: bool = True              # Verified purchase flag
    retrieval_method: str = "hybrid"            # 'dense', 'bm25', 'hybrid', 'direct_filter', 'catalog'
    retrieval_score: float = 1.0                # Normalized relevance or similarity score
    applicable_filter_scope: Optional[Dict[str, Any]] = None # Filter context under which this was retrieved
    supports_claim: bool = True                 # True if supporting; False if counter-evidence/contradictory
    validation_state: str = "VERIFIED"          # 'VERIFIED', 'UNVERIFIED', 'CONTRADICTED'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "product_asin": self.product_asin,
            "product_title": self.product_title,
            "category": self.category,
            "aspect": self.aspect,
            "excerpt": self.excerpt,
            "rating": self.rating,
            "timestamp": self.timestamp,
            "verified_purchase": self.verified_purchase,
            "retrieval_method": self.retrieval_method,
            "retrieval_score": round(self.retrieval_score, 4),
            "applicable_filter_scope": self.applicable_filter_scope or {},
            "supports_claim": self.supports_claim,
            "validation_state": self.validation_state
        }


@dataclass
class StepExecutionRecord:
    step_id: str
    name: str                                   # e.g. 'Aspect Sentiment Analysis'
    operation_type: str                         # 'retrieval', 'analytics', 'xai', 'counter_evidence', 'synthesis'
    status: str = "PENDING"                     # 'PENDING', 'RUNNING', 'SUCCESS', 'FAILED', 'SKIPPED'
    inputs_summary: str = ""
    outputs_summary: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "operation_type": self.operation_type,
            "status": self.status,
            "inputs_summary": self.inputs_summary,
            "outputs_summary": self.outputs_summary,
            "evidence_ids": self.evidence_ids,
            "error_message": self.error_message,
            "execution_time_ms": round(self.execution_time_ms, 2)
        }


@dataclass
class CognitiveAnalysisPlan:
    plan_id: str
    query: str
    resolved_intent: str
    resolved_entities: List[str]
    steps: List[StepExecutionRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "query": self.query,
            "resolved_intent": self.resolved_intent,
            "resolved_entities": self.resolved_entities,
            "created_at": self.created_at,
            "steps": [s.to_dict() for s in self.steps]
        }


@dataclass
class CounterEvidenceSummary:
    has_counter_evidence: bool = False
    counter_evidence_count: int = 0
    contradictory_review_ids: List[str] = field(default_factory=list)
    contradictory_snippets: List[Dict[str, Any]] = field(default_factory=list)
    claim_re_evaluation_note: Optional[str] = None
    quotation_fidelity_score: float = 1.0       # 0.0 to 1.0: exactness of cited quotes
    unsupported_claims_flagged: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_counter_evidence": self.has_counter_evidence,
            "counter_evidence_count": self.counter_evidence_count,
            "contradictory_review_ids": self.contradictory_review_ids,
            "contradictory_snippets": self.contradictory_snippets,
            "claim_re_evaluation_note": self.claim_re_evaluation_note,
            "quotation_fidelity_score": round(self.quotation_fidelity_score, 2),
            "unsupported_claims_flagged": self.unsupported_claims_flagged
        }
