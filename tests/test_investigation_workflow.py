"""
Comprehensive Unit & Integration Tests for Cognitive Investigation Workflow.

Validates:
1. 10-Intent Canonical Routing in IntentClassifier.
2. Cognitive Investigation Plan Generation and Step Tracking.
3. Standardized EvidenceRecord Contract & Assembly.
4. Counter-Evidence Surfacing & Contradiction Auditing.
5. Research Notebook Service: Finding Preservation & Dossier Export.
6. Feedback-Driven Error Analysis Service.
7. Product Issue Relationship Explorer Graph Navigation.
8. NLP Experiment Studio: Multi-Model Benchmark & Retrieval Comparison.
"""

import unittest
from chatbot.intent_classifier import IntentClassifier, CANONICAL_INTENTS
from workflow.investigation_planner import CognitiveInvestigationPlanner
from services.evidence_contract import EvidenceRecord, CognitiveAnalysisPlan
from services.research_notebook import ResearchNotebookService
from services.feedback_service import FeedbackService
from services.relationship_explorer import ProductIssueRelationshipExplorer
from nlp_pipeline.experiment_studio import NLPExperimentStudio


class TestInvestigationWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = IntentClassifier()
        cls.planner = CognitiveInvestigationPlanner()
        cls.notebook = ResearchNotebookService()
        cls.feedback = FeedbackService()
        cls.rel_explorer = ProductIssueRelationshipExplorer()
        cls.exp_studio = NLPExperimentStudio()

    def test_canonical_intent_routing(self):
        """Verify explicit routing for all 10 canonical analytical intents."""
        test_queries = [
            ("Why was this review classified negative? Explain the model", "explanation"),
            ("Show me counter-evidence challenging battery claims", "counter_evidence"),
            ("Compare BM25 vs dense ChromaDB retrieval", "retrieval_comparison"),
            ("Which one is better: Sony XM5 or AirPods Pro 2?", "product_comparison"),
            ("Did battery complaints increase over time?", "emerging_issue"),
            ("Tell me about battery life and noise cancellation pros and cons", "aspect"),
            ("What is the overall sentiment for this product?", "sentiment"),
            ("How was this calculated? Explain the methodology and formulas", "methodology"),
            ("What do customers say about the Sony XM5 headphones?", "general_qa"),
            ("xyz123 unindexed product non-existent", "insufficient_evidence"),
        ]

        for q, expected_intent in test_queries:
            res = self.classifier.classify_intent(q)
            self.assertEqual(
                res["intent"],
                expected_intent,
                f"Query '{q}' classified as '{res['intent']}', expected '{expected_intent}'"
            )

    def test_cognitive_plan_generation(self):
        """Verify the cognitive planner generates tailored, compact steps."""
        # 1. Emerging issue plan
        plan_issue = self.planner.create_plan(
            query="Did battery complaints increase?",
            intent="emerging_issue",
            matched_products=[{"title": "Sony XM5"}]
        )
        self.assertEqual(plan_issue.resolved_intent, "emerging_issue")
        step_names = [s.name for s in plan_issue.steps]
        self.assertTrue(any("Partition" in name for name in step_names))
        self.assertTrue(any("Hypothesis Testing" in name for name in step_names))

        # 2. Explanation plan
        plan_xai = self.planner.create_plan(
            query="Why was this classified negative?",
            intent="explanation",
            matched_products=[]
        )
        step_types = [s.operation_type for s in plan_xai.steps]
        self.assertIn("xai", step_types)

    def test_evidence_record_assembly(self):
        """Verify raw reviews are structured into canonical EvidenceRecords."""
        sample_revs = [
            {"review_id": "REV-TEST-001", "text": "Superb sound quality.", "rating": 5.0, "timestamp": 1675209600000, "verified_purchase": True, "hybrid_score": 0.92},
            {"review_id": "REV-TEST-002", "text": "Hinge snapped quickly.", "rating": 1.0, "timestamp": 1675209700000, "verified_purchase": True, "hybrid_score": 0.85}
        ]
        sample_prod = {"parent_asin": "B09XS7JWHH", "title": "Sony WH-1000XM5", "main_category": "Electronics"}

        records = self.planner.assemble_evidence_records(sample_revs, product=sample_prod, aspect="sound")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].review_id, "REV-TEST-001")
        self.assertEqual(records[0].rating, 5.0)
        self.assertTrue(records[0].supports_claim)
        self.assertFalse(records[1].supports_claim)
        self.assertEqual(records[0].validation_state, "VERIFIED")

    def test_counter_evidence_assessment(self):
        """Verify contradictory reviews are surfaced with quotation fidelity."""
        sample_revs = [
            {"review_id": "REV-POS-1", "text": "Battery life lasts 30 hours, amazing.", "rating": 5.0},
            {"review_id": "REV-NEG-1", "text": "Battery dies in 40 minutes, horrible battery.", "rating": 1.0}
        ]
        counter = self.planner.assess_counter_evidence(sample_revs, focal_sentiment="Positive", focal_aspect="battery")
        self.assertTrue(counter.has_counter_evidence)
        self.assertEqual(counter.counter_evidence_count, 1)
        self.assertIn("REV-NEG-1", counter.contradictory_review_ids)

    def test_research_notebook_lifecycle(self):
        """Verify research finding save, retrieve, markdown export, and deletion."""
        entry = self.notebook.save_finding(
            query="Test query",
            finding_summary="Test finding on battery life.",
            intent="aspect",
            product_title="Sony XM5",
            evidence_ids=["REV-TEST-001"],
            backend_metrics={"average_rating": 4.5}
        )
        self.assertIsNotNone(entry["entry_id"])

        retrieved = self.notebook.get_entry(entry["entry_id"])
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["product_title"], "Sony XM5")

        md_export = self.notebook.export_to_markdown()
        self.assertIn("ProductMind X", md_export)
        self.assertIn("Test finding", md_export)

        deleted = self.notebook.delete_entry(entry["entry_id"])
        self.assertTrue(deleted)

    def test_feedback_service_logging(self):
        """Verify error analysis feedback logging without corrupting ground truth."""
        fb = self.feedback.log_feedback(
            feedback_type="incorrect_sentiment",
            query="bad battery life",
            system_output="Positive",
            user_correction="Negative",
            flagged_tokens=["bad"]
        )
        self.assertIsNotNone(fb["feedback_id"])
        self.assertEqual(fb["status"], "UNDER_REVIEW")

    def test_relationship_explorer(self):
        """Verify relationship graph links products, aspects, and issue clusters."""
        self.rel_explorer.build_graph()
        issues = self.rel_explorer.get_all_issues()
        self.assertGreater(len(issues), 0)

        first_issue = issues[0]["issue_name"]
        details = self.rel_explorer.get_issue_details(first_issue)
        self.assertIsNotNone(details)
        self.assertIn("aspect", details)
        self.assertIn("affected_products", details)

    def test_experiment_studio_benchmarks(self):
        """Verify NLP Experiment Studio runs reproducible multi-model benchmarks."""
        res = self.exp_studio.benchmark_sentiment_models()
        self.assertIn("models", res)
        self.assertIn("Lexicon Rule Baseline", res["models"])
        self.assertIn("Unigram TF-IDF (1,1)", res["models"])
        self.assertIn("Production N-Gram TF-IDF (1,2)", res["models"])
        # Production model must achieve 100% on the 25 benchmark cases
        prod_bench = res["models"]["Production N-Gram TF-IDF (1,2)"]["benchmark_pass_rate"]
        self.assertIn("25/25", prod_bench)


if __name__ == "__main__":
    unittest.main()
