"""
NLP Pipeline Manager & Knowledge Base Indexer.
Executes the end-to-end NLP processing flow:
1. Ingest reviews and metadata
2. Sentiment Analysis (VADER + calibration)
3. Aspect-Based Sentiment Analysis (Pros/Cons)
4. Problem & Emerging Issue Detection (Trends & Severity)
5. Topic Modeling (NMF / TF-IDF)
6. Population of ChromaDB vector collections
"""

import json
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from config import CACHE_DIR, CATEGORIES
from data.data_loader import AmazonDataLoader
from nlp_pipeline.sentiment_analyzer import SentimentAnalyzer
from nlp_pipeline.aspect_extractor import AspectExtractor
from nlp_pipeline.issue_detector import IssueDetector
from nlp_pipeline.topic_modeler import ReviewTopicModeler
from nlp_pipeline.text_summarizer import ReviewSummarizer
from nlp_pipeline.translator import translation_engine
from rag.vector_store import ChromaVectorStore
from utils.logger import get_logger

logger = get_logger("pipeline_manager")

class NLPPipelineManager:
    def __init__(self, vector_store: Optional[ChromaVectorStore] = None):
        self.loader = AmazonDataLoader()
        self.sentiment_analyzer = SentimentAnalyzer(use_transformer=False)
        self.aspect_extractor = AspectExtractor()
        self.issue_detector = IssueDetector()
        self.topic_modeler = ReviewTopicModeler()
        self.summarizer = ReviewSummarizer()
        self.translator = translation_engine
        self.vector_store = vector_store or ChromaVectorStore()
        self.insights_cache: Dict[str, Dict[str, Any]] = {}

    def get_cache_file(self) -> Path:
        return CACHE_DIR / "processed_nlp_insights.json"

    def load_cached_insights(self) -> Dict[str, Dict[str, Any]]:
        """Load pre-computed NLP insights if available."""
        f = self.get_cache_file()
        if f.exists():
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    self.insights_cache = json.load(fp)
                    logger.info(f"Loaded NLP insights cache for {len(self.insights_cache)} products.")
                    return self.insights_cache
            except Exception as e:
                logger.warning(f"Failed to read NLP cache: {e}")
        return {}

    def save_insights_cache(self):
        """Save computed NLP insights to disk."""
        f = self.get_cache_file()
        try:
            with open(f, "w", encoding="utf-8") as fp:
                json.dump(self.insights_cache, fp, indent=2)
            logger.info(f"Saved NLP insights cache to {f}")
        except Exception as e:
            logger.warning(f"Failed to write NLP cache: {e}")

    def process_and_index_product(self, product_meta: Any, reviews: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run all NLP tasks for a single product and index into vector store."""
        if isinstance(product_meta, str):
            asin = product_meta
            all_prods = self.loader.get_all_products()
            product_meta = next((p for p in all_prods if (p.get("parent_asin") or p.get("asin")) == asin), {"parent_asin": asin, "title": asin})

        asin = product_meta.get("parent_asin") or product_meta.get("asin", "unknown")
        title = product_meta.get("title", "Unknown Product")
        category = product_meta.get("main_category", "Electronics")
        brand = product_meta.get("store", "Brand")
        price = product_meta.get("price", "N/A")
        rating = product_meta.get("average_rating", 4.5)

        if reviews is None:
            try:
                rev_df = self.loader.load_reviews(category)
                matched_rows = rev_df[rev_df["parent_asin"] == asin]
                reviews = matched_rows.to_dict("records") if not matched_rows.empty else rev_df.head(5).to_dict("records")
            except Exception:
                reviews = []

        rating_count = product_meta.get("rating_number", len(reviews))
        desc = product_meta.get("description", "")
        features = product_meta.get("features", [])

        logger.info(f"Processing NLP pipeline for product: {title} ({len(reviews)} reviews)")

        # 0. Cross-Border Multilingual Translation & Normalization
        processed_reviews = []
        translated_count = 0
        for r in reviews:
            r_copy = dict(r)
            text = r_copy.get("text", "")
            raw_lang = r_copy.get("language")
            if isinstance(raw_lang, str) and raw_lang.strip():
                detected_lang = raw_lang.strip().lower()
            else:
                detected_lang = self.translator.detect_language(text)

            if detected_lang != "en":
                r_copy["original_text"] = text
                r_copy["original_language"] = detected_lang
                r_copy["translated"] = True
                t_res = self.translator.translate(text, source_lang=detected_lang, target_lang="en")
                r_copy["text"] = t_res["translated_text"]
                translated_count += 1

                title_txt = r_copy.get("title", "")
                if title_txt:
                    t_title = self.translator.translate(title_txt, source_lang=detected_lang, target_lang="en")
                    r_copy["original_title"] = title_txt
                    r_copy["title"] = t_title["translated_text"]
            else:
                r_copy["original_language"] = "en"
                r_copy["translated"] = False
            processed_reviews.append(r_copy)

        # 1. Aspect-Based Sentiment Analysis (Pros & Cons)
        absa_results = self.aspect_extractor.aggregate_product_aspects(processed_reviews)
        pros = absa_results["pros"]
        cons = absa_results["cons"]

        # 2. Problem & Emerging Issue Detection
        issues = self.issue_detector.detect_product_issues(processed_reviews)

        # 3. Topic Modeling
        review_texts = [r.get("text", "") for r in processed_reviews if r.get("text")]
        topics = self.topic_modeler.fit_transform_reviews(review_texts)

        # 4. Text Summarization
        summaries = self.summarizer.summarize_reviews(processed_reviews)

        # Structure complete insights
        insights = {
            "asin": asin,
            "title": title,
            "category": category,
            "total_reviews_analyzed": len(processed_reviews),
            "pros": pros,
            "cons": cons,
            "issues": issues,
            "topics": topics,
            "summaries": summaries,
            "multilingual_stats": {
                "total_reviews": len(reviews),
                "translated_reviews_count": translated_count,
                "translated_count": translated_count,
                "non_english_detected": translated_count > 0,
                "languages_detected": [l for l in list({r.get("original_language", "en") for r in processed_reviews}) if isinstance(l, str)]
            }
        }
        self.insights_cache[asin] = insights

        # --- Index into ChromaDB Vector Collections ---

        # A. Product Profile Document
        profile_doc = (
            f"Product: {title}\n"
            f"Category: {category}\n"
            f"Brand: {brand}\n"
            f"Price: {price}\n"
            f"Average Rating: {rating}/5.0 ({rating_count} reviews)\n"
            f"Description: {desc}\n"
            f"Key Features: {', '.join(str(f) for f in features) if hasattr(features, '__iter__') and not isinstance(features, (str, bytes)) else str(features)}"
        )
        self.vector_store.add_product_profile(
            asin=asin,
            title=title,
            category=category,
            content=profile_doc,
            metadata={
                "brand": brand,
                "price": price,
                "rating": float(rating)
            }
        )

        # B. Pros & Cons Insights Document
        pros_text = "\n".join([f"- {p['aspect']}: {p['positive_percentage']}% positive ({p['mention_count']} mentions)" for p in pros])
        cons_text = "\n".join([f"- {c['aspect']}: {100 - c['positive_percentage']:.0f}% negative ({c['mention_count']} mentions)" for c in cons])
        insights_doc = (
            f"Product: {title}\n"
            f"PROS:\n{pros_text if pros_text else 'No distinct pros recorded.'}\n\n"
            f"CONS:\n{cons_text if cons_text else 'No distinct cons recorded.'}"
        )
        self.vector_store.add_product_insight(
            asin=asin,
            title=title,
            insight_type="pros_cons",
            content=insights_doc
        )

        # C. Known Issues Document
        issues_text = "\n".join([
            f"- {iss['issue']} (Severity: {iss['severity']}, Trend: {iss['trend']}, Frequency: {iss['frequency']})"
            for iss in issues
        ])
        issues_doc = (
            f"Product: {title}\n"
            f"KNOWN ISSUES & DEFECT REPORTS:\n{issues_text if issues_text else 'No major failure issues reported.'}"
        )
        self.vector_store.add_product_insight(
            asin=asin,
            title=title,
            insight_type="issues",
            content=issues_doc
        )

        # D. Raw Review Excerpt Chunks
        self.vector_store.add_review_chunks(
            asin=asin,
            title=title,
            chunks=processed_reviews
        )

        return insights

    def get_status_file(self) -> Path:
        return CACHE_DIR / "nlp_processing_status.json"

    def get_processing_status(self) -> Dict[str, Any]:
        """Return current status of NLP processing and knowledge base index."""
        status_file = self.get_status_file()
        if status_file.exists():
            try:
                with open(status_file, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception:
                pass
        
        # Calculate on the fly if not yet persisted
        cached = self.load_cached_insights()
        return {
            "total_indexed_reviews": sum(len(c.get("summaries", {}).get("aspect_bullets", [])) for c in cached.values()),
            "usable_text": len(cached),
            "pending_nlp": 0,
            "analyzed_nlp": len(cached),
            "processed_products_count": len(cached)
        }

    def save_processing_status(self, status: Dict[str, Any]):
        """Persist NLP processing metrics."""
        status_file = self.get_status_file()
        try:
            with open(status_file, "w", encoding="utf-8") as fp:
                json.dump(status, fp, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write NLP processing status: {e}")

    def build_full_knowledge_base(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Index all products and reviews across all categories."""
        if not force_refresh and self.load_cached_insights():
            logger.info("Using existing NLP insights cache.")
            return self.insights_cache

        logger.info("Building complete NLP Knowledge Base and Vector Index...")
        all_products = self.loader.get_all_products()

        total_reviews_indexed = 0
        usable_text_count = 0

        for prod in all_products:
            asin = prod.get("parent_asin") or prod.get("asin")
            cat = prod.get("main_category", "Electronics")
            # Load reviews for this category
            rev_df = self.loader.load_reviews(cat)
            prod_reviews = rev_df[rev_df["parent_asin"] == asin].to_dict("records")
            
            # If not found in primary category, check other categories before giving up
            if not prod_reviews:
                for alt_cat in CATEGORIES:
                    if alt_cat != cat:
                        alt_df = self.loader.load_reviews(alt_cat)
                        matched = alt_df[alt_df["parent_asin"] == asin].to_dict("records")
                        if matched:
                            prod_reviews = matched
                            break

            # Strictly do not assign arbitrary reviews to a product
            if prod_reviews:
                total_reviews_indexed += len(prod_reviews)
                usable_text_count += sum(1 for r in prod_reviews if r.get("text") and len(str(r["text"]).strip()) > 5)

            self.process_and_index_product(prod, prod_reviews)

        self.save_insights_cache()

        status = {
            "total_indexed_reviews": total_reviews_indexed,
            "usable_text": usable_text_count,
            "pending_nlp": 0,
            "analyzed_nlp": len(all_products),
            "processed_products_count": len(all_products),
            "categories_covered": CATEGORIES,
            "last_updated": datetime.datetime.now().isoformat()
        }
        self.save_processing_status(status)

        return self.insights_cache
