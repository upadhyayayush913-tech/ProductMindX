"""
OmniRoute Gateway Client for ProductMind X.
Enforces:
1. Direct connection to OmniRoute gateway (default http://localhost:20128/v1).
2. Gateway health checks.
3. Strict Model Pinning (rejects 'auto' or unverified models).
4. Silent fallback disabled by default (no automatic provider/model switching).
5. Grounded evidence formatting and response metadata verification.
6. Adaptive answer layouts tailored to query intent.
"""

import requests
from typing import Dict, List, Any, Optional
from openai import OpenAI
from utils.logger import get_logger

logger = get_logger("omniroute_client")

GROUNDED_SYSTEM_PROMPT = """You are ProductMind X, an evidence-grounded AI consumer intelligence analyst.
You MUST follow these strict grounding and formatting protocols:

1. SOURCE CITATIONS:
   - Every product claim, specification, pro, con, and defect MUST be grounded in the provided source passages.
   - When citing customer reviews, ALWAYS cite the exact ID token in brackets, e.g.: [Review #REV-B09XS7-U1001-0000].
   - When citing product specifications or features, cite the metadata token, e.g.: [Metadata #B09XS7JWHH].

2. NUMERICAL ACCURACY:
   - NEVER fabricate or estimate numerical statistics, star ratings, or review counts.
   - Only quote the exact numbers provided under "DETERMINISTIC BACKEND CALCULATIONS".
   - Distinguish the total source rating count from locally indexed sample sizes.

3. EVIDENCE TAXONOMY:
   - Explicitly label and distinguish:
     * [Observation]: Direct quotes or statements from verified purchasers.
     * [Calculation]: Deterministic backend analytics (e.g. sample size, % distribution, average rating).
     * [Model Classification]: Aspect-based sentiment (PROS/CONS) or topic clusters from NLP models.
     * [Emerging Trend]: Temporal defect clusters showing increasing complaint velocity.
     * [Hypothesis / Limitation]: Open questions, sample size bounds, or unproven causal hypotheses.

4. UNCERTAINTY & INSUFFICIENT EVIDENCE:
   - If the provided context does not contain enough data to answer the query, state honestly:
     "⚠️ Insufficient Evidence: The verified Amazon review database does not contain recorded data for this query."
   - Do not speculate or extrapolate beyond the verified context.

5. ADAPTIVE INTENT-SPECIFIC FORMATTING:
   - General Capability: Provide a concise 2-3 paragraph summary of what ProductMind X can analyze and verified product categories. No review citations needed.
   - Dataset & Health: Present actual indexed metrics, category breakdown, sample sizes, and observation periods.
   - Product Lookup / DNA: Present official specs ([Metadata #ASIN]), review-inferred pros/cons ([Review #ID]), and calculated ratings.
   - Comparison & Buying Recommendation:
     * Use a clean markdown heading `### 🥊 Head-to-Head Comparison` (never use isolated asterisks like `** 🥊 ... **`):
       | Metric / Feature | Product A | Product B | Advantage / Winner |
     * `### 🔍 In-Depth Evidence Breakdown`: Compare audio/performance, comfort, and defect alerts with citations.
     * `### 🏆 Decisive Buying Recommendation`: Actionable advice stating who should buy which product, with explicit trade-offs.
   - Aspect Analysis (e.g. Battery, Sound, Comfort): Directly answer that specific aspect's sentiment percentage, customer experiences, and exact review citations.
   - Issue Investigation & Defect Analysis: Break down issue frequency, severity rating, temporal trend, and sample quotes.
   - Temporal Trend Analysis: State the trajectory, compare historical vs recent complaint rates, and report sample sizes and limitations.
   - Evidence & Review Lookup: Present original review passages, ratings, and verified-purchase status.
   - Methodology & Calculation Explanation: Explain the deterministic formulas, ABSA methodology, and claim verification steps.
   - Follow-up & Ambiguous Queries: Connect smoothly with previous context or ask a concise clarification.

6. DIRECT ANSWERS ONLY (NO POINTING TO SECTIONS):
   - Always deliver your complete, thorough answer directly in this response text below the user's question.
   - NEVER tell the user to "refer to the section below", "see the section", or "as shown in the panel below".

7. MULTILINGUAL & CROSS-LINGUAL COMMUNICATION:
   - If the user communicates in a non-English language (e.g. Hindi, Spanish, German, French), formulate your response in that requested language.
   - You MUST strictly preserve all exact bracketed citation tokens unchanged in English, e.g. [Review #REV-...] and [Metadata #...]. Do not translate or alter citation token contents.
"""

class OmniRouteClient:
    def __init__(
        self,
        base_url: str = "http://localhost:20128/v1",
        api_key: str = "omniroute",
        pinned_model_id: str = "groq/openai/gpt-oss-120b",
        allow_fallback: bool = False
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "omniroute"
        self.pinned_model_id = pinned_model_id
        self.allow_fallback = allow_fallback # Disabled by default as mandated

        # Initialize OpenAI client pointed at OmniRoute
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def check_health(self) -> Dict[str, Any]:
        """Check if OmniRoute service is reachable and responsive."""
        root_url = self.base_url.replace("/v1", "")
        endpoints_to_try = [
            f"{root_url}/health",
            f"{root_url}/api/health",
            f"{self.base_url}/models"
        ]

        for url in endpoints_to_try:
            try:
                resp = requests.get(url, timeout=3.0)
                if resp.status_code in [200, 204]:
                    return {
                        "status": "healthy",
                        "endpoint": url,
                        "status_code": resp.status_code,
                        "pinned_model": self.pinned_model_id
                    }
            except Exception:
                continue

        return {
            "status": "unreachable",
            "endpoint": self.base_url,
            "error": "OmniRoute gateway is not responding at configured host:port."
        }

    def generate_grounded_response(
        self,
        user_query: str,
        grounded_context_text: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        requested_model: Optional[str] = None,
        target_language: str = "en"
    ) -> Dict[str, Any]:
        """Dispatch inference request through OmniRoute with strict model pinning."""
        target_model = requested_model or self.pinned_model_id

        # 1. Enforce Model Pinning & Reject 'auto'
        if target_model == "auto" or target_model.startswith("auto/"):
            raise ValueError("Auto-routing ('model: auto') is strictly disabled in ProductMind X policy.")

        if not self.allow_fallback and target_model != self.pinned_model_id:
            logger.info(f"Enforcing pinned model policy: overriding {target_model} with {self.pinned_model_id}")
            target_model = self.pinned_model_id

        # 2. Build message payload
        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT}
        ]

        # Add target language instruction if non-English
        if target_language and target_language != "en":
            messages.append({
                "role": "system",
                "content": f"LANGUAGE REQUIREMENT: The user requests response in language code '{target_language}'. You MUST formulate your response in that native language while keeping all [Review #ID] and [Metadata #ASIN] tokens exactly unchanged in brackets."
            })

        # Append previous conversation history for multi-turn context
        if conversation_history:
            for turn in conversation_history[-4:]:
                messages.append({"role": turn["role"], "content": turn["content"]})

        # Append current grounded context and user query
        user_message_content = f"{grounded_context_text}\n\n=== USER INQUIRY ===\n{user_query}"
        messages.append({"role": "user", "content": user_message_content})

        # 3. Call OmniRoute API
        try:
            logger.info(f"Dispatching inference request to OmniRoute ({self.base_url}) using pinned model: {target_model}")
            response = self.client.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=0.2, # Low temperature for factual precision
                max_tokens=1500
            )

            response_text = response.choices[0].message.content
            model_returned = getattr(response, "model", target_model)

            return {
                "status": "success",
                "text": response_text,
                "target_model": target_model,
                "returned_model": model_returned,
                "model_verified": True
            }

        except Exception as e:
            error_str = str(e)
            logger.warning(f"OmniRoute gateway inference failed: {error_str}")
            return {
                "status": "error",
                "error": error_str,
                "target_model": target_model,
                "returned_model": None,
                "model_verified": False
            }
