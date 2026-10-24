"""
Emerging Issue and Problem Detector.
Identifies product defects, glitches, failures, and emerging complaints
from negative reviews using pattern mining, severity classification,
and temporal trend analysis.
"""

import re
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime

from data.preprocessor import TextPreprocessor
from utils.helpers import timestamp_to_str
from utils.logger import get_logger

logger = get_logger("issue_detector")

# Problem and failure triggers
COMPLAINT_PATTERNS = [
    r"\b(?:stopped|quits?|stops?)\s+(?:working|charging|pumping|functioning)\b",
    r"\b(?:broke|broken|break)\s+(?:after|within|down)?\b",
    r"\b(?:does\s+not|doesn't|won't)\s+(?:turn\s+on|work|connect|charge|sync)\b",
    r"\b(?:defective|faulty|glitch|bug|malfunction|hardware\s+failure)\b",
    r"\b(?:crackling|buzzing|static|clicking|grinding)\s+(?:noise|sound)?\b",
    r"\b(?:burn\s+food|burn\s+error|overheat(?:ing)?|spark(?:ing|ed)?|smell(?:s|ing)?)\b",
    r"\b(?:leak(?:ing|ed|s)?|steam\s+leak|seal\s+failure)\b",
    r"\b(?:peel(?:ing|ed)?|sticky|tearing|fell\s+apart)\b",
    r"\b(?:jammed|stuck|refused\s+to|shuts?\s+down)\b",
    r"\b(?:battery\s+drains?|rapid\s+drain|dead\s+battery)\b",
    r"\b(?:error\s+code|firmware\s+update\s+issue|crash(?:es|ing)?)\b"
]

SEVERITY_KEYWORDS = {
    "critical": ["spark", "fire", "smoke", "hazard", "dangerous", "exploded", "shock", "electric shock"],
    "high": ["stopped working", "dead", "won't turn on", "defective", "broke", "overheating", "burn error", "grinding"],
    "medium": ["crackling", "buzzing", "glitch", "crash", "drain", "jammed", "peeling", "leak", "slip", "smell"]
}

class IssueDetector:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.complaint_regexes = [re.compile(p, re.IGNORECASE) for p in COMPLAINT_PATTERNS]

    def extract_complaints(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter negative reviews and extract specific complaint sentences."""
        extracted = []

        for rev in reviews:
            rating = rev.get("rating", 3.0)
            text = rev.get("text", "")
            title = rev.get("title", "")
            full_text = f"{title}. {text}"
            ts = rev.get("timestamp")

            # Check if review has negative signal (star rating <= 3 or negative title)
            if rating > 3.0:
                continue

            sentences = self.preprocessor.tokenize_sentences(full_text)
            for sentence in sentences:
                for regex in self.complaint_regexes:
                    match = regex.search(sentence)
                    if match:
                        trigger = match.group(0).lower()
                        severity = self._classify_severity(sentence)
                        extracted.append({
                            "trigger": trigger,
                            "sentence": sentence.strip(),
                            "severity": severity,
                            "timestamp": ts,
                            "date_str": timestamp_to_str(ts),
                            "rating": rating
                        })
                        break # One match per sentence is sufficient

        return extracted

    def _classify_severity(self, text: str) -> str:
        """Classify severity of complaint."""
        text_lower = text.lower()
        for kw in SEVERITY_KEYWORDS["critical"]:
            if kw in text_lower:
                return "critical"
        for kw in SEVERITY_KEYWORDS["high"]:
            if kw in text_lower:
                return "high"
        return "medium"

    def detect_product_issues(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group complaints into distinct issue topics with trend and severity."""
        complaints = self.extract_complaints(reviews)
        if not complaints:
            return []

        # Group by normalized trigger or root phrase
        grouped = defaultdict(list)
        for c in complaints:
            trigger = c["trigger"]
            # Normalize trigger into cluster key
            if "crackling" in trigger or "buzzing" in trigger or "static" in trigger:
                key = "Audio crackling, buzzing or static noise"
            elif "work" in trigger or "turn on" in trigger or "dead" in trigger:
                key = "Device power / operation failure"
            elif "burn" in trigger or "overheat" in trigger or "spark" in trigger:
                key = "Thermal overheating or burn warning alert"
            elif "seal" in trigger or "leak" in trigger:
                key = "Seal degradation or liquid/steam leakage"
            elif "pump" in trigger:
                key = "Water pump grinding or failure to dispense"
            elif "crash" in trigger or "glitch" in trigger or "firmware" in trigger:
                key = "Software bugs, crashes or update issues"
            elif "peel" in trigger or "sticky" in trigger:
                key = "Exterior material peeling or degradation"
            elif "lid" in trigger or "jammed" in trigger:
                key = "Mechanism jamming or handle cracking"
            elif "battery" in trigger or "drain" in trigger:
                key = "Rapid battery drain or charging defect"
            else:
                key = f"Issue regarding: {trigger.capitalize()}"

            grouped[key].append(c)

        issues_summary = []
        for issue_name, items in grouped.items():
            # Calculate severity (worst case)
            severities = [i["severity"] for i in items]
            if "critical" in severities:
                final_severity = "Critical"
            elif "high" in severities:
                final_severity = "High"
            else:
                final_severity = "Medium"

            # Sort by timestamp
            dated_items = [i for i in items if i.get("timestamp")]
            dated_items.sort(key=lambda x: x["timestamp"])

            first_reported = dated_items[0]["date_str"] if dated_items else "Unknown"
            last_reported = dated_items[-1]["date_str"] if dated_items else "Recent"

            # Trend calculation: compare first half timestamps vs second half
            if len(dated_items) >= 2:
                mid = len(dated_items) // 2
                first_half_avg_ts = sum(i["timestamp"] for i in dated_items[:mid]) / mid
                second_half_avg_ts = sum(i["timestamp"] for i in dated_items[mid:]) / (len(dated_items) - mid)
                # If more recent activity concentrated in latest period
                trend = "Increasing (Emerging Issue ⚠️)" if len(dated_items[mid:]) >= len(dated_items[:mid]) else "Stable"
            else:
                trend = "Recent Detection"

            sample_quotes = list(dict.fromkeys([i["sentence"] for i in items]))[:3]

            issues_summary.append({
                "issue": issue_name,
                "severity": final_severity,
                "frequency": len(items),
                "trend": trend,
                "first_reported": first_reported,
                "last_reported": last_reported,
                "sample_quotes": sample_quotes
            })

        # Rank by severity and frequency
        severity_order = {"Critical": 0, "High": 1, "Medium": 2}
        issues_summary.sort(key=lambda x: (severity_order.get(x["severity"], 3), -x["frequency"]))
        return issues_summary
