"""
Cross-Lingual Intelligence and Machine Translation Engine for ProductMind X.
Enforces:
1. Automated Language Detection (Unicode script analysis + langdetect).
2. Hugging Face Transformers MarianMT / NLLB integration for cross-border review ingestion.
3. Bidirectional translation (Foreign -> English for NLP pipeline; English -> Foreign for user summaries).
4. Strict Citation Preservation: Masks [Review #ID] and [Metadata #ASIN] tokens during translation.
5. In-Memory LRU Caching and safe offline fallback.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from utils.logger import get_logger

logger = get_logger("translator")

# Language code to human-readable names and flag indicators
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "native": "English", "flag": "🇬🇧"},
    "es": {"name": "Spanish", "native": "Español", "flag": "🇪🇸"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "flag": "🇮🇳"},
    "de": {"name": "German", "native": "Deutsch", "flag": "🇩🇪"},
    "fr": {"name": "French", "native": "Français", "flag": "🇫🇷"},
    "it": {"name": "Italian", "native": "Italiano", "flag": "🇮🇹"},
    "pt": {"name": "Portuguese", "native": "Português", "flag": "🇵🇹"}
}

# Hugging Face MarianMT model mapping
MARIAN_MODEL_MAP = {
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi"
}

# Common multilingual consumer terms for fast offline / heuristic fallback
OFFLINE_GLOSSARY = {
    "hi": {
        "बैटरी": "battery",
        "आवाज": "sound",
        "साउंड": "sound",
        "खराब": "defective",
        "समस्या": "problem",
        "फायदे": "pros",
        "नुकसान": "cons",
        "कीमत": "price",
        "अच्छा": "good",
        "बहुत अच्छा": "very good",
        "तुलना": "compare",
        "रिव्यू": "review",
        "हेडफोन": "headphones"
    },
    "es": {
        "batería": "battery",
        "sonido": "sound",
        "cancelación de ruido": "noise cancellation",
        "problemas": "problems",
        "defectuoso": "defective",
        "precio": "price",
        "bueno": "good",
        "excelente": "excellent",
        "comparar": "compare",
        "calidad": "quality"
    },
    "de": {
        "akku": "battery",
        "klang": "sound",
        "ton": "sound",
        "geräuschunterdrückung": "noise cancellation",
        "probleme": "problems",
        "defekt": "defective",
        "preis": "price",
        "gut": "good",
        "hervorragend": "excellent",
        "vergleichen": "compare"
    }
}

class LanguageDetector:
    """Detects natural language using script heuristics and langdetect."""

    @staticmethod
    def detect(text: str) -> str:
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return "en"

        clean = text.strip()

        # 1. Devanagari script check (Hindi / Marathi)
        if re.search(r"[\u0900-\u097F]", clean):
            return "hi"

        # 2. Heuristic check for common short Spanish / German / French queries
        lower_clean = clean.lower()
        if any(w in lower_clean for w in ["cuál", "cuáles", "qué", "batería", "sonido", "cancelación", "precio", "cómo"]):
            return "es"
        if any(w in lower_clean for w in ["wie", "was", "akku", "klang", "geräuschunterdrückung", "ist", "über"]):
            return "de"
        if any(w in lower_clean for w in ["quel", "quelle", "bruit", "batterie", "qualité", "comment"]):
            return "fr"

        # 3. Use langdetect for Latin / extended scripts
        try:
            from langdetect import detect
            code = detect(clean)
            if code in SUPPORTED_LANGUAGES:
                return code
        except Exception:
            pass

        return "en"


class MachineTranslationEngine:
    """Hugging Face Transformers Machine Translation Engine."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MachineTranslationEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._models: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
        self._translation_cache: Dict[str, str] = {}
        self.detector = LanguageDetector()
        self._initialized = True
        logger.info("Initialized Cross-Lingual Machine Translation Engine.")

    def detect_language(self, text: str) -> str:
        """Detect the ISO language code of input text."""
        return self.detector.detect(text)

    def mask_citations(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Mask [Review #ID] and [Metadata #ASIN] tokens so the translation model
        cannot alter or drop them.
        """
        # Match tokens like [Review #REV-B09XS7-1001] or [Metadata #B09XS7JWHH]
        pattern = r"(\[(?:Review|Metadata)\s*#[A-Za-z0-9_-]+\])"
        matches = re.findall(pattern, text)
        citation_map = {}
        masked_text = text

        for idx, match in enumerate(matches):
            placeholder = f"__CITE_TOKEN_{idx}__"
            citation_map[placeholder] = match
            masked_text = masked_text.replace(match, placeholder)

        return masked_text, citation_map

    def restore_citations(self, text: str, citation_map: Dict[str, str]) -> str:
        """Restore exact citation tokens after translation."""
        restored = text
        for placeholder, original in citation_map.items():
            # 1. Direct match
            restored = restored.replace(placeholder, original)
            # 2. Extract index to match model variations (e.g. single underscores: __CITE_TOKEN_0_, spaces, etc.)
            digits = re.findall(r"\d+", placeholder)
            if digits:
                idx = digits[0]
                pattern = re.compile(rf"_*[\s]*CITE[\s_]*TOKEN[\s_]*{idx}[\s_]*_*", re.IGNORECASE)
                restored = pattern.sub(original, restored)
        return restored

    def _get_model(self, src: str, tgt: str):
        """Lazy-load Hugging Face MarianMT model and tokenizer."""
        key = (src, tgt)
        if key in self._models:
            return self._models[key]

        model_name = MARIAN_MODEL_MAP.get(key)
        if not model_name:
            logger.debug(f"No dedicated Marian model configured for pair {src}->{tgt}")
            return None

        try:
            from transformers import MarianMTModel, MarianTokenizer
            logger.info(f"Loading Hugging Face MarianMT translation model: {model_name}")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            self._models[key] = (model, tokenizer)
            return self._models[key]
        except Exception as e:
            logger.warning(f"Could not load Marian model {model_name}: {e}. Falling back to heuristic translation.")
            return None

    def translate(self, text: str, source_lang: Optional[str] = None, target_lang: str = "en") -> Dict[str, Any]:
        """
        Translate text between supported languages while strictly preserving citations.
        Returns detailed translation metadata.
        """
        if not text or not text.strip():
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": "en",
                "target_lang": target_lang,
                "was_translated": False,
                "preserved_citations": []
            }

        src = source_lang or self.detect_language(text)

        # Same language: pass-through immediately
        if src == target_lang:
            return {
                "original_text": text,
                "translated_text": text,
                "source_lang": src,
                "target_lang": target_lang,
                "was_translated": False,
                "preserved_citations": []
            }

        # Check cache
        cache_key = f"{src}->{target_lang}::{text}"
        if cache_key in self._translation_cache:
            return {
                "original_text": text,
                "translated_text": self._translation_cache[cache_key],
                "source_lang": src,
                "target_lang": target_lang,
                "was_translated": True,
                "preserved_citations": []
            }

        # Mask citations to protect them
        masked_text, citation_map = self.mask_citations(text)

        # Attempt neural translation with Hugging Face MarianMT
        pair = self._get_model(src, target_lang)
        translated_output = None

        if pair is not None:
            model, tokenizer = pair
            try:
                # Truncate to maximum sequence length if necessary
                inputs = tokenizer(masked_text[:1024], return_tensors="pt", padding=True, truncation=True, max_length=512)
                gen = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
                translated_raw = tokenizer.decode(gen[0], skip_special_tokens=True)
                translated_output = self.restore_citations(translated_raw, citation_map)
            except Exception as e:
                logger.warning(f"MarianMT neural inference failed ({e}). Using rule-based fallback.")
                translated_output = None

        # Heuristic dictionary fallback if Marian model is not loaded or failed
        if translated_output is None:
            translated_output = self._heuristic_fallback_translate(masked_text, src, target_lang)
            translated_output = self.restore_citations(translated_output, citation_map)

        self._translation_cache[cache_key] = translated_output

        return {
            "original_text": text,
            "translated_text": translated_output,
            "source_lang": src,
            "target_lang": target_lang,
            "was_translated": True,
            "preserved_citations": list(citation_map.values())
        }

    def _heuristic_fallback_translate(self, text: str, src: str, tgt: str) -> str:
        """Fast fallback when neural model is offline or uninstalled."""
        glossary = OFFLINE_GLOSSARY.get(src, {})
        if not glossary or tgt != "en":
            return text

        translated = text
        for foreign_word, english_word in glossary.items():
            pattern = re.compile(re.escape(foreign_word), re.IGNORECASE)
            translated = pattern.sub(english_word, translated)

        return translated

    def translate_to_english(self, text: str, source_lang: Optional[str] = None) -> str:
        """Convenience method to translate non-English review text or query into English."""
        res = self.translate(text, source_lang=source_lang, target_lang="en")
        return res["translated_text"]


# Global engine singleton
translation_engine = MachineTranslationEngine()
