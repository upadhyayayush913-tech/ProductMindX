"""
Text Preprocessor for Amazon Reviews and Metadata.
Includes:
- HTML / URL removal
- Contraction expansion
- Tokenization & Lemmatization
- Stopword filtering (configurable)
- Sentence splitting for aspect & issue mining
"""

import re
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from utils.logger import get_logger

logger = get_logger("preprocessor")

# Ensure required NLTK resources are available
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception as e:
        logger.debug(f"NLTK download notice for {resource}: {e}")

CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "can't": "cannot",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
}

class TextPreprocessor:
    def __init__(self, remove_stopwords: bool = False):
        self.lemmatizer = WordNetLemmatizer()
        self.remove_stopwords = remove_stopwords
        try:
            self.stop_words = set(stopwords.words("english"))
            # Keep sentiment-bearing negation words in stopwords
            self.stop_words -= {"no", "not", "nor", "neither", "never", "very", "but", "against"}
        except Exception:
            self.stop_words = set()

    def expand_contractions(self, text: str) -> str:
        """Expand English contractions."""
        lower_text = text.lower()
        for contraction, expansion in CONTRACTIONS.items():
            pattern = re.compile(re.escape(contraction), re.IGNORECASE)
            text = pattern.sub(expansion, text)
        return text

    def clean_text(self, text: Optional[str]) -> str:
        """Normalize whitespace, remove URLs, HTML and special artifacts."""
        if not text or not isinstance(text, str):
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Expand contractions
        text = self.expand_contractions(text)
        # Replace non-standard apostrophes and quotes
        text = text.replace("’", "'").replace("`", "'").replace("“", '"').replace("”", '"')
        # Remove excessive punctuation repetitions like !!!! or ????
        text = re.sub(r"([!?,.]){2,}", r"\1", text)
        # Clean extra whitespaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences for fine-grained sentiment/issue analysis."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        try:
            return sent_tokenize(cleaned)
        except Exception:
            return [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]

    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words, optionally lemmatize and filter stopwords."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        try:
            tokens = word_tokenize(cleaned.lower())
        except Exception:
            tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", cleaned.lower())

        result = []
        for token in tokens:
            # Keep alphanumeric words
            if not token.isalnum():
                continue
            if self.remove_stopwords and token in self.stop_words:
                continue
            try:
                lemma = self.lemmatizer.lemmatize(token)
            except Exception:
                lemma = token
            result.append(lemma)

        return result
