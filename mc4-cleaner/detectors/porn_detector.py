"""
detectors/porn_detector.py

Explicit / pornographic content detector for Bahasa Melayu and English.

Detection layers:
  1. Fast regex keyword matching (CPU, vectorised)
  2. Optional ML model scoring via HuggingFace (CUDA-aware)
     - Uses 'unitary/toxic-bert' or 'valurank/distilroberta-base-offensive-value'
       as fallback when a dedicated NSFW model is unavailable.
  3. Hash-based near-duplicate dedup via MinHash (datasketch)
"""

from __future__ import annotations

import re
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword loader
# ---------------------------------------------------------------------------
KEYWORD_DIR = Path(__file__).resolve().parent.parent / "keywords"


def _load_patterns(filename: str) -> list[re.Pattern]:
    """Load regex patterns from a keyword file (skip blank lines / comments)."""
    path = KEYWORD_DIR / filename
    patterns: list[re.Pattern] = []
    if not path.exists():
        logger.warning("Keyword file not found: %s", path)
        return patterns
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                patterns.append(re.compile(line, re.IGNORECASE | re.UNICODE))
            except re.error as exc:
                logger.debug("Bad regex pattern '%s': %s", line, exc)
    return patterns


_PORN_EN_PATTERNS: list[re.Pattern] = []
_PORN_BM_PATTERNS: list[re.Pattern] = []


def _ensure_patterns_loaded() -> None:
    global _PORN_EN_PATTERNS, _PORN_BM_PATTERNS
    if not _PORN_EN_PATTERNS:
        _PORN_EN_PATTERNS = _load_patterns("porn_en.txt")
    if not _PORN_BM_PATTERNS:
        _PORN_BM_PATTERNS = _load_patterns("porn_bm.txt")


# ---------------------------------------------------------------------------
# Optional ML model (NSFW / toxicity)
# ---------------------------------------------------------------------------
_ml_pipeline = None
_ml_available = False


def _try_load_ml_model(device: str = "cpu") -> None:
    """Attempt to load an NSFW/toxicity model from HuggingFace Transformers."""
    global _ml_pipeline, _ml_available
    if _ml_pipeline is not None:
        return
    # Try GPU first; fall back gracefully
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore

        model_name = os.environ.get(
            "NSFW_MODEL", "unitary/unbiased-toxic-roberta"
        )
        logger.info("Loading ML model '%s' on device '%s'…", model_name, device)
        _ml_pipeline = hf_pipeline(
            "text-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
            truncation=True,
            max_length=512,
            top_k=None,
        )
        _ml_available = True
        logger.info("ML model loaded successfully.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "ML model not available (%s). Falling back to keyword-only detection.", exc
        )
        _ml_available = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PornDetectionResult:
    """Result of a single text inspection."""

    __slots__ = (
        "is_explicit",
        "language",
        "matched_keywords",
        "ml_score",
        "ml_label",
    )

    def __init__(
        self,
        is_explicit: bool,
        language: str,
        matched_keywords: list[str],
        ml_score: Optional[float] = None,
        ml_label: Optional[str] = None,
    ):
        self.is_explicit = is_explicit
        self.language = language
        self.matched_keywords = matched_keywords
        self.ml_score = ml_score
        self.ml_label = ml_label

    def to_dict(self) -> dict:
        return {
            "is_explicit": self.is_explicit,
            "language": self.language,
            "matched_keywords": self.matched_keywords,
            "ml_score": self.ml_score,
            "ml_label": self.ml_label,
        }

    def __repr__(self) -> str:
        return (
            f"PornDetectionResult(explicit={self.is_explicit}, "
            f"lang={self.language}, "
            f"keywords={self.matched_keywords[:3]}, "
            f"ml_score={self.ml_score})"
        )


class PornDetector:
    """
    Two-stage explicit content detector.

    Parameters
    ----------
    use_ml : bool
        Enable ML-based rescoring (requires transformers + GPU/CPU).
    device : str
        'cuda' or 'cpu'.
    ml_threshold : float
        Minimum toxicity/NSFW score to flag via ML (0–1).
    keyword_threshold : int
        Minimum number of matching keywords to flag via regex alone.
    """

    def __init__(
        self,
        use_ml: bool = False,
        device: str = "cpu",
        ml_threshold: float = 0.7,
        keyword_threshold: int = 1,
    ):
        self.use_ml = use_ml
        self.device = device
        self.ml_threshold = ml_threshold
        self.keyword_threshold = keyword_threshold

        _ensure_patterns_loaded()
        if use_ml:
            _try_load_ml_model(device)

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def _keyword_check(self, text: str) -> tuple[bool, str, list[str]]:
        """
        Run keyword scan across both BM and EN pattern sets.

        Returns (flagged, language_detected, matched_patterns).
        """
        matched: list[str] = []
        language = "unknown"

        for pat in _PORN_EN_PATTERNS:
            m = pat.search(text)
            if m:
                matched.append(m.group(0))

        bm_matched: list[str] = []
        for pat in _PORN_BM_PATTERNS:
            m = pat.search(text)
            if m:
                bm_matched.append(m.group(0))

        if matched and bm_matched:
            language = "both"
            matched.extend(bm_matched)
        elif matched:
            language = "en"
        elif bm_matched:
            language = "bm"
            matched = bm_matched

        flagged = len(matched) >= self.keyword_threshold
        return flagged, language, matched

    def _ml_check(self, text: str) -> tuple[Optional[float], Optional[str]]:
        """Run ML inference if available."""
        if not _ml_available or _ml_pipeline is None:
            return None, None
        try:
            results = _ml_pipeline(text[:512])
            # results is a list of dicts [{"label": ..., "score": ...}]
            flat = results[0] if isinstance(results[0], dict) else results[0][0]
            # Find the highest toxicity/explicit score
            if isinstance(results[0], list):
                best = max(results[0], key=lambda x: x["score"])
            else:
                best = flat
            label: str = best["label"].upper()
            score: float = best["score"]
            toxic_labels = {"TOXIC", "SEVERE_TOXIC", "OBSCENE", "THREAT", "INSULT", "IDENTITY_HATE", "LABEL_1"}
            if label in toxic_labels:
                return score, label
            return score, label
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("ML inference error: %s", exc)
            return None, None

    def detect(self, text: str) -> PornDetectionResult:
        """
        Detect explicit content in *text*.

        Returns a :class:`PornDetectionResult`.
        """
        # Stage 1: keyword
        kw_flagged, language, matched = self._keyword_check(text)

        # Stage 2: ML
        ml_score, ml_label = None, None
        if self.use_ml:
            ml_score, ml_label = self._ml_check(text)

        # Decision logic
        ml_flagged = False
        if ml_score is not None:
            toxic_labels = {
                "TOXIC", "SEVERE_TOXIC", "OBSCENE",
                "THREAT", "INSULT", "IDENTITY_HATE", "LABEL_1",
            }
            if ml_label in (toxic_labels) and ml_score >= self.ml_threshold:
                ml_flagged = True

        is_explicit = kw_flagged or ml_flagged

        return PornDetectionResult(
            is_explicit=is_explicit,
            language=language,
            matched_keywords=matched,
            ml_score=ml_score,
            ml_label=ml_label,
        )

    def detect_batch(self, texts: list[str]) -> list[PornDetectionResult]:
        """Batch detect for a list of texts (vectorised keyword scan)."""
        return [self.detect(t) for t in texts]
