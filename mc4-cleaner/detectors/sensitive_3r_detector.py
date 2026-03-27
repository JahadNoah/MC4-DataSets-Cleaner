"""
detectors/sensitive_3r_detector.py

Malaysia 3R Sensitive Issues Detector
--------------------------------------
Detects content that may violate Malaysia's Sedition Act 1948, Penal Code s.298/298A,
and related laws around three protected categories:

  R1 = Race    (Kaum / Ras)
  R2 = Religion (Agama)
  R3 = Royalty  (Raja-raja Melayu / Institusi Diraja)

Both Bahasa Melayu (BM) and English (EN) patterns are included.

Detection layers:
  1. Regex keyword matching from keyword files
  2. Context-aware rule boosting (co-occurrence of target + insult words)
  3. Optional ML classifier (zero-shot via HuggingFace) for ambiguous text
"""

from __future__ import annotations

import re
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KEYWORD_DIR = Path(__file__).resolve().parent.parent / "keywords"

# ---------------------------------------------------------------------------
# Insult/negative amplifier words (applied as context boosting)
# These are combined with entity words to catch indirect insults.
# ---------------------------------------------------------------------------
_INSULT_AMPLIFIERS = re.compile(
    r"\b(hina|celaka|bodoh|gila|jahat|zalim|busuk|korup|corrupt|evil|stupid|"
    r"idiot|trash|garbage|fuck|babi|sial|bangsat|penipu|liar|kafir|sesat|"
    r"destroy|kill|murder|bunuh|bakar|hapus|usir|layang|tendang)\b",
    re.IGNORECASE | re.UNICODE,
)

# ---------------------------------------------------------------------------
# Safe context indicators — presence of these suggests the text is
# news / academic / historical / educational rather than hateful.
# Used by the autonomous verification layer to rescue false positives.
# ---------------------------------------------------------------------------
_SAFE_CONTEXT = re.compile(
    r"\b("
    # News / reporting
    r"berita|laporan|dilaporkan|menurut|kata beliau|sumber|agensi|"
    r"bernama|astro awani|the star|nst|utusan|harian metro|"
    r"wartawan|sidang media|kenyataan akhbar|press statement|"
    r"reported|according to|news|journalist|press conference|"
    # Academic / research
    r"kajian|penyelidikan|jurnal|universiti|profesor|dr\.|"
    r"thesis|dissertation|research|study|academic|scholar|findings|"
    r"analisis|analysis|data menunjukkan|data shows|"
    # Historical / educational
    r"sejarah|bersejarah|historical|history|peringatan|"
    r"commemorate|memorial|muzium|museum|dokumentari|documentary|"
    r"pendidikan|education|kurikulum|curriculum|"
    # Legal / policy discussion
    r"perlembagaan|constitution|undang.?undang|legislation|"
    r"parlimen|parliament|dewan rakyat|dewan negara|"
    r"mahkamah|court|tribunal|"
    # Positive / harmony context
    r"perpaduan|keharmonian|muhibbah|toleransi|"
    r"unity|harmony|tolerance|diversity|inclusiv|"
    r"bersatu|bersama.?sama|kerjasama|cooperation|"
    r"perayaan|celebration|sambutan|meraikan|"
    r"menghormati|respect|hormati"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

# ---------------------------------------------------------------------------
# Hard patterns — explicit slur bigrams that are ALWAYS hateful regardless
# of context. These bypass the autonomous verification.
# Pattern strings are matched against keyword file patterns.
# ---------------------------------------------------------------------------
_HARD_RACE_SLURS = re.compile(
    r"\b("
    # Ethnic slur + insult combos (always hateful, no innocent usage)
    r"melayu bodoh|melayu malas|melayu celaka|melayu babi|melayu hapus|melayu hina|"
    r"orang melayu babi|bumi tak guna|bumiputera bodoh|bumiputera malas|bumi celaka|"
    r"cina babi|cina celaka|cina balik china|cina pendatang|cina komunis|cina bodoh|cina hina|"
    r"chinese go back china|chinese pendatang|"
    r"keling babi|keling celaka|keling bodoh|indian celaka|indian pariah|mamak celaka|"
    r"indian balik india|"
    r"dayak babi|dayak bodoh|orang asli bodoh|orang asli hina|kadazan bodoh|iban babi|"
    # Direct violence
    r"kill all chinese|kill all malays|kill all indians|"
    r"deport all chinese|deport all indians|deport all malays|"
    r"bunuh kaum|hapus kaum|usir cina|usir india|usir pendatang|"
    # Ethnic hierarchy insults
    r"kaum hina|kaumnya rendah|kaum kafir|ethnik rendah|"
    r"bangsa.*lebih hina|ras.*lebih inferior"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)

# ---------------------------------------------------------------------------
# Pattern loader
# ---------------------------------------------------------------------------

def _load_patterns(filename: str) -> list[re.Pattern]:
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


# Lazy-loaded pattern sets
_RACE_PATTERNS: list[re.Pattern] = []
_RELIGION_PATTERNS: list[re.Pattern] = []
_ROYALTY_PATTERNS: list[re.Pattern] = []


def _ensure_loaded() -> None:
    global _RACE_PATTERNS, _RELIGION_PATTERNS, _ROYALTY_PATTERNS
    if not _RACE_PATTERNS:
        _RACE_PATTERNS = _load_patterns("3r_race.txt")
    if not _RELIGION_PATTERNS:
        _RELIGION_PATTERNS = _load_patterns("3r_religion.txt")
    if not _ROYALTY_PATTERNS:
        _ROYALTY_PATTERNS = _load_patterns("3r_royalty.txt")


# ---------------------------------------------------------------------------
# Optional zero-shot ML classifier
# ---------------------------------------------------------------------------
_zs_pipeline = None
_zs_available = False
_ZS_LABELS = ["racial hatred", "religious hatred", "insult to royalty", "safe content"]


def _try_load_zs_model(device: str = "cpu") -> None:
    global _zs_pipeline, _zs_available
    if _zs_pipeline is not None:
        return
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore

        model_name = os.environ.get("ZS_MODEL", "facebook/bart-large-mnli")
        logger.info(
            "Loading zero-shot model '%s' on device '%s'…", model_name, device
        )
        _zs_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )
        _zs_available = True
        logger.info("Zero-shot model loaded.")
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Zero-shot ML model not available (%s). Using keyword-only mode.", exc
        )
        _zs_available = False


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ThreeRResult:
    """Detection result for the 3R check."""

    is_sensitive: bool = False

    # Per-category flags
    race_flagged: bool = False
    religion_flagged: bool = False
    royalty_flagged: bool = False

    # Matched keyword evidence per category
    race_matches: list[str] = field(default_factory=list)
    religion_matches: list[str] = field(default_factory=list)
    royalty_matches: list[str] = field(default_factory=list)

    # ML scores (if enabled)
    ml_scores: dict[str, float] = field(default_factory=dict)

    # Autonomous verification metadata
    race_verified: bool | None = None   # None = not checked, True = confirmed, False = rescued
    verification_reason: str = ""       # Why verified/rescued

    # Human-readable category list
    @property
    def categories(self) -> list[str]:
        cats: list[str] = []
        if self.race_flagged:
            cats.append("RACE")
        if self.religion_flagged:
            cats.append("RELIGION")
        if self.royalty_flagged:
            cats.append("ROYALTY")
        return cats

    def to_dict(self) -> dict:
        return {
            "is_sensitive": self.is_sensitive,
            "categories": self.categories,
            "race_flagged": self.race_flagged,
            "religion_flagged": self.religion_flagged,
            "royalty_flagged": self.royalty_flagged,
            "race_matches": self.race_matches,
            "religion_matches": self.religion_matches,
            "royalty_matches": self.royalty_matches,
            "ml_scores": self.ml_scores,
            "race_verified": self.race_verified,
            "verification_reason": self.verification_reason,
        }


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class ThreeRDetector:
    """
    Malaysia 3R Sensitive Issues Detector.

    Parameters
    ----------
    use_ml : bool
        Enable zero-shot ML classification for improved recall on indirect/implicit
        sensitive language.
    device : str
        'cuda' or 'cpu'.
    ml_threshold : float
        Minimum zero-shot confidence score to flag via ML.
    keyword_threshold : int
        Minimum number of regex hits to consider the text flagged.
    context_window : int
        Characters around a matched keyword to look for amplifier words.
    """

    def __init__(
        self,
        use_ml: bool = False,
        device: str = "cpu",
        ml_threshold: float = 0.65,
        keyword_threshold: int = 1,
        context_window: int = 80,
    ):
        self.use_ml = use_ml
        self.device = device
        self.ml_threshold = ml_threshold
        self.keyword_threshold = keyword_threshold
        self.context_window = context_window

        _ensure_loaded()
        if use_ml:
            _try_load_zs_model(device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan(text: str, patterns: list[re.Pattern]) -> list[str]:
        matches: list[str] = []
        for pat in patterns:
            m = pat.search(text)
            if m:
                matches.append(m.group(0))
        return matches

    def _context_boost(self, text: str, matches: list[str]) -> bool:
        """
        Return True if any matched keyword appears within *context_window*
        characters of an insult amplifier word (increases confidence for
        indirect insults not covered by explicit compound patterns).
        """
        for match in matches:
            idx = text.lower().find(match.lower())
            if idx == -1:
                continue
            start = max(0, idx - self.context_window)
            end = min(len(text), idx + len(match) + self.context_window)
            snippet = text[start:end]
            if _INSULT_AMPLIFIERS.search(snippet):
                return True
        return False

    def _verify_race_flag(self, text: str, race_matches: list[str]) -> tuple[bool, str]:
        """
        Autonomous verification for race flags.

        Determines whether a race-flagged text is truly 'menghina' (insulting)
        or is a false positive from neutral/news/academic context.

        Returns (is_confirmed, reason).

        Logic:
          1. Hard slur match → always confirm (no innocent usage)
          2. Check safe context density vs insult density
          3. If ML available, use it as tiebreaker for ambiguous cases
        """
        text_lower = text.lower()

        # --- Layer 1: Hard slur check (unambiguous hatred) ---
        if _HARD_RACE_SLURS.search(text_lower):
            return True, "hard_slur"

        # --- Layer 2: Safe context analysis ---
        safe_hits = _SAFE_CONTEXT.findall(text_lower)
        insult_hits = _INSULT_AMPLIFIERS.findall(text_lower)
        safe_count = len(safe_hits)
        insult_count = len(insult_hits)

        # Strong safe context with few/no insults → likely false positive
        if safe_count >= 2 and insult_count <= 1:
            logger.debug(
                "Race flag RESCUED (safe_context=%d, insults=%d): %.80s…",
                safe_count, insult_count, text,
            )
            return False, f"safe_context({safe_count}≥2,insults={insult_count}≤1)"

        # If safe context present and outnumbers insults, also rescue
        if safe_count > 0 and safe_count > insult_count:
            logger.debug(
                "Race flag RESCUED (safe>insult: %d>%d): %.80s…",
                safe_count, insult_count, text,
            )
            return False, f"safe_dominates({safe_count}>{insult_count})"

        # --- Layer 3: Targeted insult check ---
        # Verify the insult amplifier is actually directed AT a racial entity,
        # not just nearby by coincidence. Check tighter proximity (40 chars).
        race_entities = re.compile(
            r"\b(melayu|cina|india|iban|kadazan|dayak|orang asli|bumi|"
            r"bumiputera|pendatang|chinese|malay|indian|kaum|bangsa)\b",
            re.IGNORECASE,
        )
        entity_hits = list(race_entities.finditer(text_lower))
        is_targeted = False
        for entity_match in entity_hits:
            e_start = entity_match.start()
            e_end = entity_match.end()
            # Tight window: 40 chars before and after the entity
            window_start = max(0, e_start - 40)
            window_end = min(len(text_lower), e_end + 40)
            window = text_lower[window_start:window_end]
            if _INSULT_AMPLIFIERS.search(window):
                is_targeted = True
                break

        if not is_targeted and safe_count > 0:
            return False, "insult_not_targeted_at_entity"

        # --- Layer 4: ML tiebreaker (if available and ambiguous) ---
        if self.use_ml and _zs_available:
            ml_scores = self._ml_classify(text)
            race_score = ml_scores.get("racial hatred", 0)
            safe_score = ml_scores.get("safe content", 0)
            if safe_score > race_score and race_score < self.ml_threshold:
                logger.debug(
                    "Race flag RESCUED by ML (safe=%.2f > racial=%.2f): %.80s…",
                    safe_score, race_score, text,
                )
                return False, f"ml_safe({safe_score:.2f}>{race_score:.2f})"
            if race_score >= self.ml_threshold:
                return True, f"ml_confirmed(racial={race_score:.2f})"

        # Default: if we got here with matches, confirm the flag
        if is_targeted:
            return True, "targeted_insult"

        return True, "keyword_match"

    def _ml_classify(self, text: str) -> dict[str, float]:
        """Run zero-shot classification and return label→score dict."""
        if not _zs_available or _zs_pipeline is None:
            return {}
        try:
            result = _zs_pipeline(
                text[:1024],
                candidate_labels=_ZS_LABELS,
                multi_label=True,
            )
            return dict(zip(result["labels"], result["scores"]))
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Zero-shot ML error: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> ThreeRResult:
        """
        Detect 3R sensitive content in *text*.

        Returns a :class:`ThreeRResult`.
        """
        result = ThreeRResult()

        # --- Keyword scan -------------------------------------------------
        race_m = self._scan(text, _RACE_PATTERNS)
        religion_m = self._scan(text, _RELIGION_PATTERNS)
        royalty_m = self._scan(text, _ROYALTY_PATTERNS)

        result.race_matches = race_m
        result.religion_matches = religion_m
        result.royalty_matches = royalty_m

        # Primary flag: direct keyword hits
        result.race_flagged = len(race_m) >= self.keyword_threshold
        result.religion_flagged = len(religion_m) >= self.keyword_threshold
        result.royalty_flagged = len(royalty_m) >= self.keyword_threshold

        # Context boost: entity word + amplifier in proximity
        if not result.race_flagged:
            # Look for race entity words even without explicit slur
            race_entities = re.compile(
                r"\b(melayu|cina|india|iban|kadazan|dayak|orang asli|bumi|"
                r"bumiputera|pendatang|chinese|malay|indian|ethnic|racial|kaum|bangsa)\b",
                re.IGNORECASE,
            )
            race_entity_hits = race_entities.findall(text)
            if race_entity_hits and self._context_boost(text, race_entity_hits):
                result.race_flagged = True
                result.race_matches.extend(race_entity_hits[:3])

        if not result.religion_flagged:
            religion_entities = re.compile(
                r"\b(islam|muslim|kristian|christian|hindu|buddha|sikh|gereja|"
                r"masjid|surau|kuil|tokong|bible|quran|al-quran|bible|veda|agama)\b",
                re.IGNORECASE,
            )
            religion_entity_hits = religion_entities.findall(text)
            if religion_entity_hits and self._context_boost(text, religion_entity_hits):
                result.religion_flagged = True
                result.religion_matches.extend(religion_entity_hits[:3])

        if not result.royalty_flagged:
            royalty_entities = re.compile(
                r"\b(sultan|raja|agong|ydpa|diraja|monarchy|king|queen|putera|"
                r"puteri|tengku|tunku|duli|istana|palace)\b",
                re.IGNORECASE,
            )
            royalty_entity_hits = royalty_entities.findall(text)
            if royalty_entity_hits and self._context_boost(text, royalty_entity_hits):
                result.royalty_flagged = True
                result.royalty_matches.extend(royalty_entity_hits[:3])

        # --- ML classification (optional) ---------------------------------
        if self.use_ml:
            ml_scores = self._ml_classify(text)
            result.ml_scores = ml_scores
            if ml_scores.get("racial hatred", 0) >= self.ml_threshold:
                result.race_flagged = True
            if ml_scores.get("religious hatred", 0) >= self.ml_threshold:
                result.religion_flagged = True
            if ml_scores.get("insult to royalty", 0) >= self.ml_threshold:
                result.royalty_flagged = True

        # --- Autonomous verification for RACE flags -----------------------
        # Checks whether flagged race content truly "menghina" or is
        # a false positive from news / academic / historical context.
        if result.race_flagged:
            verified, reason = self._verify_race_flag(text, result.race_matches)
            result.race_verified = verified
            result.verification_reason = reason
            if not verified:
                result.race_flagged = False
                logger.info(
                    "RACE flag auto-rescued (%s): matches=%s text=%.100s…",
                    reason, result.race_matches, text,
                )

        result.is_sensitive = (
            result.race_flagged
            or result.religion_flagged
            or result.royalty_flagged
        )
        return result

    def detect_batch(self, texts: list[str]) -> list[ThreeRResult]:
        """Batch detect for a list of texts."""
        return [self.detect(t) for t in texts]
