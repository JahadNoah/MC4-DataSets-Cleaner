"""
reporter.py

Flagged-content reporter for the MC4 cleaning pipeline.

Outputs:
  - Detailed JSONL report of every flagged record (line number, issues, evidence)
  - Summary CSV for quick analysis
  - Console-friendly coloured preview of sensitive lines

Usage:
    reporter = Reporter(output_dir="output/")
    reporter.report(line_no=42, text="...", porn_result=..., three_r_result=...)
    reporter.finalize()
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Optional colour support
try:
    from colorama import Fore, Style, init as colorama_init  # type: ignore
    colorama_init(autoreset=True)
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

def _red(s: str) -> str:
    return f"{Fore.RED}{s}{Style.RESET_ALL}" if _HAS_COLOR else s


def _yellow(s: str) -> str:
    return f"{Fore.YELLOW}{s}{Style.RESET_ALL}" if _HAS_COLOR else s


def _cyan(s: str) -> str:
    return f"{Fore.CYAN}{s}{Style.RESET_ALL}" if _HAS_COLOR else s


def _green(s: str) -> str:
    return f"{Fore.GREEN}{s}{Style.RESET_ALL}" if _HAS_COLOR else s


def _bold(s: str) -> str:
    return f"{Style.BRIGHT}{s}{Style.RESET_ALL}" if _HAS_COLOR else s


# ---------------------------------------------------------------------------
# FlaggedRecord
# ---------------------------------------------------------------------------

@dataclass
class FlaggedRecord:
    line_no: int
    dataset_id: str  # e.g. shard filename or split name
    text: str
    is_porn: bool
    porn_language: str
    porn_keywords: list[str]
    porn_ml_score: Optional[float]
    is_sensitive_3r: bool
    three_r_categories: list[str]
    race_keywords: list[str]
    religion_keywords: list[str]
    royalty_keywords: list[str]
    three_r_ml_scores: dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "line_no": self.line_no,
            "dataset_id": self.dataset_id,
            "timestamp": self.timestamp,
            "is_porn": self.is_porn,
            "porn_language": self.porn_language,
            "porn_keywords": self.porn_keywords,
            "porn_ml_score": self.porn_ml_score,
            "is_sensitive_3r": self.is_sensitive_3r,
            "three_r_categories": self.three_r_categories,
            "race_keywords": self.race_keywords,
            "religion_keywords": self.religion_keywords,
            "royalty_keywords": self.royalty_keywords,
            "three_r_ml_scores": self.three_r_ml_scores,
            "text_preview": self.text[:300],
        }


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class Reporter:
    """
    Collects flagged records and writes them to:
      - ``flagged.jsonl``   — full details for every flagged line
      - ``flagged_summary.csv`` — one row per record, columnar format
      - stdout / log       — real-time preview during scanning
    """

    CSV_FIELDS = [
        "line_no",
        "dataset_id",
        "timestamp",
        "is_porn",
        "porn_language",
        "porn_keywords_sample",
        "porn_ml_score",
        "is_sensitive_3r",
        "three_r_categories",
        "race_keywords_sample",
        "religion_keywords_sample",
        "royalty_keywords_sample",
        "text_preview",
    ]

    def __init__(
        self,
        output_dir: str | Path = "output",
        verbose: bool = True,
        max_preview_chars: int = 200,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.max_preview = max_preview_chars

        self._records: list[FlaggedRecord] = []

        # Open files for streaming writes
        jsonl_path = self.output_dir / "flagged.jsonl"
        csv_path = self.output_dir / "flagged_summary.csv"

        self._jsonl_fh = open(jsonl_path, "w", encoding="utf-8")  # noqa: WPS515
        self._csv_fh = open(csv_path, "w", encoding="utf-8", newline="")  # noqa: WPS515
        self._csv_writer = csv.DictWriter(
            self._csv_fh, fieldnames=self.CSV_FIELDS
        )
        self._csv_writer.writeheader()

        # Counters
        self.total_porn = 0
        self.total_race = 0
        self.total_religion = 0
        self.total_royalty = 0
        self.total_flagged = 0

        logger.info("Reporter initialised. Writing to %s", self.output_dir)

    # ------------------------------------------------------------------

    def report(
        self,
        line_no: int,
        text: str,
        dataset_id: str = "unknown",
        porn_result=None,
        three_r_result=None,
    ) -> Optional[FlaggedRecord]:
        """
        Evaluate results and record if flagged.

        Parameters
        ----------
        line_no : int
            1-based line index in the dataset.
        text : str
            Raw document text.
        dataset_id : str
            Shard or file identifier for traceability.
        porn_result : PornDetectionResult | None
        three_r_result : ThreeRResult | None
        """
        is_porn = porn_result.is_explicit if porn_result else False
        is_3r = three_r_result.is_sensitive if three_r_result else False

        if not (is_porn or is_3r):
            return None

        record = FlaggedRecord(
            line_no=line_no,
            dataset_id=dataset_id,
            text=text,
            is_porn=is_porn,
            porn_language=getattr(porn_result, "language", ""),
            porn_keywords=getattr(porn_result, "matched_keywords", []),
            porn_ml_score=getattr(porn_result, "ml_score", None),
            is_sensitive_3r=is_3r,
            three_r_categories=three_r_result.categories if three_r_result else [],
            race_keywords=three_r_result.race_matches if three_r_result else [],
            religion_keywords=three_r_result.religion_matches if three_r_result else [],
            royalty_keywords=three_r_result.royalty_matches if three_r_result else [],
            three_r_ml_scores=three_r_result.ml_scores if three_r_result else {},
        )

        self._records.append(record)
        self.total_flagged += 1
        if is_porn:
            self.total_porn += 1
        if three_r_result:
            if three_r_result.race_flagged:
                self.total_race += 1
            if three_r_result.religion_flagged:
                self.total_religion += 1
            if three_r_result.royalty_flagged:
                self.total_royalty += 1

        # Streaming write
        self._jsonl_fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._jsonl_fh.flush()

        csv_row = {
            "line_no": record.line_no,
            "dataset_id": record.dataset_id,
            "timestamp": record.timestamp,
            "is_porn": record.is_porn,
            "porn_language": record.porn_language,
            "porn_keywords_sample": "|".join(record.porn_keywords[:5]),
            "porn_ml_score": record.porn_ml_score,
            "is_sensitive_3r": record.is_sensitive_3r,
            "three_r_categories": "|".join(record.three_r_categories),
            "race_keywords_sample": "|".join(record.race_keywords[:5]),
            "religion_keywords_sample": "|".join(record.religion_keywords[:5]),
            "royalty_keywords_sample": "|".join(record.royalty_keywords[:5]),
            "text_preview": text[:self.max_preview].replace("\n", " "),
        }
        self._csv_writer.writerow(csv_row)
        self._csv_fh.flush()

        if self.verbose:
            self._print_inline(record, text)

        return record

    # ------------------------------------------------------------------

    def _print_inline(self, record: FlaggedRecord, text: str) -> None:
        """Print a coloured inline summary to stdout."""
        issues: list[str] = []
        if record.is_porn:
            tag = f"PORN[{record.porn_language.upper()}]"
            issues.append(_red(tag))
        for cat in record.three_r_categories:
            issues.append(_yellow(f"3R:{cat}"))

        prefix = _bold(f"[Line {record.line_no:>8}]")
        issue_str = " ".join(issues)
        preview = text[:self.max_preview].replace("\n", " ↵ ")
        print(f"{prefix} {issue_str}  ➜  {_cyan(preview)}", flush=True)

        # Print evidence keywords
        if record.is_porn and record.porn_keywords:
            kws = ", ".join(f"«{k}»" for k in record.porn_keywords[:6])
            print(f"          Porn keywords   : {_red(kws)}")
        if record.race_keywords:
            kws = ", ".join(f"«{k}»" for k in record.race_keywords[:6])
            print(f"          Race keywords   : {_yellow(kws)}")
        if record.religion_keywords:
            kws = ", ".join(f"«{k}»" for k in record.religion_keywords[:6])
            print(f"          Religion kws    : {_yellow(kws)}")
        if record.royalty_keywords:
            kws = ", ".join(f"«{k}»" for k in record.royalty_keywords[:6])
            print(f"          Royalty kws     : {_yellow(kws)}")
        if record.porn_ml_score is not None:
            print(
                f"          ML score        : {record.porn_ml_score:.3f} "
                f"({record.porn_language})"
            )
        print()  # blank line separator

    # ------------------------------------------------------------------

    def finalize(self) -> dict:
        """Flush buffers, close files, print final summary stats."""
        self._jsonl_fh.close()
        self._csv_fh.close()

        summary = {
            "total_flagged": self.total_flagged,
            "total_porn": self.total_porn,
            "total_3r_race": self.total_race,
            "total_3r_religion": self.total_religion,
            "total_3r_royalty": self.total_royalty,
        }

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        print("\n" + "=" * 60)
        print(_bold("PIPELINE SUMMARY"))
        print("=" * 60)
        print(f"  Total flagged records : {_red(str(self.total_flagged))}")
        print(f"  Pornographic content  : {_red(str(self.total_porn))}")
        print(f"  3R - Race             : {_yellow(str(self.total_race))}")
        print(f"  3R - Religion         : {_yellow(str(self.total_religion))}")
        print(f"  3R - Royalty          : {_yellow(str(self.total_royalty))}")
        print("=" * 60)
        print(f"\nOutput files in: {self.output_dir.resolve()}")
        print(f"  flagged.jsonl          (per-record full detail)")
        print(f"  flagged_summary.csv    (columnar summary)")
        print(f"  summary.json           (aggregate stats)")

        return summary
