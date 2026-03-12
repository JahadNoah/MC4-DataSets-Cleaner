"""
pipeline.py
===========
MC4 / Large Dataset Content Cleaning Pipeline
- Detects pornographic content (Bahasa Melayu + English)
- Detects Malaysia 3R sensitive issues (Race / Religion / Royalty)
- Reports flagged lines with evidence before cleaning
- Outputs cleaned dataset (flagged records removed)
- Supports CUDA (via HuggingFace transformers), multiprocessing, and streaming

Usage examples
--------------
# Basic scan (keyword-only, fast):
python pipeline.py --input mc4_ms.jsonl --output cleaned/

# With ML models + CUDA:
python pipeline.py --input mc4_ms.jsonl --output cleaned/ --use-ml --device cuda

# HuggingFace datasets streaming (mc4 directly):
python pipeline.py --hf-dataset mc4 --hf-subset ms --output cleaned/ --use-ml --device cuda

# Parallel processing (4 workers):
python pipeline.py --input mc4_ms.jsonl --output cleaned/ --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Module imports (local)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))

from detectors.porn_detector import PornDetector
from detectors.sensitive_3r_detector import ThreeRDetector
from reporter import Reporter

# ---------------------------------------------------------------------------
# Dataset reader helpers
# ---------------------------------------------------------------------------

def _iter_local_jsonl(path: str) -> Iterator[tuple[int, str, str]]:
    """
    Yield (line_no, dataset_id, text) from a local JSONL file.
    Supports plain text lines (one doc per line) or JSON objects with a
    'text' field (standard MC4 format).
    """
    dataset_id = Path(path).name
    with open(path, encoding="utf-8") as fh:
        for idx, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            # Try JSON first (MC4 format: {"text": "...", "url": "...", ...})
            if raw.startswith("{"):
                try:
                    obj = json.loads(raw)
                    text = obj.get("text", raw)
                except json.JSONDecodeError:
                    text = raw
            else:
                text = raw
            yield idx, dataset_id, text


def _iter_hf_dataset(
    dataset_name: str,
    subset: str,
    split: str = "train",
    streaming: bool = True,
) -> Iterator[tuple[int, str, str]]:
    """
    Stream a HuggingFace dataset (e.g. mc4, ms subset).
    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        logger.error(
            "HuggingFace `datasets` package not found. "
            "Install with: pip install datasets"
        )
        sys.exit(1)

    logger.info(
        "Loading HuggingFace dataset '%s' subset='%s' split='%s' streaming=%s",
        dataset_name, subset, split, streaming,
    )
    ds = load_dataset(dataset_name, subset, split=split, streaming=streaming, trust_remote_code=True)
    dataset_id = f"{dataset_name}/{subset}/{split}"
    for idx, record in enumerate(ds, start=1):
        text = record.get("text", "")
        if text:
            yield idx, dataset_id, text


# ---------------------------------------------------------------------------
# Worker process function (multiprocessing)
# ---------------------------------------------------------------------------

def _worker_init(porn_cfg: dict, three_r_cfg: dict) -> None:
    """Initialise detectors in each worker process."""
    global _worker_porn, _worker_3r
    _worker_porn = PornDetector(**porn_cfg)
    _worker_3r = ThreeRDetector(**three_r_cfg)


def _worker_process(item: tuple[int, str, str]) -> dict:
    """Process a single (line_no, dataset_id, text) item in a worker."""
    line_no, dataset_id, text = item
    porn_result = _worker_porn.detect(text)
    three_r_result = _worker_3r.detect(text)
    return {
        "line_no": line_no,
        "dataset_id": dataset_id,
        "text": text,
        "porn": porn_result.to_dict(),
        "three_r": three_r_result.to_dict(),
        "keep": not (porn_result.is_explicit or three_r_result.is_sensitive),
    }


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class CleaningPipeline:
    """
    End-to-end MC4 cleaning pipeline.

    Parameters
    ----------
    output_dir : str
        Directory for cleaned output and reports.
    use_ml : bool
        Enable ML-based detection (requires transformers).
    device : str
        'cuda' or 'cpu'. Auto-detected if not specified.
    workers : int
        Number of parallel worker processes (0 = main process only).
    batch_size : int
        Records per batch in parallel mode.
    verbose : bool
        Print flagged records to stdout in real time.
    skip_porn : bool
        Disable pornographic content detection.
    skip_3r : bool
        Disable 3R sensitive content detection.
    write_cleaned : bool
        Write a cleaned (filtered) output JSONL file.
    """

    def __init__(
        self,
        output_dir: str = "output",
        use_ml: bool = False,
        device: Optional[str] = None,
        workers: int = 0,
        batch_size: int = 256,
        verbose: bool = True,
        skip_porn: bool = False,
        skip_3r: bool = False,
        write_cleaned: bool = True,
        ml_threshold: float = 0.70,
        keyword_threshold: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect CUDA
        if device is None:
            device = _auto_detect_device()
        self.device = device

        self.use_ml = use_ml
        self.workers = workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.skip_porn = skip_porn
        self.skip_3r = skip_3r
        self.write_cleaned = write_cleaned

        logger.info("Pipeline initialised | device=%s | workers=%s | use_ml=%s",
                    device, workers, use_ml)

        # Detector config dicts (used for pickling in multiprocessing)
        porn_cfg = dict(use_ml=use_ml, device=device, ml_threshold=ml_threshold,
                        keyword_threshold=keyword_threshold)
        three_r_cfg = dict(use_ml=use_ml, device=device, ml_threshold=ml_threshold,
                           keyword_threshold=keyword_threshold)

        if workers == 0:
            # Single-process: init detectors directly
            self._porn = PornDetector(**porn_cfg) if not skip_porn else None
            self._three_r = ThreeRDetector(**three_r_cfg) if not skip_3r else None
        else:
            self._porn = None
            self._three_r = None

        self._porn_cfg = porn_cfg
        self._three_r_cfg = three_r_cfg
        self._reporter = Reporter(
            output_dir=self.output_dir / "reports",
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # CUDA auto-detect
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Core scan + clean
    # ------------------------------------------------------------------

    def run_local(self, input_path: str) -> dict:
        """Scan a local JSONL file."""
        logger.info("Scanning local file: %s", input_path)
        records = _iter_local_jsonl(input_path)
        return self._process_stream(records, output_stem=Path(input_path).stem)

    def run_hf(
        self,
        dataset_name: str,
        subset: str,
        split: str = "train",
    ) -> dict:
        """Scan a HuggingFace dataset."""
        records = _iter_hf_dataset(dataset_name, subset, split=split)
        output_stem = f"{dataset_name}_{subset}_{split}".replace("/", "_")
        return self._process_stream(records, output_stem=output_stem)

    def _process_stream(
        self,
        stream: Iterator[tuple[int, str, str]],
        output_stem: str = "dataset",
    ) -> dict:
        """
        Main processing loop.

        Handles both single-process and multiprocessing modes.
        """
        cleaned_path = self.output_dir / f"{output_stem}_cleaned.jsonl"
        total_processed = 0
        start_time = time.perf_counter()

        if self.workers > 0:
            result = self._process_parallel(stream, cleaned_path)
        else:
            result = self._process_sequential(stream, cleaned_path)

        elapsed = time.perf_counter() - start_time
        total_processed = result.get("total_processed", 0)
        throughput = total_processed / elapsed if elapsed > 0 else 0

        logger.info(
            "Processed %d records in %.1fs (%.0f rec/s)",
            total_processed, elapsed, throughput,
        )

        summary = self._reporter.finalize()
        summary["total_processed"] = total_processed
        summary["elapsed_seconds"] = round(elapsed, 2)
        summary["throughput_records_per_sec"] = round(throughput, 1)
        summary["device"] = self.device
        summary["cleaned_output"] = str(cleaned_path) if self.write_cleaned else None

        return summary

    def _process_sequential(
        self,
        stream: Iterator[tuple[int, str, str]],
        cleaned_path: Path,
    ) -> dict:
        """Single-process sequential scan."""
        total = 0
        cleaned_fh = open(cleaned_path, "w", encoding="utf-8") if self.write_cleaned else None

        try:
            for line_no, dataset_id, text in stream:
                total += 1

                porn_result = self._porn.detect(text) if self._porn else None
                three_r_result = self._three_r.detect(text) if self._three_r else None

                self._reporter.report(
                    line_no=line_no,
                    text=text,
                    dataset_id=dataset_id,
                    porn_result=porn_result,
                    three_r_result=three_r_result,
                )

                is_flagged = (
                    (porn_result and porn_result.is_explicit)
                    or (three_r_result and three_r_result.is_sensitive)
                )

                if not is_flagged and cleaned_fh:
                    cleaned_fh.write(
                        json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    )

                if total % 10_000 == 0:
                    logger.info("  Processed %d records…", total)

        finally:
            if cleaned_fh:
                cleaned_fh.close()

        return {"total_processed": total}

    def _process_parallel(
        self,
        stream: Iterator[tuple[int, str, str]],
        cleaned_path: Path,
    ) -> dict:
        """
        Multiprocessing scan using a Pool.
        Results are gathered in order; reporter runs in main process.
        """
        total = 0
        cleaned_fh = open(cleaned_path, "w", encoding="utf-8") if self.write_cleaned else None

        ctx = mp.get_context("spawn")
        pool = ctx.Pool(
            processes=self.workers,
            initializer=_worker_init,
            initargs=(self._porn_cfg, self._three_r_cfg),
        )

        try:
            batch: list[tuple[int, str, str]] = []

            def flush_batch(b: list) -> None:
                nonlocal total
                for result in pool.map(_worker_process, b):
                    total += 1
                    # Reconstruct lightweight result objects for reporter
                    _fake_porn = _DictResult(result["porn"])
                    _fake_3r = _DictResult3R(result["three_r"])

                    self._reporter.report(
                        line_no=result["line_no"],
                        text=result["text"],
                        dataset_id=result["dataset_id"],
                        porn_result=_fake_porn,
                        three_r_result=_fake_3r,
                    )
                    if result["keep"] and cleaned_fh:
                        cleaned_fh.write(
                            json.dumps({"text": result["text"]}, ensure_ascii=False) + "\n"
                        )

                    if total % 10_000 == 0:
                        logger.info("  Processed %d records…", total)

            for item in stream:
                batch.append(item)
                if len(batch) >= self.batch_size:
                    flush_batch(batch)
                    batch.clear()

            if batch:
                flush_batch(batch)

        finally:
            pool.close()
            pool.join()
            if cleaned_fh:
                cleaned_fh.close()

        return {"total_processed": total}


# ---------------------------------------------------------------------------
# Lightweight proxy objects for parallel mode result passing
# ---------------------------------------------------------------------------

class _DictResult:
    """Proxy for PornDetectionResult when reconstructed from dict in main process."""
    def __init__(self, d: dict):
        self.is_explicit: bool = d.get("is_explicit", False)
        self.language: str = d.get("language", "")
        self.matched_keywords: list[str] = d.get("matched_keywords", [])
        self.ml_score: Optional[float] = d.get("ml_score")
        self.ml_label: Optional[str] = d.get("ml_label")

    def to_dict(self) -> dict:
        return self.__dict__


class _DictResult3R:
    """Proxy for ThreeRResult when reconstructed from dict in main process."""
    def __init__(self, d: dict):
        self.is_sensitive: bool = d.get("is_sensitive", False)
        self.race_flagged: bool = d.get("race_flagged", False)
        self.religion_flagged: bool = d.get("religion_flagged", False)
        self.royalty_flagged: bool = d.get("royalty_flagged", False)
        self.race_matches: list[str] = d.get("race_matches", [])
        self.religion_matches: list[str] = d.get("religion_matches", [])
        self.royalty_matches: list[str] = d.get("royalty_matches", [])
        self.ml_scores: dict = d.get("ml_scores", {})

    @property
    def categories(self) -> list[str]:
        cats = []
        if self.race_flagged:
            cats.append("RACE")
        if self.religion_flagged:
            cats.append("RELIGION")
        if self.royalty_flagged:
            cats.append("ROYALTY")
        return cats

    def to_dict(self) -> dict:
        return {**self.__dict__, "categories": self.categories}


# ---------------------------------------------------------------------------
# CUDA auto-detect (module-level)
# ---------------------------------------------------------------------------

def _auto_detect_device() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info("CUDA available: %s (%.1f GB VRAM)", gpu_name, vram)
            return "cuda"
    except ImportError:
        pass
    logger.info("CUDA not available; using CPU.")
    return "cpu"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MC4 / Large Dataset Content Cleaning Pipeline\n"
                    "Detects: Pornographic content (BM+EN) + Malaysia 3R Sensitive Issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keyword-only scan of local JSONL:
  python pipeline.py --input data/mc4_ms.jsonl --output output/

  # Full ML + CUDA scan:
  python pipeline.py --input data/mc4_ms.jsonl --output output/ --use-ml --device cuda

  # Stream from HuggingFace mc4 (ms subset):
  python pipeline.py --hf-dataset mc4 --hf-subset ms --output output/ --use-ml

  # Parallel with 8 workers:
  python pipeline.py --input data/mc4_ms.jsonl --output output/ --workers 8

  # Scan only, no cleaned output:
  python pipeline.py --input data/mc4_ms.jsonl --no-write-cleaned
        """,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", metavar="FILE", help="Path to local JSONL input file")
    src.add_argument(
        "--hf-dataset",
        metavar="DATASET",
        help="HuggingFace dataset name (e.g. mc4)",
    )

    parser.add_argument(
        "--hf-subset", default="ms", metavar="SUBSET",
        help="HuggingFace dataset subset / language code (default: ms)",
    )
    parser.add_argument(
        "--hf-split", default="train", metavar="SPLIT",
        help="HuggingFace dataset split (default: train)",
    )
    parser.add_argument(
        "--output", "-o", default="output", metavar="DIR",
        help="Output directory for cleaned data + reports (default: output/)",
    )
    parser.add_argument(
        "--use-ml", action="store_true",
        help="Enable ML-based detection (requires transformers)",
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu"], default=None,
        help="Compute device (auto-detected if not specified)",
    )
    parser.add_argument(
        "--workers", type=int, default=0, metavar="N",
        help="Number of parallel worker processes (0=single process)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, metavar="N",
        help="Batch size for parallel processing (default: 256)",
    )
    parser.add_argument(
        "--ml-threshold", type=float, default=0.70, metavar="F",
        help="ML confidence threshold 0–1 (default: 0.70)",
    )
    parser.add_argument(
        "--keyword-threshold", type=int, default=1, metavar="N",
        help="Min keyword hits to flag (default: 1)",
    )
    parser.add_argument(
        "--skip-porn", action="store_true",
        help="Disable pornographic content scan",
    )
    parser.add_argument(
        "--skip-3r", action="store_true",
        help="Disable 3R sensitive content scan",
    )
    parser.add_argument(
        "--no-write-cleaned", action="store_true",
        help="Do not write cleaned output file (scan/report only)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-record console output",
    )
    parser.add_argument(
        "--nsfw-model", metavar="MODEL_ID",
        help="Override HuggingFace NSFW model ID (sets NSFW_MODEL env var)",
    )
    parser.add_argument(
        "--zs-model", metavar="MODEL_ID",
        help="Override zero-shot model ID for 3R detection (sets ZS_MODEL env var)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Override model IDs via env vars
    if args.nsfw_model:
        os.environ["NSFW_MODEL"] = args.nsfw_model
    if args.zs_model:
        os.environ["ZS_MODEL"] = args.zs_model

    pipeline = CleaningPipeline(
        output_dir=args.output,
        use_ml=args.use_ml,
        device=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        verbose=not args.quiet,
        skip_porn=args.skip_porn,
        skip_3r=args.skip_3r,
        write_cleaned=not args.no_write_cleaned,
        ml_threshold=args.ml_threshold,
        keyword_threshold=args.keyword_threshold,
    )

    if args.input:
        summary = pipeline.run_local(args.input)
    else:
        summary = pipeline.run_hf(
            dataset_name=args.hf_dataset,
            subset=args.hf_subset,
            split=args.hf_split,
        )

    logger.info("Pipeline complete. Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
