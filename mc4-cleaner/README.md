# MC4 Content Cleaning Pipeline

A high-performance pipeline to detect and remove **pornographic content** and **Malaysia 3R sensitive issues** from large datasets such as [mC4](https://huggingface.co/datasets/mc4) in both **Bahasa Melayu (BM)** and **English (EN)**.

---

## What It Detects

### Pornographic / Explicit Content
- Direct sexual terms, acts, body parts (EN + BM slang)
- Adult platform references (OnlyFans, PornHub, xvideos, etc.)
- Bahasa Melayu-specific slang: `bogel`, `lucah`, `liwat`, `burit`, `konek`, `puki`, etc.
- Optional ML scoring via HuggingFace toxicity/NSFW models

### Malaysia 3R Sensitive Issues

| Category | Description | Legal Reference |
|----------|-------------|-----------------|
| **Race (Kaum)** | Racial slurs, incitement, ethnic supremacy, racial riots | Sedition Act 1948; Penal Code |
| **Religion (Agama)** | Blasphemy, apostasy content, religious incitement, insult to prophets | Penal Code s.298, s.298A |
| **Royalty (Raja-raja)** | Insults to YDPA, Sultans, Diraja institution, abolish monarchy calls | Sedition Act 1948 s.3(1)(a); Penal Code s.499 |

---

## Architecture

```
mc4-cleaner/
├── pipeline.py                  ← Main entry point (CLI + orchestrator)
├── reporter.py                  ← Outputs: JSONL, CSV, console preview
├── requirements.txt
├── detectors/
│   ├── porn_detector.py         ← Explicit content detector (BM + EN)
│   └── sensitive_3r_detector.py ← 3R classifier (Race/Religion/Royalty)
├── keywords/
│   ├── porn_en.txt              ← English explicit keyword patterns
│   ├── porn_bm.txt              ← Bahasa Melayu explicit keyword patterns
│   ├── 3r_race.txt              ← Race-sensitive patterns (BM + EN)
│   ├── 3r_religion.txt          ← Religion-sensitive patterns (BM + EN)
│   └── 3r_royalty.txt           ← Royalty-sensitive patterns (BM + EN)
└── output/                      ← Generated outputs (cleaned data + reports)
    ├── reports/
    │   ├── flagged.jsonl         ← Full detail per flagged record
    │   ├── flagged_summary.csv   ← Columnar summary (line_no, issues, evidence)
    │   └── summary.json          ← Aggregate stats
    └── <dataset>_cleaned.jsonl   ← Dataset with flagged records removed
```

---

## Quick Start

### 1. Install dependencies

```bash
# CPU only (fast keyword scanning, no ML models):
pip install -r requirements.txt

# With CUDA 12.1 (recommended for ML mode):
pip install -r requirements.txt
pip install torch>=2.2.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

### 2. Scan a local JSONL file (keyword-only, fastest)

```bash
python pipeline.py --input data/mc4_ms.jsonl --output output/
```

### 3. Scan with ML models on CUDA GPU

```bash
python pipeline.py \
  --input data/mc4_ms.jsonl \
  --output output/ \
  --use-ml \
  --device cuda
```

### 4. Stream directly from HuggingFace mc4 (Malay subset)

```bash
python pipeline.py \
  --hf-dataset mc4 \
  --hf-subset ms \
  --output output/ \
  --use-ml \
  --device cuda
```

### 5. Parallel processing (8 CPU workers)

```bash
python pipeline.py \
  --input data/mc4_ms.jsonl \
  --output output/ \
  --workers 8 \
  --batch-size 512
```

### 6. Scan only (no cleaned output, just reports)

```bash
python pipeline.py \
  --input data/mc4_ms.jsonl \
  --output output/ \
  --no-write-cleaned
```

---

## CLI Reference

```
usage: pipeline.py [-h] (--input FILE | --hf-dataset DATASET)
                   [--hf-subset SUBSET] [--hf-split SPLIT]
                   [--output DIR] [--use-ml] [--device {cuda,cpu}]
                   [--workers N] [--batch-size N]
                   [--ml-threshold F] [--keyword-threshold N]
                   [--skip-porn] [--skip-3r]
                   [--no-write-cleaned] [--quiet]
                   [--nsfw-model MODEL_ID] [--zs-model MODEL_ID]

Options:
  --input FILE           Local JSONL input file (one JSON per line, 'text' field)
  --hf-dataset DATASET   HuggingFace dataset name (e.g. mc4)
  --hf-subset SUBSET     Dataset subset/language (default: ms)
  --hf-split SPLIT       Dataset split (default: train)
  --output DIR           Output directory (default: output/)
  --use-ml               Enable ML models (transformers)
  --device {cuda,cpu}    Compute device (auto-detected)
  --workers N            Parallel worker processes (0=single, default: 0)
  --batch-size N         Records per batch in parallel mode (default: 256)
  --ml-threshold F       ML confidence threshold 0–1 (default: 0.70)
  --keyword-threshold N  Min keyword hits to flag (default: 1)
  --skip-porn            Skip pornographic content scan
  --skip-3r              Skip 3R sensitive content scan
  --no-write-cleaned     Scan/report only, don't write cleaned JSONL
  --quiet                Suppress per-record console output
  --nsfw-model MODEL_ID  Override NSFW HuggingFace model
  --zs-model MODEL_ID    Override zero-shot HF model for 3R detection
```

---

## Console Output Example

```
[   Line 142] PORN[BM] 3R:RELIGION  ➜  Dia kata dalam artikel tu bahawa puki mat saleh tu kafir semua...
          Porn keywords   : «puki»
          Religion kws    : «kafir»

[   Line 891] 3R:RACE  ➜  Cina celaka semua patut balik China, Melayu tanah ini...
          Race keywords   : «cina celaka», «balik china»

[   Line 2034] 3R:ROYALTY  ➜  Sultan tu korup, kena bunuh je lah...
          Royalty kws     : «sultan», «korup», «bunuh»
```

---

## Output Files

| File | Description |
|------|-------------|
| `output/reports/flagged.jsonl` | Full JSON detail for every flagged record (line_no, text, keywords, ML scores) |
| `output/reports/flagged_summary.csv` | CSV with one row per flagged record |
| `output/reports/summary.json` | Aggregate counts (porn, race, religion, royalty, total) |
| `output/<name>_cleaned.jsonl` | Input dataset with all flagged records removed |

---

## ML Models Used

| Purpose | Default Model | Override Env Var |
|---------|--------------|------------------|
| NSFW / Toxicity | `unitary/unbiased-toxic-roberta` | `NSFW_MODEL` |
| 3R Zero-shot | `facebook/bart-large-mnli` | `ZS_MODEL` |

Override via CLI:
```bash
python pipeline.py --input data.jsonl \
  --nsfw-model "martin-ha/toxic-comment-model" \
  --zs-model "typeform/distilbart-mnli-12-3"
```

---

## Extending the Keyword Lists

All keyword lists are plain text files in `keywords/`. Each line is a **Python regex pattern** (case-insensitive). Lines starting with `#` are comments.

```
# Example: add Javanese slang to BM porn list
\bjembut\b
\bmemek\b
```

Re-run the pipeline — new keywords take effect immediately, no code changes needed.

---

## Legal Notice

This pipeline is built to assist in cleaning datasets for AI/NLP research in compliance with Malaysian law. The keyword lists cover:
- **Sedition Act 1948** — Racial and royalty incitement
- **Penal Code s.298 / s.298A** — Religious insult / harmony
- **Communications and Multimedia Act 1998 s.211** — Indecent/obscene content
- **Film Censorship Act 2002** — Pornographic material

The pipeline is for **content moderation and dataset cleaning purposes only**.
