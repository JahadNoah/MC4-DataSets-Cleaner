"""
app.py
======
Local web UI for the MC4 Content Cleaning Pipeline.

Run:
    python app.py
Then open:
    http://localhost:5000
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path("output") / "_uploads"
ALLOWED_SUFFIXES = {".jsonl", ".json", ".txt"}
MAX_CONTENT_MB = 2048  # 2 GB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# In-memory job store: job_id -> state dict
JOBS: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Global job queue — jobs are processed one at a time by a worker thread
# ---------------------------------------------------------------------------

JOB_QUEUE: list[str] = []          # ordered list of job_ids waiting to run
_JOB_QUEUE_LOCK = threading.Lock()
_JOB_QUEUE_EVENT = threading.Event()  # signals the worker that a new job arrived


def _queue_worker() -> None:
    """Background thread: pops jobs from JOB_QUEUE and runs them sequentially."""
    while True:
        _JOB_QUEUE_EVENT.wait()
        _JOB_QUEUE_EVENT.clear()
        while True:
            with _JOB_QUEUE_LOCK:
                if not JOB_QUEUE:
                    break
                job_id = JOB_QUEUE[0]  # peek, don't pop yet

            job = JOBS.get(job_id)
            if job is None or job["status"] == "cancelled":
                with _JOB_QUEUE_LOCK:
                    if JOB_QUEUE and JOB_QUEUE[0] == job_id:
                        JOB_QUEUE.pop(0)
                continue

            # Mark as running
            job["status"] = "running"
            _run_job(job_id, job["input_path"], job["opts"])

            # Remove from queue after completion
            with _JOB_QUEUE_LOCK:
                if JOB_QUEUE and JOB_QUEUE[0] == job_id:
                    JOB_QUEUE.pop(0)


_worker_thread = threading.Thread(target=_queue_worker, daemon=True, name="queue-worker")
_worker_thread.start()

# ---------------------------------------------------------------------------
# Logging → SSE bridge
# ---------------------------------------------------------------------------

_WATCHED_LOGGERS = [
    "pipeline",
    "detectors.porn_detector",
    "detectors.sensitive_3r_detector",
    "reporter",
]


class _QueueHandler(logging.Handler):
    """Forward log records into an in-memory queue for SSE streaming."""

    def __init__(self, q: "queue.Queue[dict]") -> None:
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._q.put_nowait(
                {
                    "type": "log",
                    "level": record.levelname,
                    "msg": self.format(record),
                }
            )
        except queue.Full:
            pass  # drop message rather than block


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------


def _run_job(job_id: str, input_path: str, opts: dict) -> None:
    """Run the cleaning pipeline in a daemon thread and feed events to SSE."""
    q: queue.Queue = JOBS[job_id]["queue"]
    output_dir = Path("output") / job_id

    handler = _QueueHandler(q)
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    )
    for name in _WATCHED_LOGGERS:
        logging.getLogger(name).addHandler(handler)

    try:
        from pipeline import CleaningPipeline  # local import per thread

        pipeline = CleaningPipeline(
            output_dir=str(output_dir),
            use_ml=opts.get("use_ml", False),
            workers=opts.get("workers", 0),
            verbose=False,
            skip_porn=opts.get("skip_porn", False),
            skip_3r=opts.get("skip_3r", False),
            ml_threshold=opts.get("ml_threshold", 0.70),
        )
        summary = pipeline.run_local(input_path)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["summary"] = summary
        q.put({"type": "done", "summary": summary})
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("pipeline").exception("Pipeline error in job %s", job_id)
        JOBS[job_id]["status"] = "error"
        q.put({"type": "error", "msg": str(exc)})
    finally:
        for name in _WATCHED_LOGGERS:
            logging.getLogger(name).removeHandler(handler)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/clean", methods=["POST"])
def clean():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(f.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return jsonify(
            {"error": f"Unsupported type '{suffix}'. Use .jsonl, .json, or .txt"}
        ), 400

    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / job_id
    upload_path.mkdir(parents=True, exist_ok=True)

    safe_name = secure_filename(f.filename)
    input_path = upload_path / safe_name
    f.save(str(input_path))

    try:
        opts = {
            "skip_porn": request.form.get("skip_porn") == "true",
            "skip_3r": request.form.get("skip_3r") == "true",
            "use_ml": request.form.get("use_ml") == "true",
            "workers": max(0, min(int(request.form.get("workers") or 0), 64)),
            "ml_threshold": max(
                0.5, min(float(request.form.get("ml_threshold") or 0.70), 0.99)
            ),
        }
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid option value: {exc}"}), 400

    JOBS[job_id] = {
        "queue": queue.Queue(maxsize=5000),
        "status": "queued",
        "summary": None,
        "output_dir": str(Path("output") / job_id),
        "filename": safe_name,
        "input_path": str(input_path),
        "opts": opts,
    }

    with _JOB_QUEUE_LOCK:
        JOB_QUEUE.append(job_id)
    _JOB_QUEUE_EVENT.set()

    queue_pos = JOB_QUEUE.index(job_id) + 1 if job_id in JOB_QUEUE else 1
    return jsonify({"job_id": job_id, "queue_position": queue_pos})


@app.route("/progress/<job_id>")
def progress(job_id: str):
    """SSE stream — sends log events until the job finishes."""
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        q: queue.Queue = JOBS[job_id]["queue"]
        while True:
            try:
                event = q.get(timeout=25)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                yield 'data: {"type":"heartbeat"}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/download/<job_id>/<kind>")
def download(job_id: str, kind: str):
    """Return a result file as a download attachment."""
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404

    job = JOBS[job_id]
    out = Path(job["output_dir"])
    stem = Path(job["filename"]).stem

    file_map = {
        "cleaned":      out / f"{stem}_cleaned.jsonl",
        "flagged_csv":  out / "reports" / "flagged_summary.csv",
        "flagged_jsonl": out / "reports" / "flagged.jsonl",
    }

    if kind not in file_map:
        return jsonify({"error": "Unknown file kind"}), 400

    target = file_map[kind]
    if not target.exists():
        return jsonify({"error": "File not ready"}), 404

    return send_file(
        str(target.resolve()), as_attachment=True, download_name=target.name
    )


# ---------------------------------------------------------------------------
# Queue management endpoints
# ---------------------------------------------------------------------------


@app.route("/queue")
def queue_status():
    """Return full queue state: waiting, running, and recently finished jobs."""
    with _JOB_QUEUE_LOCK:
        waiting = list(JOB_QUEUE)

    rows = []
    for job_id, job in JOBS.items():
        queue_pos = waiting.index(job_id) + 1 if job_id in waiting else None
        rows.append({
            "job_id": job_id,
            "filename": job.get("filename", ""),
            "status": job["status"],
            "queue_position": queue_pos,
            "summary": job.get("summary"),
        })
    # Sort: running first, then queued by position, then done/error/cancelled by recency
    def sort_key(r):
        s = r["status"]
        if s == "running":    return (0, 0)
        if s == "queued":     return (1, r["queue_position"] or 99)
        if s == "done":       return (2, 0)
        if s == "error":      return (3, 0)
        return (4, 0)
    rows.sort(key=sort_key)
    return jsonify({"jobs": rows, "queue_length": len(waiting)})


@app.route("/queue/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id: str):
    """Cancel a queued (not yet running) job."""
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404
    job = JOBS[job_id]
    if job["status"] != "queued":
        return jsonify({"error": f"Cannot cancel a job with status '{job['status']}'"}), 400

    job["status"] = "cancelled"
    with _JOB_QUEUE_LOCK:
        if job_id in JOB_QUEUE:
            JOB_QUEUE.remove(job_id)
    job["queue"].put({"type": "error", "msg": "Job cancelled by user."})
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Review endpoints
# ---------------------------------------------------------------------------


@app.route("/flagged/<job_id>")
def flagged_records(job_id: str):
    """Return paginated, filterable flagged records for a completed job."""
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404
    job = JOBS[job_id]
    if job["status"] != "done":
        return jsonify({"error": "Job not complete"}), 400

    flagged_path = Path(job["output_dir"]) / "reports" / "flagged.jsonl"
    if not flagged_path.exists():
        return jsonify({"records": [], "total": 0, "page": 1, "total_pages": 0})

    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = max(1, min(int(request.args.get("per_page", 20)), 100))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid pagination params"}), 400

    filter_type = request.args.get("filter", "all")
    search = request.args.get("q", "").lower().strip()

    records: list[dict] = []
    with open(flagged_path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                pass

    if filter_type == "porn":
        records = [r for r in records if r.get("is_porn")]
    elif filter_type == "3r":
        records = [r for r in records if r.get("is_sensitive_3r")]
    elif filter_type == "race":
        records = [r for r in records if "race" in r.get("three_r_categories", [])]
    elif filter_type == "religion":
        records = [r for r in records if "religion" in r.get("three_r_categories", [])]
    elif filter_type == "royalty":
        records = [r for r in records if "royalty" in r.get("three_r_categories", [])]

    if search:
        records = [r for r in records if search in r.get("text_preview", "").lower()]

    total = len(records)
    start = (page - 1) * per_page
    return jsonify({
        "records": records[start: start + per_page],
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page) if total else 0,
    })


@app.route("/reapply/<job_id>", methods=["POST"])
def reapply(job_id: str):
    """Re-generate cleaned file honouring user keep/remove decisions."""
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404
    job = JOBS[job_id]
    if job["status"] != "done":
        return jsonify({"error": "Job not complete"}), 400

    data = request.get_json(silent=True) or {}
    try:
        keep_lines: set[int] = {int(x) for x in data.get("keep_lines", [])}
    except (ValueError, TypeError):
        return jsonify({"error": "keep_lines must be a list of integers"}), 400

    input_path = job.get("input_path")
    if not input_path or not Path(input_path).exists():
        return jsonify({"error": "Original input file not found on server"}), 404

    flagged_path = Path(job["output_dir"]) / "reports" / "flagged.jsonl"
    flagged_line_nos: set[int] = set()
    if flagged_path.exists():
        with open(flagged_path, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    flagged_line_nos.add(int(json.loads(raw)["line_no"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

    remove_lines = flagged_line_nos - keep_lines
    out_dir = Path(job["output_dir"])
    stem = Path(job["filename"]).stem
    custom_path = out_dir / f"{stem}_custom_cleaned.jsonl"

    from pipeline import _iter_local_jsonl  # local import

    count_written = 0
    with open(custom_path, "w", encoding="utf-8") as out_fh:
        for line_no, _dataset_id, text in _iter_local_jsonl(input_path):
            if line_no not in remove_lines:
                out_fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                count_written += 1

    job["custom_clean_path"] = str(custom_path)
    job["custom_clean_name"] = custom_path.name
    return jsonify({
        "records_written": count_written,
        "records_removed": len(remove_lines),
        "download_url": f"/download-custom/{job_id}",
        "filename": custom_path.name,
    })


@app.route("/download-custom/<job_id>")
def download_custom(job_id: str):
    """Serve the custom-cleaned file."""
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404
    job = JOBS[job_id]
    path = job.get("custom_clean_path")
    if not path or not Path(path).exists():
        return jsonify({"error": "Custom cleaned file not ready"}), 404
    return send_file(
        str(Path(path).resolve()),
        as_attachment=True,
        download_name=job.get("custom_clean_name", "custom_cleaned.jsonl"),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    print("MC4 Cleaner UI →  http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
