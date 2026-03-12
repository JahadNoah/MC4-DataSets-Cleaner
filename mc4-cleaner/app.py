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
        "status": "running",
        "summary": None,
        "output_dir": str(Path("output") / job_id),
        "filename": safe_name,
    }

    thread = threading.Thread(
        target=_run_job, args=(job_id, str(input_path), opts), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id})


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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    print("MC4 Cleaner UI →  http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
