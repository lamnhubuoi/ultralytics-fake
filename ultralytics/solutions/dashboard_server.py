# Ultralytics \ud83d\ude80 AGPL-3.0 License - https://ultralytics.com/license
from __future__ import annotations

import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


class DashboardServer:
    """Simple Flask server that exposes a dashboard for running YOLO predictions from the browser.

    The server accepts image uploads, runs inference using a configured YOLO model, and serves the
    annotated results through a lightweight HTML dashboard. It is designed as a minimal reference
    implementation that can be extended to include additional endpoints or monitoring widgets.

    Attributes:
        model (YOLO): YOLO model used for running predictions.
        root (Path): Base directory for uploads and generated artifacts.
        upload_dir (Path): Directory used to store uploaded images before inference.
        results_dir (Path): Directory where annotated prediction images are written.
        app (Flask): Underlying Flask application instance.

    Examples:
        Launch the dashboard on http://localhost:8000
        >>> from ultralytics.solutions import DashboardServer
        >>> server = DashboardServer(model="yolo11n.pt")
        >>> server.run()
    """

    def __init__(self, model: str = "yolo11n.pt", root: str | Path = "runs/dashboard") -> None:
        check_requirements("flask>=3.0.1")
        from flask import Flask, jsonify, render_template, request, url_for
        from werkzeug.utils import secure_filename

        self.model = YOLO(model)
        self.root = Path(root)
        self.upload_dir = self.root / "uploads"
        self.results_dir = self.root / "results"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._jsonify = jsonify
        self._request = request
        self._render_template = render_template
        self._url_for = url_for
        self._secure_filename = secure_filename

        self.app = Flask(
            __name__,
            template_folder="templates",
            static_folder=self.results_dir,
            static_url_path="/results",
        )
        self.app.add_url_rule("/", view_func=self.index, methods=["GET"])
        self.app.add_url_rule("/health", view_func=self.health, methods=["GET"])
        self.app.add_url_rule("/predict", view_func=self.predict, methods=["POST"])

    def _recent_predictions(self, limit: int = 8) -> list[str]:
        """Return the relative paths for the most recent prediction images."""

        def is_image(path: Path) -> bool:
            mime_type, _ = mimetypes.guess_type(path.name)
            return mime_type is not None and mime_type.startswith("image")

        candidates: Iterable[Path] = self.results_dir.glob("*/*")
        images = [p for p in candidates if p.is_file() and is_image(p)]
        images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [f"{p.parent.name}/{p.name}" for p in images[:limit]]

    def index(self) -> str:
        """Render the dashboard home page."""
        context = {
            "model_label": str(self.model),
            "recent_results": self._recent_predictions(),
            "upload_dir": str(self.upload_dir),
        }
        return self._render_template("dashboard.html", **context)

    def health(self):
        """Simple health check endpoint."""
        return self._jsonify({"status": "ok", "model": str(self.model), "results_dir": str(self.results_dir)})

    def predict(self):
        """Run prediction on an uploaded image and return the result location."""
        uploaded = self._request.files.get("file")
        if uploaded is None or uploaded.filename == "":
            return self._jsonify({"error": "Please attach an image file."}), 400

        filename = self._secure_filename(uploaded.filename)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        upload_path = self.upload_dir / f"{timestamp}-{filename}"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        uploaded.save(upload_path)

        LOGGER.info("Running dashboard prediction for %s", upload_path)
        results = self.model.predict(
            source=str(upload_path), save=True, project=str(self.results_dir), name=timestamp, exist_ok=True, verbose=False
        )

        result_dir = Path(results[0].save_dir)
        relative_path = f"{result_dir.name}/{upload_path.name}"

        speed = results[0].speed or {}
        timings = {k: round(v, 2) for k, v in speed.items()}
        response = {
            "result_path": relative_path,
            "result_url": self._url_for("static", filename=relative_path),
            "timings_ms": timings,
        }
        return self._jsonify(response)

    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False) -> None:
        """Start the Flask application."""
        self.app.run(host=host, port=port, debug=debug)
