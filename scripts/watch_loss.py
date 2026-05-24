#!/usr/bin/env python3
"""Serve a tiny live loss dashboard for MicroLM JSONL training logs."""

from __future__ import annotations

import argparse
import json
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


LOSS_KEYS = {
    "train_loss": ("train_loss", "train/loss", "loss"),
    "val_loss": ("val_loss", "valid_loss", "validation_loss", "val/loss"),
    "lr": ("lr", "learning_rate", "train/lr"),
}


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_number(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _as_float(record.get(key))
        if value is not None:
            return value
    return None


def load_loss_points(log_path: Path) -> list[dict[str, float]]:
    """Read a JSONL training log and normalize common loss field names."""
    if not log_path.exists():
        return []

    points: list[dict[str, float]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue

            step = _as_float(record.get("step"))
            if step is None:
                step = _as_float(record.get("iter"))
            if step is None:
                step = _as_float(record.get("iteration"))
            if step is None:
                step = float(index)

            point: dict[str, float] = {"step": step}
            for output_key, input_keys in LOSS_KEYS.items():
                value = _first_number(record, input_keys)
                if value is not None:
                    point[output_key] = value
            if "train_loss" in point or "val_loss" in point:
                points.append(point)
    return points


def build_index_html(refresh_ms: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MicroLM Loss</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f7f7f3;
      --fg: #202124;
      --muted: #6b7280;
      --grid: rgba(32, 33, 36, 0.14);
      --train: #2563eb;
      --val: #dc2626;
      --panel: #ffffff;
      --border: rgba(32, 33, 36, 0.14);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #121313;
        --fg: #f2f2ee;
        --muted: #a8adb7;
        --grid: rgba(242, 242, 238, 0.14);
        --panel: #1b1c1e;
        --border: rgba(242, 242, 238, 0.16);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--fg);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1120px, calc(100vw - 32px));
      margin: 24px auto;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    #status {{
      color: var(--muted);
      text-align: right;
      overflow-wrap: anywhere;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: min(68vh, 620px);
      min-height: 360px;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      align-items: center;
      margin-top: 12px;
      color: var(--muted);
      flex-wrap: wrap;
    }}
    .legend span {{
      display: inline-flex;
      gap: 7px;
      align-items: center;
    }}
    .swatch {{
      width: 22px;
      height: 3px;
      border-radius: 999px;
      display: inline-block;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>MicroLM Loss</h1>
      <div id="status">Waiting for log...</div>
    </header>
    <section class="panel">
      <canvas id="chart" aria-label="Training and validation loss chart"></canvas>
      <div class="legend">
        <span><i class="swatch" style="background: var(--train)"></i>train_loss</span>
        <span><i class="swatch" style="background: var(--val)"></i>val_loss</span>
      </div>
    </section>
  </main>
  <script>
    const canvas = document.getElementById("chart");
    const statusEl = document.getElementById("status");
    const css = getComputedStyle(document.documentElement);

    function fmt(value) {{
      if (value === undefined || value === null || Number.isNaN(value)) return "-";
      if (Math.abs(value) >= 100) return value.toFixed(1);
      if (Math.abs(value) >= 10) return value.toFixed(2);
      return value.toFixed(4);
    }}

    function resizeCanvas() {{
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      return {{ width: canvas.width, height: canvas.height, ratio }};
    }}

    function draw(points) {{
      const ctx = canvas.getContext("2d");
      const {{ width, height, ratio }} = resizeCanvas();
      ctx.scale(ratio, ratio);
      const w = width / ratio;
      const h = height / ratio;
      ctx.clearRect(0, 0, w, h);

      const pad = {{ left: 60, right: 22, top: 24, bottom: 42 }};
      const plotW = w - pad.left - pad.right;
      const plotH = h - pad.top - pad.bottom;
      const fg = css.getPropertyValue("--fg").trim();
      const muted = css.getPropertyValue("--muted").trim();
      const grid = css.getPropertyValue("--grid").trim();
      const trainColor = css.getPropertyValue("--train").trim();
      const valColor = css.getPropertyValue("--val").trim();

      ctx.font = "12px system-ui, sans-serif";
      ctx.lineWidth = 1;

      if (!points.length) {{
        ctx.fillStyle = muted;
        ctx.textAlign = "center";
        ctx.fillText("No loss points yet. Start training or check the log path.", w / 2, h / 2);
        return;
      }}

      const xs = points.map(p => Number(p.step)).filter(Number.isFinite);
      const ys = points.flatMap(p => [p.train_loss, p.val_loss]).map(Number).filter(Number.isFinite);
      let xMin = Math.min(...xs);
      let xMax = Math.max(...xs);
      let yMin = Math.min(...ys);
      let yMax = Math.max(...ys);
      if (xMin === xMax) {{ xMin -= 1; xMax += 1; }}
      if (yMin === yMax) {{ yMin -= 0.5; yMax += 0.5; }}
      const yPad = (yMax - yMin) * 0.08 || 0.5;
      yMin = Math.max(0, yMin - yPad);
      yMax += yPad;

      const x = value => pad.left + ((value - xMin) / (xMax - xMin)) * plotW;
      const y = value => pad.top + (1 - ((value - yMin) / (yMax - yMin))) * plotH;

      ctx.strokeStyle = grid;
      ctx.fillStyle = muted;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i = 0; i <= 5; i++) {{
        const value = yMin + ((yMax - yMin) * i) / 5;
        const py = y(value);
        ctx.beginPath();
        ctx.moveTo(pad.left, py);
        ctx.lineTo(w - pad.right, py);
        ctx.stroke();
        ctx.fillText(fmt(value), pad.left - 8, py);
      }}

      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i = 0; i <= 5; i++) {{
        const value = xMin + ((xMax - xMin) * i) / 5;
        const px = x(value);
        ctx.fillText(Math.round(value).toString(), px, h - pad.bottom + 12);
      }}

      ctx.strokeStyle = fg;
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, h - pad.bottom);
      ctx.lineTo(w - pad.right, h - pad.bottom);
      ctx.stroke();

      function line(key, color) {{
        const series = points
          .filter(p => Number.isFinite(Number(p[key])))
          .map(p => [Number(p.step), Number(p[key])]);
        if (!series.length) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        series.forEach(([sx, sy], index) => {{
          const px = x(sx);
          const py = y(sy);
          if (index === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }});
        ctx.stroke();
        ctx.lineWidth = 1;
      }}

      line("train_loss", trainColor);
      line("val_loss", valColor);
    }}

    async function refresh() {{
      try {{
        const res = await fetch("/data", {{ cache: "no-store" }});
        const data = await res.json();
        draw(data.points || []);
        const last = (data.points || []).at(-1) || {{}};
        const parts = [
          data.exists ? data.path : `Missing: ${{data.path}}`,
          `${{data.points.length}} points`,
          `step ${{fmt(last.step)}}`,
          `train ${{fmt(last.train_loss)}}`,
          `val ${{fmt(last.val_loss)}}`,
        ];
        if (last.lr !== undefined) parts.push(`lr ${{fmt(last.lr)}}`);
        statusEl.textContent = parts.join(" | ");
      }} catch (error) {{
        statusEl.textContent = `Refresh failed: ${{error}}`;
      }}
    }}

    window.addEventListener("resize", refresh);
    refresh();
    setInterval(refresh, {refresh_ms});
  </script>
</body>
</html>
"""


def make_handler(log_path: Path, refresh_ms: int) -> type[BaseHTTPRequestHandler]:
    class LossDashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send_text(build_index_html(refresh_ms), "text/html; charset=utf-8")
                return
            if path == "/data":
                payload = {
                    "path": str(log_path),
                    "exists": log_path.exists(),
                    "updated_at": time.time(),
                    "points": load_loss_points(log_path),
                }
                self._send_json(payload)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_text(self, body: str, content_type: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return LossDashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch MicroLM train_log.jsonl as a live loss chart.")
    parser.add_argument("--log", type=Path, required=True, help="Path to train_log.jsonl.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--refresh", type=float, default=2.0, help="Refresh interval in seconds.")
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the default browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.log.expanduser().resolve()
    refresh_ms = max(250, int(args.refresh * 1000))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(log_path, refresh_ms))
    url = f"http://{args.host}:{args.port}"
    print(f"Watching {log_path}")
    print(f"Dashboard: {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
