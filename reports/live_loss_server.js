const http = require("http");
const fs = require("fs");
const path = require("path");

const cwd = path.resolve(__dirname, "..");
const logPath = path.join(cwd, "outputs", "sft_baseline", "train_log.jsonl");
const port = Number(process.env.LOSS_PORT || 7860);

function readRuns() {
  let text = "";
  try {
    text = fs.readFileSync(logPath, "utf8");
  } catch {
    return [];
  }

  const rows = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const item = JSON.parse(line);
      if (
        Number.isFinite(item.step) &&
        Number.isFinite(item.train_loss) &&
        Number.isFinite(item.val_loss)
      ) {
        rows.push({
          step: Number(item.step),
          train_loss: Number(item.train_loss),
          val_loss: Number(item.val_loss),
        });
      }
    } catch {}
  }

  const runs = [];
  let current = [];
  let previousStep = -Infinity;
  for (const row of rows) {
    if (current.length && row.step <= previousStep) {
      runs.push(current);
      current = [];
    }
    current.push(row);
    previousStep = row.step;
  }
  if (current.length) runs.push(current);
  return runs.map((points, index) => ({ index: index + 1, points }));
}

function jsonResponse(res, data) {
  const body = JSON.stringify(data);
  res.writeHead(200, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  res.end(body);
}

const html = String.raw`<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MicroLM SFT Loss</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f7f8fa;
      --fg: #17202a;
      --muted: #687280;
      --panel: #ffffff;
      --line: #d8dde5;
      --train: #1f77b4;
      --val: #d62728;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111418;
        --fg: #e7ebf0;
        --muted: #9aa5b1;
        --panel: #171c22;
        --line: #2b333d;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Segoe UI, Microsoft YaHei, Arial, sans-serif;
      background: var(--bg);
      color: var(--fg);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }
    main {
      padding: 18px 20px;
      max-width: 1180px;
      margin: 0 auto;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      color: var(--muted);
      font-size: 14px;
    }
    select, button {
      font: inherit;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--fg);
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .stat {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .value {
      font-size: 22px;
      font-variant-numeric: tabular-nums;
    }
    .chartWrap {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    canvas {
      display: block;
      width: 100%;
      height: 520px;
    }
    .legend {
      display: flex;
      gap: 16px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .swatch {
      display: inline-block;
      width: 14px;
      height: 3px;
      vertical-align: middle;
      margin-right: 6px;
    }
    @media (max-width: 760px) {
      header { align-items: flex-start; flex-direction: column; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      canvas { height: 360px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>MicroLM SFT Loss</h1>
    <div class="toolbar">
      <label>Run <select id="runSelect"></select></label>
      <button id="latestBtn">Latest</button>
      <span id="status">loading</span>
    </div>
  </header>
  <main>
    <section class="stats">
      <div class="stat"><div class="label">latest step</div><div class="value" id="step">-</div></div>
      <div class="stat"><div class="label">train loss</div><div class="value" id="train">-</div></div>
      <div class="stat"><div class="label">val loss</div><div class="value" id="val">-</div></div>
      <div class="stat"><div class="label">points</div><div class="value" id="points">-</div></div>
    </section>
    <section class="chartWrap">
      <canvas id="chart"></canvas>
      <div class="legend">
        <span><span class="swatch" style="background:var(--train)"></span>train_loss</span>
        <span><span class="swatch" style="background:var(--val)"></span>val_loss</span>
      </div>
    </section>
  </main>
  <script>
    const canvas = document.getElementById("chart");
    const ctx = canvas.getContext("2d");
    const runSelect = document.getElementById("runSelect");
    let selected = "latest";

    document.getElementById("latestBtn").onclick = () => {
      selected = "latest";
      refresh();
    };
    runSelect.onchange = () => {
      selected = runSelect.value;
      refresh();
    };

    function fitCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function draw(points) {
      fitCanvas();
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      ctx.clearRect(0, 0, w, h);
      ctx.font = "12px Segoe UI, Arial";

      const pad = { l: 52, r: 18, t: 18, b: 42 };
      const xs = points.map(p => p.step);
      const ys = points.flatMap(p => [p.train_loss, p.val_loss]).filter(Number.isFinite);
      if (!points.length || !ys.length) {
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--muted");
        ctx.fillText("waiting for train_log.jsonl points", pad.l, pad.t + 22);
        return;
      }
      let minX = Math.min(...xs), maxX = Math.max(...xs);
      let minY = Math.min(...ys), maxY = Math.max(...ys);
      if (minX === maxX) maxX += 1;
      if (minY === maxY) maxY += 1;
      const yPad = (maxY - minY) * 0.08;
      minY -= yPad; maxY += yPad;

      const sx = x => pad.l + (x - minX) / (maxX - minX) * (w - pad.l - pad.r);
      const sy = y => pad.t + (maxY - y) / (maxY - minY) * (h - pad.t - pad.b);
      const lineColor = getComputedStyle(document.documentElement).getPropertyValue("--line");
      const muted = getComputedStyle(document.documentElement).getPropertyValue("--muted");

      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.l, pad.t);
      ctx.lineTo(pad.l, h - pad.b);
      ctx.lineTo(w - pad.r, h - pad.b);
      ctx.stroke();

      ctx.fillStyle = muted;
      for (let i = 0; i <= 4; i++) {
        const y = minY + (maxY - minY) * i / 4;
        const py = sy(y);
        ctx.strokeStyle = lineColor;
        ctx.beginPath();
        ctx.moveTo(pad.l, py);
        ctx.lineTo(w - pad.r, py);
        ctx.stroke();
        ctx.fillText(y.toFixed(2), 8, py + 4);
      }
      for (let i = 0; i <= 5; i++) {
        const x = Math.round(minX + (maxX - minX) * i / 5);
        const px = sx(x);
        ctx.fillText(String(x), px - 12, h - 14);
      }

      function plot(key, color) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        points.forEach((p, i) => {
          const x = sx(p.step);
          const y = sy(p[key]);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        const last = points[points.length - 1];
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(sx(last.step), sy(last[key]), 4, 0, Math.PI * 2);
        ctx.fill();
      }
      const styles = getComputedStyle(document.documentElement);
      plot("train_loss", styles.getPropertyValue("--train"));
      plot("val_loss", styles.getPropertyValue("--val"));
    }

    async function refresh() {
      const response = await fetch("/api/loss", { cache: "no-store" });
      const data = await response.json();
      const runs = data.runs || [];
      runSelect.innerHTML = "";
      runs.forEach(run => {
        const option = document.createElement("option");
        option.value = String(run.index);
        option.textContent = "run " + run.index + " (" + run.points.length + " pts)";
        runSelect.appendChild(option);
      });
      if (!runs.length) {
        document.getElementById("status").textContent = "no points yet";
        draw([]);
        return;
      }
      const run = selected === "latest"
        ? runs[runs.length - 1]
        : runs.find(r => String(r.index) === selected) || runs[runs.length - 1];
      runSelect.value = String(run.index);
      const points = run.points;
      const last = points[points.length - 1];
      document.getElementById("step").textContent = last.step;
      document.getElementById("train").textContent = last.train_loss.toFixed(4);
      document.getElementById("val").textContent = last.val_loss.toFixed(4);
      document.getElementById("points").textContent = points.length;
      document.getElementById("status").textContent = "updated " + new Date().toLocaleTimeString();
      draw(points);
    }

    window.addEventListener("resize", refresh);
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>`;

const server = http.createServer((req, res) => {
  if (req.url === "/api/loss") {
    jsonResponse(res, {
      logPath,
      runs: readRuns(),
      updatedAt: new Date().toISOString(),
    });
    return;
  }
  res.writeHead(200, {
    "content-type": "text/html; charset=utf-8",
    "cache-control": "no-store",
  });
  res.end(html);
});

server.listen(port, "127.0.0.1", () => {
  console.log(`Live loss dashboard: http://127.0.0.1:${port}`);
  console.log(`Reading: ${logPath}`);
});
