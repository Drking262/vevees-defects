"""Local, dependency-free web tool for drawing bounding boxes on photos.

YOLO detection needs box coordinates, and none exist in data/ (photos are
only labeled by filename). With ~10-16 photos, a tiny local HTTP server +
canvas annotator is faster than pulling in CVAT/labelImg.

Draws *multiple* boxes per photo (a photo can show more than one defect,
e.g. a knot plus scattered insect holes), each independently classed
(defaults to the filename-derived class, changeable per box).

Saved to annotations/boxes.json as
{image_path: [{"box": [x1,y1,x2,y2], "cls": "..."}, ...]}, normalized 0-1
coords (resolution independent). Real manual work product -- not
git-ignored.

Usage:
    python -m pipeline.annotate
    # then open http://localhost:8765 in a browser, draw boxes per photo
"""

from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from pipeline.config import DETECT_CLASS_NAMES, INCLUDED_CLASSES
from pipeline.dataset_scan import dedupe, discover

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_PATH = REPO_ROOT / "annotations" / "boxes.json"
DISPLAY_MAX_SIDE = 900


def build_image_list() -> list[dict]:
    by_class = discover(INCLUDED_CLASSES)
    items = []
    for cls, paths in by_class.items():
        for p in dedupe(paths, verbose=False):
            rel = str(p.relative_to(REPO_ROOT))
            items.append({"path": rel, "cls": cls})
    items.sort(key=lambda d: d["path"])
    return items


def load_annotations() -> dict:
    if ANNOTATIONS_PATH.exists():
        return json.loads(ANNOTATIONS_PATH.read_text())
    return {}


def save_annotations(data: dict) -> None:
    ANNOTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Defect box annotator</title>
<style>
  body { font-family: sans-serif; background: #222; color: #eee; margin: 0; padding: 16px; }
  #wrap { display: flex; flex-direction: column; align-items: center; gap: 10px; }
  #info { font-size: 16px; }
  #canvasWrap { position: relative; border: 2px solid #555; }
  canvas { display: block; cursor: crosshair; }
  button { font-size: 15px; padding: 6px 14px; margin: 0 4px; cursor: pointer; }
  #status { color: #8f8; min-height: 20px; }
  #list { font-size: 12px; color: #999; max-width: 700px; text-align: center; }
  #boxList { display: flex; flex-direction: column; gap: 4px; min-width: 420px; }
  .boxRow { display: flex; align-items: center; gap: 8px; background: #333; padding: 4px 8px; border-radius: 4px; }
  .swatch { width: 14px; height: 14px; border-radius: 3px; flex-shrink: 0; }
  .boxRow select { flex: 1; }
  .hint { color: #999; font-size: 13px; }
</style>
</head>
<body>
<div id="wrap">
  <div id="info"></div>
  <div class="hint">Click-drag to add a box. Draw as many as the photo actually shows.</div>
  <div id="canvasWrap"><canvas id="cv"></canvas></div>
  <div id="boxList"></div>
  <div>
    <button onclick="prevImg()">&larr; Prev</button>
    <button onclick="clearBoxes()">Clear all boxes</button>
    <button onclick="saveAndNext()">Save &amp; Next &rarr;</button>
  </div>
  <div id="status"></div>
  <div id="list"></div>
</div>
<script>
const items = ITEMS_JSON;
const classNames = CLASS_NAMES_JSON;
const colors = ['#00ff66', '#ff6b6b', '#4dabf7', '#ffd43b', '#da77f2', '#ff922b'];
let idx = 0;
let img = new Image();
let boxes = [];      // [{x1,y1,x2,y2,cls}] in canvas pixel coords
let dragStart = null;
const canvas = document.getElementById('cv');
const ctx = canvas.getContext('2d');

function colorFor(cls) {
  const i = classNames.indexOf(cls);
  return colors[i >= 0 ? i % colors.length : 0];
}

// The photo's filename-derived class (e.g. "cerny_soucek_drevokazny_hmyz")
// can be coarser than the detection class list (e.g. split into
// "cerny_soucek" and "drevokazny_hmyz") -- fall back to the first detect
// class rather than prefilling something that isn't a real option.
function defaultClassFor(it) {
  return classNames.includes(it.cls) ? it.cls : classNames[0];
}

async function loadExisting() {
  const r = await fetch('/existing?idx=' + idx);
  return r.json();
}

async function loadImage() {
  const it = items[idx];
  document.getElementById('info').textContent =
    `[${idx+1}/${items.length}] ${it.path}  (default class: ${defaultClassFor(it)}, filename says: ${it.cls})`;
  img = new Image();
  img.onload = async () => {
    canvas.width = img.width;
    canvas.height = img.height;
    const existing = await loadExisting();
    boxes = (existing.boxes || []).map(b => ({
      x1: b.box[0]*img.width, y1: b.box[1]*img.height,
      x2: b.box[2]*img.width, y2: b.box[3]*img.height,
      cls: b.cls
    }));
    render();
  };
  img.src = '/img?idx=' + idx;
  document.getElementById('status').textContent = '';
}

function draw() {
  ctx.drawImage(img, 0, 0);
  for (const b of boxes) {
    ctx.strokeStyle = colorFor(b.cls);
    ctx.lineWidth = 3;
    ctx.strokeRect(b.x1, b.y1, b.x2-b.x1, b.y2-b.y1);
  }
}

function renderBoxList() {
  const el = document.getElementById('boxList');
  el.innerHTML = '';
  boxes.forEach((b, i) => {
    const row = document.createElement('div');
    row.className = 'boxRow';
    const sw = document.createElement('div');
    sw.className = 'swatch';
    sw.style.background = colorFor(b.cls);
    const sel = document.createElement('select');
    for (const c of classNames) {
      const opt = document.createElement('option');
      opt.value = c; opt.textContent = c;
      if (c === b.cls) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.onchange = () => { b.cls = sel.value; draw(); renderBoxList(); };
    const del = document.createElement('button');
    del.textContent = 'Delete';
    del.onclick = () => { boxes.splice(i, 1); render(); };
    row.appendChild(sw); row.appendChild(sel); row.appendChild(del);
    el.appendChild(row);
  });
}

function render() { draw(); renderBoxList(); }

canvas.addEventListener('mousedown', e => {
  const r = canvas.getBoundingClientRect();
  dragStart = { x: (e.clientX - r.left) * canvas.width / r.width,
                y: (e.clientY - r.top) * canvas.height / r.height };
});
canvas.addEventListener('mousemove', e => {
  if (!dragStart) return;
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * canvas.width / r.width;
  const y = (e.clientY - r.top) * canvas.height / r.height;
  draw();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.strokeRect(Math.min(dragStart.x, x), Math.min(dragStart.y, y),
                 Math.abs(x - dragStart.x), Math.abs(y - dragStart.y));
  ctx.setLineDash([]);
});
window.addEventListener('mouseup', e => {
  if (!dragStart) return;
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) * canvas.width / r.width;
  const y = (e.clientY - r.top) * canvas.height / r.height;
  const x1 = Math.min(dragStart.x, x), x2 = Math.max(dragStart.x, x);
  const y1 = Math.min(dragStart.y, y), y2 = Math.max(dragStart.y, y);
  dragStart = null;
  if (x2 - x1 > 3 && y2 - y1 > 3) {
    boxes.push({ x1, y1, x2, y2, cls: defaultClassFor(items[idx]) });
    render();
  } else {
    draw();
  }
});

function clearBoxes() { boxes = []; render(); }

function prevImg() {
  if (idx > 0) { idx--; loadImage(); }
}

async function saveAndNext() {
  if (boxes.length === 0) {
    document.getElementById('status').textContent =
      'Draw at least one box first (click-drag on the image), or use Clear to explicitly skip.';
    return;
  }
  const payloadBoxes = boxes.map(b => ({
    box: [b.x1/img.width, b.y1/img.height, b.x2/img.width, b.y2/img.height],
    cls: b.cls
  }));
  await fetch('/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({idx, boxes: payloadBoxes})
  });
  if (idx < items.length - 1) {
    idx++;
    loadImage();
  } else {
    document.getElementById('status').textContent = 'All images annotated. You can close this tab.';
  }
}

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') saveAndNext();
  if (e.key === 'ArrowLeft') prevImg();
});

document.getElementById('list').textContent =
  items.map((it, i) => it.path.split('/').pop()).join('  \\u00b7  ');
loadImage();
</script>
</body>
</html>
"""


def make_handler(items: list[dict]):
    annotations = load_annotations()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # keep stdout quiet

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html = PAGE_TEMPLATE.replace("ITEMS_JSON", json.dumps(items)).replace(
                    "CLASS_NAMES_JSON", json.dumps(DETECT_CLASS_NAMES)
                )
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            qs = dict(x.split("=") for x in (parsed.query or "").split("&") if "=" in x)
            idx = int(qs.get("idx", 0))

            if parsed.path == "/img":
                src = REPO_ROOT / items[idx]["path"]
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    scale = DISPLAY_MAX_SIDE / max(w, h)
                    if scale < 1:
                        im = im.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=90)
                body = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if parsed.path == "/existing":
                path = items[idx]["path"]
                self._send_json({"boxes": annotations.get(path, [])})
                return

            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            if self.path == "/save":
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                path = items[payload["idx"]]["path"]
                annotations[path] = payload["boxes"]
                save_annotations(annotations)
                self._send_json({"ok": True})
                return
            self.send_response(404)
            self.end_headers()

    return Handler


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    items = build_image_list()
    if not items:
        raise SystemExit("no images found for pipeline.config.INCLUDED_CLASSES")

    done = sum(1 for it in items if it["path"] in load_annotations())
    print(f"{done}/{len(items)} images already have saved boxes (from a previous session)")
    print(f"Open http://localhost:{args.port} in a browser to annotate.")
    print(
        "Click-drag to add a box, pick its class from the dropdown below the image "
        "(a photo can have several boxes/classes), then 'Save & Next'. Ctrl+C here to stop."
    )

    server = ThreadingHTTPServer(("localhost", args.port), make_handler(items))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
