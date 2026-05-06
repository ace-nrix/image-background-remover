#!/bin/bash
# cluster_init.sh
# Databricks GPU cluster init script — Image Background Remover API
#
# Runs on the DRIVER node only. Writes the FastAPI app to /tmp, installs
# dependencies into a local venv (reused across restarts), and starts
# uvicorn on port 8188.
#
# Logs:
#   /tmp/rembg_api_init.log  — init script output
#   /tmp/rembg_api.log       — uvicorn runtime output
#
# Access via Databricks driver-proxy:
#   https://<workspace>.azuredatabricks.net/driver-proxy/o/<orgId>/<clusterId>/8188/

set -e
if [[ $DB_IS_DRIVER != "TRUE" ]]; then
  exit 0
fi

# Tee output to notebook cell AND log file
exec > >(tee /tmp/rembg_api_init.log) 2>&1
echo "== RemBG API init starting: $(date)"

# ── Config ────────────────────────────────────────────────────────────────────
API_DIR=/tmp/rembg_api
VENV=/tmp/rembg_venv

mkdir -p "${API_DIR}"

# ── Write the FastAPI application ─────────────────────────────────────────────
echo "== Writing app.py"
cat > "${API_DIR}/app.py" << 'PYEOF'
#!/usr/bin/env python3
import io
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageColor
from rembg import new_session, remove

app = FastAPI(title="Image Background Remover API", version="1.0.0")

print("Loading rembg model (u2net) ...", flush=True)
_session = new_session(
    "u2net",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print("Model loaded.", flush=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/remove-background")
async def remove_background(
    image: UploadFile = File(...),
    whitespace_percent: float = Form(default=10.0),
    bg_color: Optional[str] = Form(default=None),
):
    if not (0.0 <= whitespace_percent < 100.0):
        raise HTTPException(status_code=422, detail="whitespace_percent must be >= 0 and < 100")

    raw = await image.read()
    try:
        input_img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}")

    W, H = input_img.size
    result_rgba = remove(input_img, session=_session)

    bbox = result_rgba.getbbox()
    if bbox is None:
        subject_resized = None
        new_w = new_h = 0
    else:
        subject = result_rgba.crop(bbox)
        w_s, h_s = subject.size
        target_fill = 1.0 - whitespace_percent / 100.0
        scale = min((W * target_fill) / w_s, (H * target_fill) / h_s)
        new_w = max(1, round(w_s * scale))
        new_h = max(1, round(h_s * scale))
        subject_resized = subject.resize((new_w, new_h), Image.LANCZOS)

    if bg_color:
        try:
            r, g, b = ImageColor.getrgb(bg_color)
            canvas = Image.new("RGBA", (W, H), (r, g, b, 255))
        except (ValueError, AttributeError):
            raise HTTPException(status_code=422, detail=f"Invalid bg_color: {bg_color!r}")
    else:
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    if subject_resized is not None:
        paste_x = (W - new_w) // 2
        paste_y = (H - new_h) // 2
        canvas.paste(subject_resized, (paste_x, paste_y), subject_resized)

    buf = io.BytesIO()
    if bg_color:
        canvas.convert("RGB").save(buf, format="JPEG", quality=95, optimize=True)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    else:
        canvas.save(buf, format="PNG", optimize=True)
        return Response(content=buf.getvalue(), media_type="image/png")
PYEOF
echo "== app.py written"

# ── System packages ───────────────────────────────────────────────────────────
echo "== Installing system dependencies"
apt-get update -q || true
apt-get install -y -q python3-venv python3-pip libgl1 libglib2.0-0 || true

# ── Python venv (reused across restarts — lives on local ephemeral disk) ─────
if [ ! -f "${VENV}/bin/activate" ]; then
  echo "== Creating venv at ${VENV}"
  python3 -m venv "${VENV}"
  if [ ! -f "${VENV}/bin/activate" ]; then
    echo "ERROR: venv creation failed"; exit 1
  fi

  source "${VENV}/bin/activate"

  echo "== Installing Python packages (first-time setup, may take a few minutes)"
  pip install --upgrade pip setuptools wheel
  # rembg[gpu] pulls onnxruntime-gpu for CUDA; falls back to CPU automatically
  pip install "rembg[gpu]" || pip install rembg
  pip install "fastapi" "uvicorn[standard]" "pillow"
else
  echo "== Venv already exists, skipping install"
  source "${VENV}/bin/activate"
fi

# ── Launch API ────────────────────────────────────────────────────────────────
pkill -f uvicorn || true
sleep 1
echo "== Launching RemBG API on port 8188 (logs: /tmp/rembg_api.log)"
nohup "${VENV}/bin/uvicorn" app:app \
  --host 0.0.0.0 \
  --port 8188 \
  --workers 1 \
  --app-dir "${API_DIR}" \
  > /tmp/rembg_api.log 2>&1 &

sleep 4
if pgrep -f uvicorn >/dev/null 2>&1; then
  echo "== RemBG API started OK (pid $(pgrep -f uvicorn | head -n1))"
  echo "== Endpoint: POST http://0.0.0.0:8188/remove-background"
else
  echo "ERROR: API failed to start — last 50 lines of /tmp/rembg_api.log:"
  tail -n 50 /tmp/rembg_api.log || true
  exit 1
fi

echo "== Init completed: $(date)"
exit 0
