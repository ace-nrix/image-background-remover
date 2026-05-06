#!/usr/bin/env python3
"""
Background removal API — FastAPI + rembg (CUDA).

POST /remove-background
    image              : (file)   input image, any format PIL can read
    whitespace_percent : (float)  padding around subject as % of output canvas, default 10
                                  e.g. 15 → subject fills 85% of the output image
    bg_color           : (string) optional background fill, e.g. "white" or "#ff0000"
                                  omit entirely for a transparent PNG output

GET /health
    → {"status": "ok"}
"""

import io
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageColor
from rembg import new_session, remove

app = FastAPI(title="Image Background Remover API", version="1.0.0")

# Load model once at startup (~170 MB, cached in ~/.u2net after first download)
print("Loading rembg model (u2net) …", flush=True)
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
    """
    Processing pipeline:
    1. Remove background from the uploaded image using rembg (u2net, CUDA).
    2. Tight-crop to the bounding box of the remaining subject.
    3. Scale the subject so it fills (100 - whitespace_percent)% of the
       original canvas dimensions (longest axis, aspect ratio preserved).
    4. Centre the scaled subject on a canvas matching the original image size.
    5. Fill canvas with bg_color when provided (→ JPEG), else keep transparent (→ PNG).
    """
    if not (0.0 <= whitespace_percent < 100.0):
        raise HTTPException(
            status_code=422,
            detail="whitespace_percent must be >= 0 and < 100",
        )

    # ── Read input ────────────────────────────────────────────────────────────
    raw = await image.read()
    try:
        input_img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}")

    W, H = input_img.size

    # ── Remove background ─────────────────────────────────────────────────────
    result_rgba: Image.Image = remove(input_img, session=_session)

    # ── Tight-crop to subject ─────────────────────────────────────────────────
    bbox = result_rgba.getbbox()
    if bbox is None:
        # Nothing detected — return blank canvas at original size
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

    # ── Build canvas ──────────────────────────────────────────────────────────
    if bg_color:
        try:
            r, g, b = ImageColor.getrgb(bg_color)
            canvas = Image.new("RGBA", (W, H), (r, g, b, 255))
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid bg_color: {bg_color!r}. Use a CSS name or hex, e.g. #ff0000.",
            )
    else:
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # ── Paste subject centred ─────────────────────────────────────────────────
    if subject_resized is not None:
        paste_x = (W - new_w) // 2
        paste_y = (H - new_h) // 2
        canvas.paste(subject_resized, (paste_x, paste_y), subject_resized)

    # ── Encode and return ─────────────────────────────────────────────────────
    buf = io.BytesIO()
    if bg_color:
        canvas.convert("RGB").save(buf, format="JPEG", quality=95, optimize=True)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    else:
        canvas.save(buf, format="PNG", optimize=True)
        return Response(content=buf.getvalue(), media_type="image/png")
