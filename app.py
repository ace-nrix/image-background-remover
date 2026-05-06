#!/usr/bin/env python3
"""
Background removal API — FastAPI + Databricks Model Serving (GPU).

The heavy inference (rembg u2net) runs on a GPU-backed Databricks Model
Serving endpoint. This app handles HTTP, image encoding, cropping, scaling,
and canvas composition on CPU serverless compute.

POST /remove-background
    image              : (file)   input image, any format PIL can read
    whitespace_percent : (float)  padding around subject as % of output canvas, default 10
                                  e.g. 15 → subject fills 85% of the output image
    bg_color           : (string) optional background fill, e.g. "white" or "#ff0000"
                                  omit entirely for a transparent PNG output
    output_size        : (int)    optional square output canvas in pixels, e.g. 1800
                                  crops a square region centred on the subject then
                                  scales to output_size × output_size
                                  omit to preserve the original image dimensions
    feathering         : (float)  Gaussian blur radius applied to the alpha channel, default 0
                                  0 = hard edges as returned by rembg
                                  range 0–20 (practical); higher = softer/more feathered edges
    alpha_threshold    : (int)    pixels with alpha ≤ this value are clipped to 0, default 0
                                  0 = keep all semi-transparent pixels rembg produced
                                  range 0–254; higher = harder cutoff, removes fringe pixels

GET /health
    → {"status": "ok"}

Required environment variables:
    DATABRICKS_HOST            Workspace URL, e.g. https://adb-....azuredatabricks.net
    DATABRICKS_TOKEN           Personal access token or service-principal secret
    REMBG_SERVING_ENDPOINT     Model Serving endpoint name (default: rembg-u2net)
"""

import base64
import io
import os
from typing import Optional

from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageColor

app = FastAPI(title="Image Background Remover API", version="2.0.0")

# ── Serving endpoint config ───────────────────────────────────────────────────
_ENDPOINT = os.environ.get("REMBG_SERVING_ENDPOINT", "rembg-u2net")
_client   = WorkspaceClient()  # auto-authenticates inside a Databricks App

print(f"Inference endpoint: {_ENDPOINT}", flush=True)


def _call_rembg(img: Image.Image) -> Image.Image:
    """Send image to the GPU Model Serving endpoint; return RGBA result."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    try:
        response = _client.serving_endpoints.query(
            name=_ENDPOINT,
            dataframe_records=[{"image_b64": base64.b64encode(buf.getvalue()).decode()}],
        )
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Background removal service unavailable: {e}"
        )
    rgba_b64 = response.predictions[0]["rgba_b64"]
    return Image.open(io.BytesIO(base64.b64decode(rgba_b64)))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/remove-background")
async def remove_background(
    image: UploadFile = File(...),
    whitespace_percent: float = Form(default=10.0),
    bg_color: Optional[str] = Form(default=None),
    output_size: Optional[int] = Form(default=None),
    feathering: float = Form(default=0.0),
    alpha_threshold: int = Form(default=0),
):
    """
    Processing pipeline:
    1. Remove background from the uploaded image using rembg (u2net, CUDA).
    2. Tight-crop to the bounding box of the remaining subject.
    3. If output_size is set: determine the longest axis of the subject,
       build a square canvas of that size centred on the subject, then
       resize the whole square to output_size × output_size.
       Otherwise use the original image dimensions as the canvas.
    4. Scale the subject so it fills (100 - whitespace_percent)% of the canvas
       (longest axis, aspect ratio preserved).
    5. Centre the scaled subject on the canvas.
    6. Fill canvas with bg_color when provided (→ JPEG), else keep transparent (→ PNG).
    """
    if not (0.0 <= whitespace_percent < 100.0):
        raise HTTPException(
            status_code=422,
            detail="whitespace_percent must be >= 0 and < 100",
        )
    if output_size is not None and output_size < 1:
        raise HTTPException(
            status_code=422,
            detail="output_size must be a positive integer",
        )
    if not (0.0 <= feathering <= 20.0):
        raise HTTPException(
            status_code=422,
            detail="feathering must be between 0 and 20",
        )
    if not (0 <= alpha_threshold <= 254):
        raise HTTPException(
            status_code=422,
            detail="alpha_threshold must be between 0 and 254",
        )

    # ── Read input ────────────────────────────────────────────────────────────
    raw = await image.read()
    try:
        input_img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}")

    W, H = input_img.size

    # ── Remove background via GPU serving endpoint ──────────────────────────
    result_rgba: Image.Image = _call_rembg(input_img)

    # ── Alpha channel post-processing ─────────────────────────────────────────
    if feathering > 0.0 or alpha_threshold > 0:
        from PIL import ImageChops, ImageFilter
        r, g, b, a = result_rgba.split()
        if feathering > 0.0:
            a = a.filter(ImageFilter.GaussianBlur(radius=feathering))
        if alpha_threshold > 0:
            # Zero out any pixel at or below the threshold
            cutoff = a.point(lambda p: 0 if p <= alpha_threshold else p)
            a = cutoff
        result_rgba = Image.merge("RGBA", (r, g, b, a))

    # ── Tight-crop to subject ─────────────────────────────────────────────────
    bbox = result_rgba.getbbox()
    if bbox is None:
        subject_resized = None
        new_w = new_h = 0
    else:
        subject = result_rgba.crop(bbox)
        w_s, h_s = subject.size

        # ── Determine canvas size ─────────────────────────────────────────────
        if output_size is not None:
            # Square canvas: use the requested output_size for both axes
            C = output_size
        else:
            # Preserve original image dimensions
            C = None

        canvas_w = C if C is not None else W
        canvas_h = C if C is not None else H

        target_fill = 1.0 - whitespace_percent / 100.0
        scale = min((canvas_w * target_fill) / w_s, (canvas_h * target_fill) / h_s)
        new_w = max(1, round(w_s * scale))
        new_h = max(1, round(h_s * scale))
        subject_resized = subject.resize((new_w, new_h), Image.LANCZOS)

    # ── Build canvas ──────────────────────────────────────────────────────────
    canvas_w = output_size if output_size is not None else W
    canvas_h = output_size if output_size is not None else H
    if bg_color:
        try:
            r, g, b = ImageColor.getrgb(bg_color)
            canvas = Image.new("RGBA", (canvas_w, canvas_h), (r, g, b, 255))
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid bg_color: {bg_color!r}. Use a CSS name or hex, e.g. #ff0000.",
            )
    else:
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    # ── Paste subject centred ─────────────────────────────────────────────────
    if subject_resized is not None:
        paste_x = (canvas_w - new_w) // 2
        paste_y = (canvas_h - new_h) // 2
        canvas.paste(subject_resized, (paste_x, paste_y), subject_resized)

    # ── Encode and return ─────────────────────────────────────────────────────
    buf = io.BytesIO()
    if bg_color:
        canvas.convert("RGB").save(buf, format="JPEG", quality=95, optimize=True)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    else:
        canvas.save(buf, format="PNG", optimize=True)
        return Response(content=buf.getvalue(), media_type="image/png")
