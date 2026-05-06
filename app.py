#!/usr/bin/env python3
"""
Background removal API — FastAPI + Databricks Model Serving (GPU).

The heavy inference (rembg u2net) runs on a GPU-backed Databricks Model
Serving endpoint. This app handles HTTP, image encoding, cropping, scaling,
and canvas composition on CPU serverless compute.

POST /remove-background

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Parameter                         Default   Range / Notes               │
  ├─────────────────────────────────────────────────────────────────────────┤
  │ image                (file)        —         Any PIL-readable format     │
  │                                                                         │
  │ whitespace_percent   (float)       15        0–99                        │
  │   How much empty space (padding) to leave around the subject.           │
  │   15 means the subject fills 85% of the output canvas.                 │
  │   Lower = subject fills more of the image; higher = more breathing room.│
  │                                                                         │
  │ bg_color             (string)      "white"   CSS name or hex (#ff0000)   │
  │   Background colour of the output image.                                │
  │   When set, output is a flat-colour JPEG.                               │
  │   Pass empty / omit for a transparent PNG.                              │
  │                                                                         │
  │ output_size          (int)         1800      ≥ 1                         │
  │   Output canvas size in pixels (square: output_size × output_size).     │
  │   The subject is centred and scaled to fit within the canvas.           │
  │   Omit to preserve the original image dimensions (non-square).         │
  │                                                                         │
  │ feathering           (float)       0         0–20                        │
  │   Softens the edges of the cutout by blurring the alpha channel.        │
  │   0 = hard edges exactly as the model produced them.                    │
  │   Higher values make edges progressively softer / more blended.        │
  │   Useful for compositing onto new backgrounds.                          │
  │                                                                         │
  │ alpha_threshold      (int)         0         0–254                       │
  │   Clips semi-transparent fringe pixels to fully transparent.            │
  │   0 = keep every pixel the model produced, including uncertain edges.   │
  │   Higher values cut the halo/fringe more aggressively.                 │
  │   Try 15–40 to remove colour bleeding around the subject edges.         │
  │                                                                         │
  │ alpha_matting        (bool)        false     true / false                │
  │   Enables alpha matting for significantly tighter, more accurate edges. │
  │   rembg outputs a confidence score per pixel. Pixels near the subject   │
  │   edge get intermediate values (e.g. 40–180) because the model is       │
  │   uncertain. Alpha matting re-examines those uncertain boundary pixels  │
  │   using the original image colour — "does this pixel look more like the │
  │   subject interior or the background?" — and reassigns a more accurate  │
  │   alpha value. Best when the background is a distinct colour.           │
  │   Slower than standard mode (~2–5× longer inference time).              │
  │                                                                         │
  │ alpha_matting_foreground_threshold (int)  240   1–255                   │
  │   Pixels with confidence ≥ this are treated as definite foreground.     │
  │   Lower = more pixels classified as "definitely subject".               │
  │                                                                         │
  │ alpha_matting_background_threshold (int)   10   0–254                   │
  │   Pixels with confidence ≤ this are treated as definite background.     │
  │   Higher = more pixels classified as "definitely background".           │
  │                                                                         │
  │ alpha_matting_erode_size           (int)   10   0–30                    │
  │   How far to shrink the uncertain boundary zone before matting.         │
  │   Higher = tighter initial mask; lower = more pixels re-examined.      │
  └─────────────────────────────────────────────────────────────────────────┘

GET /health  →  {"status": "ok"}

Required environment variables:
    REMBG_SERVING_ENDPOINT     Model Serving endpoint name (default: rembg-u2net)
    DATABRICKS_HOST / auth     Auto-handled by Databricks App runtime
"""

import base64
import io
import os
from typing import Optional

from databricks.sdk import WorkspaceClient
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageColor, ImageFilter

app = FastAPI(title="Image Background Remover API", version="2.0.0")

# ── Serving endpoint config ───────────────────────────────────────────────────
_ENDPOINT = os.environ.get("REMBG_SERVING_ENDPOINT", "rembg-u2net")
_client   = WorkspaceClient()  # auto-authenticates inside a Databricks App

print(f"Inference endpoint: {_ENDPOINT}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps — each function does one thing
# ─────────────────────────────────────────────────────────────────────────────

def call_rembg(
    img: Image.Image,
    alpha_matting: bool,
    alpha_matting_foreground_threshold: int,
    alpha_matting_background_threshold: int,
    alpha_matting_erode_size: int,
) -> Image.Image:
    """Send image to the GPU Model Serving endpoint; return RGBA result."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    try:
        response = _client.serving_endpoints.query(
            name=_ENDPOINT,
            dataframe_records=[{
                "image_b64": base64.b64encode(buf.getvalue()).decode(),
                "alpha_matting": alpha_matting,
                "alpha_matting_foreground_threshold": alpha_matting_foreground_threshold,
                "alpha_matting_background_threshold": alpha_matting_background_threshold,
                "alpha_matting_erode_size": alpha_matting_erode_size,
            }],
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Background removal service unavailable: {e}",
        )
    rgba_b64 = response.predictions[0]["rgba_b64"]
    return Image.open(io.BytesIO(base64.b64decode(rgba_b64)))


def apply_alpha_post_processing(
    img: Image.Image,
    feathering: float,
    alpha_threshold: int,
) -> Image.Image:
    """Apply feathering and/or threshold clipping to the alpha channel."""
    if feathering == 0.0 and alpha_threshold == 0:
        return img
    r, g, b, a = img.split()
    if feathering > 0.0:
        a = a.filter(ImageFilter.GaussianBlur(radius=feathering))
    if alpha_threshold > 0:
        a = a.point(lambda p: 0 if p <= alpha_threshold else p)
    return Image.merge("RGBA", (r, g, b, a))


def scale_subject_onto_canvas(
    subject: Image.Image,
    canvas_size: tuple,
    whitespace_percent: float,
) -> tuple:
    """Scale subject to fill (100 - whitespace_percent)% of canvas; return (resized, paste_offset)."""
    canvas_w, canvas_h = canvas_size
    w_s, h_s = subject.size
    target_fill = 1.0 - whitespace_percent / 100.0
    scale = min((canvas_w * target_fill) / w_s, (canvas_h * target_fill) / h_s)
    new_w = max(1, round(w_s * scale))
    new_h = max(1, round(h_s * scale))
    resized = subject.resize((new_w, new_h), Image.LANCZOS)
    paste_x = (canvas_w - new_w) // 2
    paste_y = (canvas_h - new_h) // 2
    return resized, (paste_x, paste_y)


def build_canvas(canvas_size: tuple, bg_color: Optional[str]) -> Image.Image:
    """Create a blank canvas — flat colour (RGBA) or transparent."""
    w, h = canvas_size
    if bg_color:
        try:
            r, g, b = ImageColor.getrgb(bg_color)
            return Image.new("RGBA", (w, h), (r, g, b, 255))
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid bg_color: {bg_color!r}. Use a CSS name or hex, e.g. #ff0000.",
            )
    return Image.new("RGBA", (w, h), (0, 0, 0, 0))


def encode_output(canvas: Image.Image, bg_color: Optional[str]) -> Response:
    """Encode canvas to JPEG (flat bg) or PNG (transparent) and return HTTP response."""
    buf = io.BytesIO()
    if bg_color:
        canvas.convert("RGB").save(buf, format="JPEG", quality=95, optimize=True)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    canvas.save(buf, format="PNG", optimize=True)
    return Response(content=buf.getvalue(), media_type="image/png")


def validate_params(
    whitespace_percent: float,
    output_size: Optional[int],
    feathering: float,
    alpha_threshold: int,
    alpha_matting_foreground_threshold: int,
    alpha_matting_background_threshold: int,
    alpha_matting_erode_size: int,
) -> None:
    """Raise 422 for any out-of-range parameter."""
    if not (0.0 <= whitespace_percent < 100.0):
        raise HTTPException(status_code=422, detail="whitespace_percent must be >= 0 and < 100")
    if output_size is not None and output_size < 1:
        raise HTTPException(status_code=422, detail="output_size must be a positive integer")
    if not (0.0 <= feathering <= 20.0):
        raise HTTPException(status_code=422, detail="feathering must be between 0 and 20")
    if not (0 <= alpha_threshold <= 254):
        raise HTTPException(status_code=422, detail="alpha_threshold must be between 0 and 254")
    if not (1 <= alpha_matting_foreground_threshold <= 255):
        raise HTTPException(status_code=422, detail="alpha_matting_foreground_threshold must be between 1 and 255")
    if not (0 <= alpha_matting_background_threshold <= 254):
        raise HTTPException(status_code=422, detail="alpha_matting_background_threshold must be between 0 and 254")
    if not (0 <= alpha_matting_erode_size <= 30):
        raise HTTPException(status_code=422, detail="alpha_matting_erode_size must be between 0 and 30")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/remove-background")
async def remove_background(
    image: UploadFile = File(..., description="Input image (JPEG, PNG, WebP, or any PIL-readable format)."),
    whitespace_percent: float = Form(
        default=15.0,
        description=(
            "Padding to leave around the subject (0–99). "
            "15 means the subject fills 85% of the canvas. "
            "Lower = subject larger; higher = more breathing room."
        ),
    ),
    bg_color: Optional[str] = Form(
        default="white",
        description=(
            "Background colour. Accepts CSS names (white, red) or hex (#ff0000). "
            "Output is a JPEG when set. Pass empty for a transparent PNG."
        ),
    ),
    output_size: Optional[int] = Form(
        default=1800,
        description=(
            "Output canvas size in pixels (square: output_size x output_size). "
            "Subject is scaled to fit and centred. Pass empty to keep original dimensions."
        ),
    ),
    feathering: float = Form(
        default=0.0,
        description=(
            "Softens cutout edges by blurring the alpha channel (0–20). "
            "0 = hard edges. Higher = softer, more gradual edge blend. "
            "Useful when compositing onto a new background."
        ),
    ),
    alpha_threshold: int = Form(
        default=0,
        description=(
            "Clips semi-transparent fringe pixels to fully transparent (0–254). "
            "0 = keep all model output. Higher cuts halo/colour bleed more aggressively. "
            "Try 15–40 to remove a thin outline around the subject."
        ),
    ),
    alpha_matting: bool = Form(
        default=True,
        description=(
            "Enable alpha matting for tighter, more accurate edges. "
            "Re-examines uncertain boundary pixels using original image colours. "
            "Best when subject and background have distinct colours. ~2-5x slower."
        ),
    ),
    alpha_matting_foreground_threshold: int = Form(
        default=240,
        description=(
            "Used when alpha_matting=true. Pixels with confidence >= this are definite foreground (1–255). "
            "Lower = more pixels classified as 'definitely subject'."
        ),
    ),
    alpha_matting_background_threshold: int = Form(
        default=30,
        description=(
            "Used when alpha_matting=true. Pixels with confidence <= this are definite background (0–254). "
            "Higher = more pixels classified as 'definitely background'."
        ),
    ),
    alpha_matting_erode_size: int = Form(
        default=1,
        description=(
            "Used when alpha_matting=true. Shrinks the uncertain boundary zone before matting (0–30). "
            "Higher = tighter initial mask; lower = more pixels re-examined."
        ),
    ),
):
    validate_params(
        whitespace_percent, output_size, feathering, alpha_threshold,
        alpha_matting_foreground_threshold, alpha_matting_background_threshold,
        alpha_matting_erode_size,
    )

    # 1. Read and decode the uploaded image
    raw = await image.read()
    try:
        input_img = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {exc}")

    original_w, original_h = input_img.size

    # 2. Remove background via GPU serving endpoint
    result_rgba = call_rembg(
        input_img,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
        alpha_matting_background_threshold=alpha_matting_background_threshold,
        alpha_matting_erode_size=alpha_matting_erode_size,
    )

    # 3. Apply feathering / alpha threshold post-processing
    result_rgba = apply_alpha_post_processing(result_rgba, feathering, alpha_threshold)

    # 4. Tight-crop to the bounding box of the subject
    bbox = result_rgba.getbbox()
    if bbox is None:
        canvas_size = (output_size, output_size) if output_size else (original_w, original_h)
        return encode_output(build_canvas(canvas_size, bg_color), bg_color)

    subject = result_rgba.crop(bbox)

    # 5. Determine canvas size (square if output_size set, else original dimensions)
    canvas_size = (output_size, output_size) if output_size else (original_w, original_h)

    # 6. Scale subject to fill canvas (respecting whitespace_percent)
    subject_resized, paste_offset = scale_subject_onto_canvas(subject, canvas_size, whitespace_percent)

    # 7. Build canvas and paste subject centred
    canvas = build_canvas(canvas_size, bg_color)
    canvas.paste(subject_resized, paste_offset, subject_resized)

    # 8. Encode and return
    return encode_output(canvas, bg_color)
