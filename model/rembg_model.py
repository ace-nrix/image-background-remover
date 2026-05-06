"""
MLflow PythonModel wrapper for rembg u2net background removal.

Intended for Databricks Model Serving on a GPU instance.

Input:  pandas DataFrame with columns:
          image_b64                      — base64-encoded source image (any PIL-readable format)
          alpha_matting                  — bool, optional (default False)
          alpha_matting_foreground_threshold — int, optional (default 240)
          alpha_matting_background_threshold — int, optional (default 10)
          alpha_matting_erode_size       — int, optional (default 10)

Output: list of dicts with key 'rgba_b64'
        — base64-encoded PNG bytes of the RGBA result (background removed)
"""

import base64
import io

import mlflow.pyfunc


class RembgModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from rembg import new_session

        print("Loading u2net model…", flush=True)
        self._session = new_session(
            "u2net",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print("Model ready.", flush=True)

    def predict(self, context, model_input):
        from PIL import Image
        from rembg import remove

        results = []
        for _, row in model_input.iterrows():
            raw = base64.b64decode(row["image_b64"])
            img = Image.open(io.BytesIO(raw)).convert("RGBA")

            am      = bool(row.get("alpha_matting", False))
            am_fg   = int(row.get("alpha_matting_foreground_threshold", 240))
            am_bg   = int(row.get("alpha_matting_background_threshold", 10))
            am_er   = int(row.get("alpha_matting_erode_size", 10))

            result = remove(
                img,
                session=self._session,
                alpha_matting=am,
                alpha_matting_foreground_threshold=am_fg,
                alpha_matting_background_threshold=am_bg,
                alpha_matting_erode_size=am_er,
            )

            buf = io.BytesIO()
            result.save(buf, format="PNG")
            results.append({"rgba_b64": base64.b64encode(buf.getvalue()).decode()})

        return results
