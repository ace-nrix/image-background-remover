"""
MLflow PythonModel wrapper for rembg u2net background removal.

Intended for Databricks Model Serving on a GPU instance.

Input:  pandas DataFrame with column 'image_b64'
        — base64-encoded bytes of the source image (any PIL-readable format)

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
            result = remove(img, session=self._session)

            buf = io.BytesIO()
            result.save(buf, format="PNG")
            results.append({"rgba_b64": base64.b64encode(buf.getvalue()).decode()})

        return results
