# Databricks notebook source
# Register rembg MLflow model for GPU Model Serving
#
# Run this notebook ONCE on a GPU cluster (e.g. your existing image-background-remover
# cluster) to log and register the u2net model in the Model Registry.
#
# After registration, create a GPU-backed serving endpoint pointing to the model.

# COMMAND ----------

# %pip install rembg[gpu] onnxruntime-gpu pillow mlflow --quiet
# dbutils.library.restartPython()

# COMMAND ----------

import base64
import io

import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema


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

# COMMAND ----------

# ── Config — update these to match your Unity Catalog ────────────────────────
# Run `display(spark.sql("SHOW CATALOGS"))` to find your available catalogs.
# Run `display(spark.sql("SHOW SCHEMAS IN <catalog>"))` to find schemas.
CATALOG         = "ai_squad_np"   # Unity Catalog
SCHEMA          = "default"
MODEL_NAME      = "rembg_u2net"
REGISTERED_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
EXPERIMENT_PATH = "/ML_ai_squad/nrix/image-background-remover"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema ready: {CATALOG}.{SCHEMA}")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_PATH)

_SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", "image_b64")]),
    outputs=Schema([ColSpec("string", "rgba_b64")]),
)

with mlflow.start_run(run_name="rembg-u2net-registration"):
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=RembgModel(),
        pip_requirements=[
            "rembg[gpu]",
            "onnxruntime-gpu",
            "pillow",
        ],
        signature=_SIGNATURE,
        registered_model_name=REGISTERED_NAME,
    )

print(f"Registered: {REGISTERED_NAME}")
print(f"Version:    {model_info.registered_model_version}")

# COMMAND ----------

print("=" * 60)
print("NEXT STEPS")
print("=" * 60)
print()
print("1. Go to Databricks UI → Serving → Create endpoint")
print(f"   Model:         {REGISTERED_NAME}")
print(f"   Version:       {model_info.registered_model_version}")
print("   Instance type: GPU_MEDIUM (A10) or GPU_LARGE")
print("   Endpoint name: rembg-u2net  (or your preferred name)")
print()
print("2. Add these environment variables to your Databricks App:")
print("   DATABRICKS_HOST            https://<workspace>.azuredatabricks.net")
print("   DATABRICKS_TOKEN           <personal access token or SP secret>")
print("   REMBG_SERVING_ENDPOINT     rembg-u2net  (must match endpoint name above)")
print()
print("3. Re-run  .\\deploy\\deploy.ps1  to deploy the updated app.")
