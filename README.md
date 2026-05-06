# Image Background Remover API

A FastAPI service that removes image backgrounds using [rembg](https://github.com/danielgatis/rembg) (u2net) with GPU acceleration. The app runs as a **Databricks App** (public, persistent URL) and delegates inference to a **GPU-backed Databricks Model Serving endpoint**. No GPU cluster needs to stay running.

## Architecture

```
User / Client
     │  HTTPS (public, auth via Databricks)
     ▼
Databricks App  (CPU serverless — FastAPI)
  • receives multipart upload
  • encodes image → base64
  • calls Model Serving endpoint via databricks-sdk
  • crops / scales / composites result
     │  HTTPS (internal, auto-authenticated)
     ▼
Databricks Model Serving  (GPU — rembg u2net)
  • runs ONNX inference on CUDA
  • returns RGBA PNG bytes
```

## API

```
POST /remove-background
```

| Field                | Type   | Required           | Description                                                           |
| -------------------- | ------ | ------------------ | --------------------------------------------------------------------- |
| `image`              | file   | yes                | Input image in any PIL-readable format                                |
| `whitespace_percent` | float  | no (default: `10`) | Padding around subject. `15` → subject fills 85% of canvas            |
| `bg_color`           | string | no                 | CSS color name (`white`) or hex (`#ff0000`). Omit for transparent PNG |

```
GET /health  →  {"status": "ok"}
```

### Example — curl

```bash
APP_URL="https://image-background-remover-<org>.azuredatabricks.net"

# Transparent PNG
curl -X POST "$APP_URL/remove-background" \
  -F "image=@photo.jpg" \
  -F "whitespace_percent=15" \
  -o result.png

# White background JPEG
curl -X POST "$APP_URL/remove-background" \
  -F "image=@photo.jpg" \
  -F "whitespace_percent=10" \
  -F "bg_color=white" \
  -o result.jpg
```

Interactive docs at `<APP_URL>/docs`.

---

## One-time setup

### Step 1 — Register the GPU model

Run `notebooks/01_register_model.py` **once** on a GPU cluster to log and register the u2net model in Unity Catalog.

1. Upload the repo to the workspace (run `.\deploy\deploy.ps1` — it will fail at Step 3 the first time if the app doesn't exist yet, that's fine for this step)
2. Open `notebooks/01_register_model.py` in the Databricks workspace
3. Attach it to a **GPU cluster** (any cluster with a GPU; the cluster used for `cluster_init.sh` works)
4. Uncomment and run the `%pip install` cell at the top, then restart Python
5. Edit `CATALOG` and `SCHEMA` to match your Unity Catalog. To discover available values:
    ```sql
    SHOW CATALOGS;
    SHOW SCHEMAS IN <catalog>;
    ```
    If your target schema doesn't exist, the notebook creates it automatically with `CREATE SCHEMA IF NOT EXISTS`.
6. Run all remaining cells — the model is logged to MLflow and registered as `<catalog>.<schema>.rembg_u2net`

> **Known gotchas:**
>
> - Unity Catalog names are **case-insensitive** and lowercased internally. `ML_ai_squad` → `ml_ai_squad`. Use the exact lowercased name.
> - Unity Catalog requires a **model signature**. The notebook includes one — do not remove it.
> - The u2net weights (~170 MB) are **not** stored during registration. They are downloaded on first inference by the serving endpoint and cached automatically.

After the notebook completes it prints the registered model name and version to use in Step 2.

---

### Step 2 — Create the GPU serving endpoint (Databricks UI)

1. Go to **Serving** (left nav) → **Create serving endpoint**
2. **Name:** `rembg-u2net`
3. **Entity:** Select the registered model — `ai_squad_np.default.rembg_u2net` (or your catalog/schema) → pick the latest version
4. **Compute:** `GPU_SMALL` (T4, cheaper) or `GPU_MEDIUM` (A10, faster)
5. Click **Create** — provisioning takes ~5–10 minutes

Monitor status:

```powershell
databricks serving-endpoints get rembg-u2net --profile nrix
```

Wait until `state` shows `READY` before proceeding.

> The serving endpoint has its own managed GPU compute and **scales to zero when idle**. You do not need to keep a GPU cluster running.

---

### Step 3 — Configure the Databricks App environment variable

In the Databricks UI → **Apps** → `image-background-remover` → **Settings** → **Environment variables**, add:

| Variable                 | Value         |
| ------------------------ | ------------- |
| `REMBG_SERVING_ENDPOINT` | `rembg-u2net` |

Authentication (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`) is **not required** — the `databricks-sdk` auto-authenticates using the App's built-in identity.

---

### Step 4 — Create and deploy the Databricks App

```powershell
# Create the app (once only — skip if it already exists)
databricks apps create image-background-remover --profile nrix

# Upload all source files and deploy
.\deploy\deploy.ps1
```

The deploy script:

- Creates the remote workspace directory
- Uploads all `.py`, `.txt`, `.yaml`, `.yml`, `.json`, `.sh` files (skipping `deploy/`, `.venv/`, `.git/`, `__pycache__/`)
- Runs `databricks apps deploy` against the `nrix` profile

After deployment, the script prints the app details including the public URL. You can also retrieve it any time with:

```powershell
databricks apps get image-background-remover --profile nrix
```

> **App start command** is defined in `app.yaml`:
>
> ```yaml
> command: ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
> ```
>
> Without this file, Databricks Apps defaults to `python app.py` which does nothing for a FastAPI app.

---

## Re-deploying after code changes

```powershell
.\deploy\deploy.ps1
```

To re-register a new version of the model (e.g. after changing `model/rembg_model.py`), re-run `notebooks/01_register_model.py` on a GPU cluster, then update the serving endpoint to point to the new version.

---

## Checking app logs

```powershell
databricks apps logs image-background-remover --tail-lines 100 --profile nrix
```

---

## Local development

```powershell
# 1. Create venv and install deps
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Set env vars (point at your live serving endpoint)
$env:DATABRICKS_HOST           = "https://adb-<orgid>.azuredatabricks.net"
$env:DATABRICKS_TOKEN          = "<your-pat>"
$env:REMBG_SERVING_ENDPOINT    = "rembg-u2net"

# 3. Start with hot-reload
.\deploy\dev.ps1
# or directly:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

If port 8000 is in use:

```powershell
.\deploy\killports.ps1
```

> For local dev, `DATABRICKS_HOST` and `DATABRICKS_TOKEN` are required because auto-auth only works inside a Databricks App.

---

## Legacy — direct GPU cluster access

The original approach runs the API directly on the cluster driver via `cluster_init.sh`. Useful for quick testing without setting up a serving endpoint.

1. Attach `cluster_init.sh` as an init script (Cluster → Advanced → Init Scripts)
2. Start the cluster — it installs deps and starts uvicorn on port 8188
3. Access via driver-proxy URL **(must be logged into Databricks in the same browser)**:
    ```
    https://adb-439895488707306.6.azuredatabricks.net/driver-proxy/o/439895488707306/<clusterId>/8188/health
    ```
    > The cluster ID changes every restart. The URL is not public or unauthenticated.

Logs:

```bash
%sh
tail -n 50 /tmp/rembg_api_init.log   # init script output
tail -n 50 /tmp/rembg_api.log        # uvicorn runtime output
```

---

## Files

| File                             | Description                                                            |
| -------------------------------- | ---------------------------------------------------------------------- |
| `app.py`                         | FastAPI app — HTTP handling, image compositing, calls serving endpoint |
| `app.yaml`                       | Databricks App start command (uvicorn)                                 |
| `requirements.txt`               | App dependencies (`fastapi`, `uvicorn`, `pillow`, `databricks-sdk`)    |
| `model/rembg_model.py`           | MLflow `PythonModel` wrapper for rembg (registered to Model Serving)   |
| `notebooks/01_register_model.py` | One-time notebook: logs & registers the model in Unity Catalog         |
| `cluster_init.sh`                | Legacy: run the API directly on a GPU cluster driver                   |
| `deploy/deploy.ps1`              | Upload source files + deploy Databricks App (profile: nrix)            |
| `deploy/dev.ps1`                 | Start local dev server with hot-reload                                 |
| `deploy/killports.ps1`           | Free port 8000 if already in use                                       |
