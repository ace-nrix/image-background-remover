# Image Background Remover API

A lightweight FastAPI service that removes image backgrounds using [rembg](https://github.com/danielgatis/rembg) (u2net model) with CUDA acceleration. Designed to run on a Databricks GPU cluster and expose a single HTTP endpoint.

## What it does

- Accepts any image format (JPEG, PNG, WebP, etc.)
- Removes the background using the u2net deep learning model (CUDA if available, CPU fallback)
- Scales the subject to fill a configurable percentage of the original canvas (whitespace control)
- Returns a transparent **PNG** by default, or a flat-color **JPEG** when a background color is provided

## Endpoint

```
POST /remove-background
```

| Field                | Type   | Required           | Description                                                           |
| -------------------- | ------ | ------------------ | --------------------------------------------------------------------- |
| `image`              | file   | yes                | Input image in any PIL-readable format                                |
| `whitespace_percent` | float  | no (default: `10`) | Padding around the subject. `15` means subject fills 85% of canvas    |
| `bg_color`           | string | no                 | CSS color name (`white`) or hex (`#ff0000`). Omit for transparent PNG |

```
GET /health  →  {"status": "ok"}
```

### Example — curl

```bash
# Transparent PNG (no bg_color)
curl -X POST http://localhost:8000/remove-background \
  -F "image=@photo.jpg" \
  -F "whitespace_percent=15" \
  -o result.png

# White background JPEG
curl -X POST http://localhost:8000/remove-background \
  -F "image=@photo.jpg" \
  -F "whitespace_percent=10" \
  -F "bg_color=white" \
  -o result.jpg
```

Interactive docs at `http://localhost:8000/docs` when running locally.

## Local development

```bash
# 1. Create venv and install deps
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Start with hot-reload
.\deploy\dev.ps1
# or directly:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

If port 8000 is in use:

```powershell
.\deploy\killports.ps1
```

## Databricks GPU cluster deployment

The service is designed to run on a Databricks GPU cluster as a cluster-scoped init script.

### Setup

1. **Attach the init script** — in Cluster settings → Advanced → Init Scripts, add:

    ```
    /Volumes/<catalog>/<schema>/<volume>/cluster_init.sh
    ```

    Or upload `cluster_init.sh` to DBFS and reference it as `/dbfs/...`.

2. **Start the cluster** — the init script will:
    - Install system packages (`python3-venv`, `libgl1`)
    - Create a Python venv in `/tmp/rembg_venv` (reused across restarts)
    - Install `rembg[gpu]`, `fastapi`, `uvicorn`, `pillow`
    - Start uvicorn on port 8188

3. **Access the API** via the Databricks driver-proxy URL:
    ```
    https://<workspace>.azuredatabricks.net/driver-proxy/o/<orgId>/<clusterId>/8188/
    ```

### Logs on the cluster

```bash
%sh
tail -n 50 /tmp/rembg_api_init.log   # init script progress
tail -n 50 /tmp/rembg_api.log        # uvicorn runtime output
```

### Deploy as a Databricks App

```powershell
.\deploy\deploy.ps1
```

This uploads source files to the Databricks workspace and redeploys the app.

## Files

| File                   | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| `app.py`               | FastAPI application                                        |
| `requirements.txt`     | Python dependencies                                        |
| `cluster_init.sh`      | Databricks cluster init script (installs + starts the API) |
| `deploy/deploy.ps1`    | Upload + deploy to Databricks Apps                         |
| `deploy/dev.ps1`       | Start local dev server with hot-reload                     |
| `deploy/killports.ps1` | Free port 8000 if already in use                           |
