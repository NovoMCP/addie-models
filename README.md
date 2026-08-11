# addie-models

ADMET prediction service for the [NovoMCP](https://github.com/NovoMCP/novomcp) open computational chemistry engine. Serves **31 base ADMET endpoints** plus a **22-model Therapeutics Data Commons (TDC) state-of-the-art overlay** — CYP inhibition/substrate, clearance, hepatotoxicity (DILI), cardiotoxicity (hERG + DICTrank), Ames, Tox21 nuclear-receptor/stress-response panels, permeability, solubility, and more.

Weights are published separately on Hugging Face: **[NovoMCP/addie-models](https://huggingface.co/NovoMCP/addie-models)** (~510 MiB, MIT).

## Quickstart

Build and run from source — the recommended path; no dependency on a prebuilt image. Weights download from Hugging Face on first boot (no cloud credentials):

```bash
docker build -t addie-models .
docker run -p 8025:8025 addie-models
# first boot downloads the weights (~510 MiB), then serves on :8025
```

Or pull the prebuilt image:

```bash
docker run -p 8025:8025 ghcr.io/novomcp/addie-models:latest
```

Or run without Docker:

```bash
pip install -r requirements.txt   # Python 3.11 recommended
python3 main.py
```

Then:

```bash
curl -s http://localhost:8025/health
# {"status":"healthy","models_loaded":31, ...}

curl -s -X POST http://localhost:8025/addie/process \
  -H 'Content-Type: application/json' \
  -d '{"molecules":[{"id":"aspirin","smiles":"CC(=O)Oc1ccccc1C(=O)O"}]}'
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + `models_loaded` count |
| GET | `/models` | List loaded model endpoints |
| POST | `/addie/process` | Predict ADMET for a batch of molecules (up to 100) |

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `STORAGE_BACKEND` | `HF` | Where weights come from: `HF` (Hugging Face) \| `S3` \| `AZURE` |
| `HF_MODEL_REPO` | `NovoMCP/addie-models` | Hugging Face weights repo (when `STORAGE_BACKEND=HF`) |
| `PORT` | `8025` | HTTP port |
| `MODEL_PREFIX` | `production/` | Base-model key prefix within the weights store |
| `MODEL_BUCKET` | — | S3 bucket (only when `STORAGE_BACKEND=S3`) |

## Model coverage

- **Base ADMET (31):** binding affinity, 4 cardiotoxicity time-windows, 5 CYP450 inhibition, 7 nuclear-receptor (Tox21), 5 stress-response (Tox21), 9 toxicity (Ames, carcinogenicity, clinical/developmental/reproductive/respiratory toxicity, eye corrosion/irritation, hepatotoxicity).
- **TDC SOTA overlay (22):** CYP (Veith) + substrate, clearance (hepatocyte/microsome), DILI, hERG, Caco-2, HIA, bioavailability, lipophilicity, solubility, BBB, PPBR, VDss, half-life, LD50, P-gp substrate. Five endpoints are 5-seed ensembles that meet or beat the published TDC leaderboard.

See the [model card](https://huggingface.co/NovoMCP/addie-models) for per-endpoint training data and attribution.

## Testing

```bash
# with the service running locally
ADDIE_URL=http://localhost:8025 python3 tests/test_model_coverage.py
```

## License

- **Code:** Apache-2.0 (see `LICENSE`).
- **Model weights:** MIT, with attribution to training-data sources — see the [model card](https://huggingface.co/NovoMCP/addie-models) and its `NOTICE`.
