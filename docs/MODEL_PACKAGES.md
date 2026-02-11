# Model Packages (Multi-Model Upload / Registry)

## Motivation
We want to support multiple **paired artifacts** per model:

- Forecast model (PatchTST) as `.pt`
- PPO policy (SB3) as `.zip`

The UI should be able to select a model package dynamically, and users should be able to upload new packages without code changes.

## Design

### Registry
A new SQLite table is introduced in `ml/db.py`:

`model_packages`
- `id` (TEXT, PK) – stable slug used in URLs / UI (`?model=<id>` and paper account `paper_<id>`)
- `name` (TEXT)
- `forecast_rel_path` (TEXT) – path **relative to repo root**
- `ppo_rel_path` (TEXT) – path **relative to repo root**
- `status` (TEXT) – currently `READY | UPLOADED | ARCHIVED` (expandable)
- `created_at` (TIMESTAMP)
- `warmup_required` (INTEGER boolean)
- `warmup_status` (TEXT) – `PENDING | RUNNING | DONE | ERROR`
- `warmup_completed_at` (TIMESTAMP)

A default package is seeded (`ppo_v1`) pointing to the existing artifacts:
- `TraderHimSelf/models/forecast_model.pt`
- `TraderHimSelf/models/ppo_policy_final.zip`

### File storage
Uploads are stored on disk under:

`TraderHimSelf/models/packages/<package_id>/`
- `forecast_model.pt`
- `ppo_policy.zip`

The directory can be overridden via env var `MODEL_PACKAGES_DIR`.

Files are **not committed** (ignored via `.gitignore`).

### Runtime wiring
- UI selects `model=<package_id>`
- Paper account id convention stays: `paper_<package_id>`
- `ml/ppo_forecast_inference.py` is refactored to maintain **one inference instance per package**.
- On first successful inference (`predict()`), warmup state is marked as `DONE` in the registry.
- `/paper/signal` refuses to auto-trade while warmup is not `DONE` (prevents trading before history/feature warmup).

## API

### List
`GET /model-packages`

Returns registry list for the UI.

### Upload
`POST /model-packages/upload` (multipart/form-data)

Parts:
- `name` (string)
- `forecast_model` (`.pt`)
- `ppo_model` (`.zip`)

The backend slugifies `name` into an `id` and ensures uniqueness.

## TODO (Future)
- Stronger pair validation (e.g., embed metadata/manifest with compatible versions).
- Add explicit warmup endpoint + background warmup worker.
- Add archiving/deleting packages + UI management page.
