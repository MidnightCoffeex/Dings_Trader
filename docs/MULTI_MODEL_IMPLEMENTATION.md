# Multi-Model Implementation Notes

## Architecture

The system supports multiple model packages, each consisting of a PatchTST forecast model (`.pt`) and a PPO policy (`.zip`).

### 1. Model Registry (`model_packages` table)
Stored in `data/trader.sqlite`. 
Columns:
- `id`: Unique slug (e.g. `smoke_test_model`).
- `forecast_rel_path`, `ppo_rel_path`: Relative paths to artifacts.
- `warmup_status`: `PENDING`, `RUNNING`, `DONE`, `FAILED`.
- `warmup_error`: Error message if warmup failed.

### 2. Inference Registry (`ppo_forecast_inference.py`)
Uses a dictionary `_inference_instances` to cache `PPOForecastInference` instances per model ID.
This ensures that:
- Multiple models can run in parallel without reloading weights every time.
- Requests for a specific model ID get the correct artifacts.

### 3. Warmup Logic
- Triggered on first inference (e.g. when Dashboard is opened or Loop starts).
- Sets status to `RUNNING`.
- On success: Sets status to `DONE` and records timestamp.
- On failure: Sets status to `FAILED` and records error message.
- Paper trading loop blocks orders until status is `DONE`.

### 4. Path Resolution
All paths are stored relative to the project root to ensure portability across different environments. Resolution happens in `PPOForecastInference.__init__`.

## How to add a new model
1. Use the UI Upload form or `POST /model-packages/upload`.
2. Select the model in the Dashboard dropdown.
3. The system automatically triggers warmup and starts displaying signals.
4. If used in a loop, start `paper_inference_loop.py` with `--model-id paper_<your_id>`.
