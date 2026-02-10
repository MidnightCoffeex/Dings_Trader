# Colab Migration Plan (Step 9.3 / 9.4)

**Ziel:** Training der beiden Modelle (PatchTST Forecast + PPO Policy) auf Google Colab (A100), ohne Chaos, ohne „works on my machine“.

**Prinzip:**
- `TRAINING_ROADMAP.md` bleibt unser *Control Panel* (Checkboxen + Links).
- Die Details (Plan + Guide) stehen in separaten Docs.
- Pipeline ist **strict fail-fast** (keine Mock-/Dummy-Fallbacks mehr).

---

## Entscheidung: Transfer-Variante

### ✅ Variante A (empfohlen): `git clone` im Colab Notebook
**Warum:** reproduzierbar, wenig Copy/Paste, sauberer Diff/History.

- Repo wird in `/content/dings-trader/` geklont
- Wir arbeiten dann **immer** im Unterordner `projects/dings-trader/TraderHimSelf/`

### Variante B: ZIP Upload
Nur falls Git im Notebook nervt (oder privates Repo ohne Auth). Mehr manuelle Schritte.

### Variante C: Google Drive als Code-Workspace
Geht, aber macht Pfade/Permissions und „Drive ist langsam“ oft unnötig kompliziert.

---

## Daten- & Artefakt-Strategie

### Daten (groß)
- Ziel: Multi-Year Binance Historie ab **2019-01-01**
- Benötigt:
  - `data_raw/btcusdt_15m.parquet`
  - `data_raw/btcusdt_3m.parquet`
  - Funding (aus API) → `data_raw/funding_rates.parquet`

**Speicher-Ort (Drive‑first, empfohlen):**
- Wir mounten **deinen privaten Google Drive** und speichern **alle fetten Dateien** dort (Daten + Modelle + Logs), damit du nicht jedes Mal stundenlang neu runterladen musst.
- In Colab arbeiten wir im Repo unter `/content/...`, aber die Ordner werden auf Drive **gespiegelt via Symlinks**.

**Vorschlag Drive-Root:**
- `/content/drive/MyDrive/dings-trader-store/TraderHimSelf/`

**00_setup.ipynb macht dann:**
```bash
# im Repo: /content/dings-trader/projects/dings-trader/TraderHimSelf
STORE=/content/drive/MyDrive/dings-trader-store/TraderHimSelf
mkdir -p $STORE/{data_raw,data_processed,models,logs,runs,checkpoints}

for d in data_raw data_processed models logs runs checkpoints; do
  rm -rf $d
  ln -s $STORE/$d $d
done
```

Damit schreiben alle Scripts weiter auf relative Pfade, aber die großen Outputs landen persistent auf Drive.

### Artefakte (wichtig)
Diese Dateien sollen nach jedem großen Step gesichert werden (Drive oder Download):
- `data_processed/scaler.pkl`
- `models/forecast_model.pt`
- `data_processed/forecast_features.parquet`
- PPO Policy/Checkpoints (z.B. `ppo_policy.zip`, je nach Script-Output)

---

## Notebook-Kette (Step 9.4)

Wir bauen/halten eine lineare Kette von Notebooks (ein Notebook pro Step). Die Notebooks rufen die bestehenden Python-Skripte auf (kein doppelter Code).

Vorschlag:
1. `00_setup.ipynb`
   - Drive mount + Store-Ordner anlegen + Symlinks setzen (data/models/logs persistent)
   - Clone, deps installieren, smoke checks
2. `01_download_data.ipynb`
   - Binance Download (15m/3m/funding)
3. `02_build_dataset.ipynb`
   - `build_dataset.py`
4. `03_feature_engine.ipynb`
   - `feature_engine.py build`
5. `04_train_patchtst.ipynb`
   - `forecast/train_patchtst.py train`
6. `05_precompute_forecast.ipynb`
   - `forecast/train_patchtst.py precompute`
7. `06_train_ppo.ipynb`
   - `policy/train_ppo.py`
8. `07_eval.ipynb`
   - Walk-forward evaluation (Step 10)

---

## Artefakt-Matrix (Input → Command → Output)

| Step | Command (aus `TraderHimSelf/`) | Muss existieren (Input) | Output (muss entstehen) |
|---|---|---|---|
| 3 | `python download_binance_data.py` | internet | `data_raw/*.parquet` |
| 4 | `python build_dataset.py` | `data_raw/*` | `data_processed/aligned_*.parquet`, `funding.parquet` |
| 5 | `python feature_engine.py build` | `data_processed/aligned_15m.parquet` + funding | `data_processed/features.parquet`, `scaler.pkl` |
| 8 (train) | `python forecast/train_patchtst.py train` | `features.parquet`, `scaler.pkl`, `aligned_15m.parquet` | `models/forecast_model.pt` |
| 8 (precompute) | `python forecast/train_patchtst.py precompute` | `forecast_model.pt` | `data_processed/forecast_features.parquet` |
| 9 | `python policy/train_ppo.py` | alle oben | PPO Artefakte/Logs |

---

## Offene „Colab-Blocker“ (müssen wir vor 9.4 fixen)

1) **`download_binance_data.py` nutzt aktuell einen hardcoded absoluten Pfad** (lokaler Workspace-Pfad).
- Für Colab muss das auf **relative Pfade** (`DATA_DIR = os.path.join(BASE_DIR, "data_raw")`) oder CLI-Args umgestellt werden.

2) Storage/Runtime Reset
- Wenn Daten/Modelle nicht in Drive liegen, verlieren wir sie bei Reset.

---

## Done-Definition für 9.3 / 9.4

- **9.3 done** wenn: Plan + Variante + Artefakt-Matrix festgelegt (dieses Doc) und in `TRAINING_ROADMAP.md` verlinkt.
- **9.4 done** wenn: die Notebook-Kette existiert (mind. Stubs) und jede Zelle ruft exakt die Scripts auf.
