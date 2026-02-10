# Colab User Guide (Step 9.5)

**Ziel:** Du kannst in Colab die komplette *strict* Training-Pipeline laufen lassen, ohne dass wir zwischendurch „im Nebel“ debuggen.

> Wichtig: Mock-/Dummy-Fallbacks sind entfernt. Wenn Daten/Artefakte fehlen, bricht es sauber ab (gewollt).

---

## 0) Colab Setup

**Optional bequem:** Du kannst auch direkt das All-in-One Notebook öffnen:
- `TraderHimSelf/notebooks/99_full_pipeline.ipynb`

1. Google Colab öffnen
2. `Runtime` → `Change runtime type`
   - Hardware accelerator: **GPU**
   - (Wenn möglich A100 auswählen / Colab Pro)
3. **Google Drive mount (empfohlen, weil Daten/Modelle fett sind)**

In einer Python-Zelle:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 1) Repo holen + in TraderHimSelf wechseln

In einer Zelle:

```bash
# clone
cd /content
git clone <DEIN-REPO-URL> dings-trader

# go to training code
cd /content/dings-trader/projects/dings-trader/TraderHimSelf
ls
```

---

## 1.5) Drive‑Storage verdrahten (damit nix neu runterlädt)

```bash
STORE=/content/drive/MyDrive/dings-trader-store/TraderHimSelf
mkdir -p $STORE/{data_raw,data_processed,models,logs,runs,checkpoints}

for d in data_raw data_processed models logs runs checkpoints; do
  rm -rf $d
  ln -s $STORE/$d $d
done

ls -lah | head
```

Ab jetzt landen alle großen Dateien persistent auf deinem Drive.

---

## 2) Dependencies installieren

```bash
python -V
pip install -r requirements.txt
```

Wenn irgendwas mit `pyarrow`/`torch`/`stable-baselines3` zickt, machen wir das in `00_setup.ipynb` sauber.

---

## 3) Daten downloaden (Step 3)

```bash
python download_binance_data.py
ls -lah data_raw | head
```

**Hinweis:** Der 3m-Download ist riesig. Mit den Symlinks oben landet er direkt auf Drive und du musst ihn später nicht nochmal ziehen.

---

## 4) Dataset Builder (Step 4)

```bash
python build_dataset.py
ls -lah data_processed | head
```

---

## 5) Feature Engine (Step 5)

```bash
python feature_engine.py build
ls -lah data_processed | egrep "features.parquet|scaler.pkl" || true
```

Wenn das hier abbricht mit „Train window 2019–2023 leer“ → dann haben wir nicht genug Historie oder falsche Dates.

---

## 6) PatchTST Forecast Training (Step 8 train)

```bash
python forecast/train_patchtst.py train
ls -lah models | egrep "forecast_model.pt" || true
```

---

## 7) Forecast Features Precompute (Step 8 precompute)

```bash
python forecast/train_patchtst.py precompute
ls -lah data_processed | egrep "forecast_features.parquet" || true
```

Wenn `forecast_model.pt` fehlt → precompute muss jetzt (strict) abbrechen. Dann erst train.

---

## 8) PPO Training (Step 9)

```bash
python policy/train_ppo.py
```

Erwartung: SB3 schreibt Logs/Checkpoints irgendwo unter `runs/` oder `checkpoints/` (je nach Script). Danach Artefakte sichern.

---

## 9) Artefakte sichern

Mindestens sichern:
- `data_processed/scaler.pkl`
- `models/forecast_model.pt`
- `data_processed/forecast_features.parquet`
- PPO Policy/Checkpoints/Logs

Wenn wir Drive mounten: einfach `cp -r` nach Drive.

---

## 10) Copy/Paste Report für Diagnose

Im Notebook `07_eval.ipynb` die letzte Zelle laufen lassen.
Dann den Block zwischen `REPORT_START` und `REPORT_END` komplett hier rein kopieren.

Wenn ein Step crasht: zusätzlich die letzte `ERROR: ...` Zeile + passende Logdatei aus `logs/colab/*.log` schicken.

---

## Troubleshooting (die typischen Killer)

- **Zu wenig Daten** → strict mode bricht (Lookback/Horizon/Train-Window)
- **`forecast_model.pt` fehlt** → precompute bricht (absichtlich)
- **Colab Reset** → Daten/Modelle weg, wenn nicht in Drive
- **3m Daten riesig** → ggf. erstmal Zeitraum begrenzen zum Debug (aber für echtes Training wieder full)
