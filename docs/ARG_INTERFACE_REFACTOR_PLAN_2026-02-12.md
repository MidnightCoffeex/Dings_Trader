# Arg-Interface Refactor Plan (2026-02-12)

## Ziel
Das Training/Deployment soll schnittstellenartig funktionieren:
- einheitliche Argumente,
- reproduzierbare Modellpaare,
- dieselbe Config von Colab -> PPO -> Backend -> UI -> Live/Paper.

## Scope (v1)
1. **Single Notebook Entry**: `notebooks/99_full_pipeline.ipynb` als zentraler Runner.
2. **Config first**: Alle relevanten Parameter als explizite Inputs.
3. **Manifest + Paketstruktur**: jedes Modellpaar mit eigener Metadaten-Datei.
4. **Strict Coupling**: PPO darf nur auf Forecast-Config trainieren, die exakt passt.

---

## Standard-Konfiguration (Start)
- decision timeframe: `15m`
- forecast horizon: `16 steps` (= 4h)
- intrabar timeframe: `3m` (immer für SL/TP-Reihenfolge)
- feature set: `train30` (30 Features)

---

## Konfigurationsschema (PipelineConfig)
Vorgeschlagenes Schema:

```json
{
  "decision_tf": "15m",
  "intrabar_tf": "3m",
  "forecast_horizon_steps": 16,
  "lookback_steps": 512,
  "feature_set": "train30",
  "symbol": "BTCUSDT",
  "exchange": "binance",
  "train_range": "2019-01-01..2025-12-31",
  "model_tag": "shortterm_v1"
}
```

## Modellpaket-Struktur
```text
models/packages/
  2026-02-12_15m_h16_train30_shortterm_v1/
    forecast_model.pt
    ppo_policy.zip
    scaler.pkl
    manifest.json
```

`manifest.json` enthält mindestens:
- alle Config-Felder,
- Feature-Order,
- Trainingstimestamp,
- Git-Commit,
- erwartete Inference-Input-Dimension.

---

## Harte Regeln
1. **Forecast/PPO Kompatibilität**
   - Kein PPO-Training ohne passendes Forecast-Manifest.
2. **Backend Strict Mode**
   - lädt nur Modellpakete mit vollständigem Manifest.
   - keine stillen Defaults bei TF/Horizon/Feature-Dim.
3. **UI Transparenz**
   - zeigt TF/Horizon/Feature-Set direkt im Modellselector.

---

## Umsetzung in Phasen

### Phase 1 – Notebook + Config
- `99_full_pipeline.ipynb` bekommt zentrale Parameterzelle.
- interne Steps lesen nur noch `PipelineConfig`.

### Phase 2 – Package/Manifest
- Training exportiert Forecast/PPO/Scaler in Paketordner.
- Manifest wird automatisch erzeugt.

### Phase 3 – Backend/UI Integration
- API lädt Paket per Manifest.
- UI zeigt Modell-Metadaten und wählt Paket explizit.

### Phase 4 – Cleanup
- alte Step-Notebooks (`00..07`) nach Freigabe archivieren/löschen.

---

## Offene Entscheidung
- Step-Notebooks sofort löschen oder erst in `notebooks/archive/` verschieben.
  - Empfehlung: **zuerst archivieren**, dann löschen nach 1-2 stabilen Runs.
