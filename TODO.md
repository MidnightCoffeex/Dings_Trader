# dings-trader TODOs

## ✅ Paper Trading Environment (FERTIG)

### Engine
- [x] Paper Trading Engine implementiert (`ml/paper_trading.py`)
- [x] Simuliertes Konto: 10.000 USDT Startkapital
- [x] Max. 5 Positionen gleichzeitig
- [x] Hebel: 7x-10x konfigurierbar
- [x] 5% Profit-Ziel, -3% Stop Loss
- [x] 48h Zeitlimit (Auto-Exit)
- [x] SQLite DB für Trade-Tracking
- [x] **Trading-Gebühren** (0,1% pro Seite) implementiert (07.02.2026)
- [x] **Position-Sizing** auf 2% per Slot (Pyramiding) für 10% Gesamt-Cap optimiert (07.02.2026)

### API
- [x] Paper Trading API (`ml/paper_api.py`)
- [x] Integration in Haupt-API (`ml/api.py`)
- [x] Auto-Trading Loop (`ml/paper_inference_loop.py`)
- [x] **Equity History** API für Chart-Daten (USDT + Timestamps)

### UI / UX
- [x] Paper Trading Panel Komponente
- [x] Paper Trading Badge/Toggle
- [x] Simulated Account Balance Anzeige (inkl. Trading-Cap Kalkulation)
- [x] Aktive virtuelle Positionen mit P&L (scrollbar ab 5 Positionen)
- [x] ML Signal Anzeige (BULLISH/BEARISH)
- [x] Performance Metriken (Win Rate, Return)
- [x] **High-Res Scaling** (1920x1080) mit dynamischen Höhen und 2cm Rändern
- [x] **Real-Time Polling** (2s Intervall) für das gesamte Dashboard
- [x] **Equity Curve Fix** (USDT, Timestamps, Zoom/Brush Feature)
- [x] **Modell-Timer** (Live seit) an Reset-Zeitpunkt gekoppelt
- [x] **Sidebar Cleanup** (Nur noch Dashboard aktiv)
- [x] **Card Consolidation** (Signals/Next/Risk in getrennten Kacheln gruppiert)

## 🚨 DRINGENDE FIXES
- [x] **Live-Zeit Bug** - Timer zeigt 00:00:00, aktualisiert nicht
- [x] **Server-Stabilität** - Next.js Production-Build stabilisiert
- [x] **CORS / API-Proxy** - Polling-Fehler im Frontend behoben

## NEUE FEATURES (nach Priorität)

### 1. **Optimierung der Paper-Modelle**
- [x] Fokus auf `paper_test_hf` (High Frequency) als Default gesetzt
- [x] Backend-Ressourcen für inaktive Modelle gespart
- [ ] `paper_v1`, `paper_v2`, `paper_v3` bei Bedarf reaktivieren

### 2. **Trading-Logik & Brain**
- [ ] Trading-Regeln aus Brain verfeinern (Portfolio-Engine v2)
- [ ] Dynamische Positionsgrößen durch ML-Modell entscheiden lassen
- [ ] Backtesting-Vergleich vs. Buy-and-Hold implementieren

### 3. **Arg-first Schnittstellen-Refactor**
- [ ] `notebooks/99_full_pipeline.ipynb` als Single Entry Point finalisieren
- [ ] Parameterisierte PipelineConfig für TF/Horizon/Feature-Set/Lookback einführen
- [ ] Modellpaar-Ordner + `manifest.json` standardisieren (Forecast+PPO+Scaler)
- [ ] Backend/UI strikt auf Manifest-Metadaten verdrahten
- [x] Alte Step-Notebooks gelöscht (2026-02-12, nur `99_full_pipeline.ipynb` bleibt)
- [ ] Referenzplan: `docs/ARG_INTERFACE_REFACTOR_PLAN_2026-02-12.md`

## Dokumentation
- [x] `docs/FUTURE_PLANS.md` erstellt (Dynamisches Scaling, ML-Sizing)
- [x] `MEMORY.md` aktualisiert (Roadmap & Evolution)

## Python Module (in ml/)
- api.py - Haupt-API Server (Equity & Status)
- paper_trading.py - Core Engine (Gebühren, Slots, DB)
- paper_api.py - API Router (Signal-Processing, Account)
- paper_inference_loop.py - Auto-Trading (10s Takt)
- features.py - Feature Engineering (RSI, etc.)
