# Dings-Trader ML Training Roadmap

## 🎯 Workflow & Prozess

Dieses Dokument beschreibt den Training-Prozess für das dings-trader ML-System.

### Sub-Agent Architektur

Wir arbeiten mit einem **2-Ebenen-Agentensystem**:

```
┌─────────────────────────────────────────┐
│  MAIN AGENT (Dings) - Orchestrator      │
│  • Steuert den Gesamtprozess            │
│  • Koordiniert Sub-Agents               │
│  • Integriert Ergebnisse                │
│  • Browser-Automation für Colab         │
└──────────────┬──────────────────────────┘
               │ spawns
                        ▼
┌──────────────────────────────────────────────────┐
│  SUB-AGENTS (je nach Aufgabe)                    │
│                                                  │
│  ┌─────────────────┐  ┌──────────────────────┐  │
│  │ Codex CLI       │  │ Gemini CLI           │  │  (aktuell deaktiviert: quota/keine Nutzung)
│  │ gpt-5.2/5.3     │  │ gemini-3-pro-preview │  │
│  │ • Coding        │  │ • Analysis           │  │
│  │ • Algorithms    │  │ • Architecture       │  │
│  │ • Notebooks     │  │ • Documentation      │  │
│  │ • Full-auto     │  │ • Reasoning          │  │
│  └─────────────────┘  └──────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### Workflow für Sub-Agents

**Für Sub-Agent (Coding/Implementation):**
> Modelle: `gpt-5.2-codex`, `gpt-5.3-codex`, `gemini-3-pro-preview`, `gemini-3-flash-preview`:
> Du bist ein Sub-Agent. Deine Aufgabe ist es, Code lokal im Workspace zu schreiben.
> - Arbeite in `/home/maxim/.openclaw/workspace/projects/dings-trader/TraderHimSelf/`
> - Lese `/home/maxim/.openclaw/workspace/projects/dings-trader/PLAN.md`
> - Schreibe modularen, gut kommentierten Code
> - Für Colab: Erstelle `.ipynb` Dateien (wir kopieren sie später manuell)
> - Speichere alle Zwischenergebnisse
> - Keine externen API-Aufrufe ohne Erlaubnis
> - Fertige Tasks/zwischen Tasks berichten


**Für Sub-Agent (Analysis/Design):**
> Modelle: `gpt-5.2-codex`, `gpt-5.3-codex`, `gemini-3-pro-preview`, `gemini-3-flash-preview`:
> Du bist ein Sub-Agent. Deine Aufgabe ist es, Analyse und Design zu liefern.
> - Arbeite lokal im `/home/maxim/.openclaw/workspace/projects/dings-trader/TraderHimSelf/` 
> - Analysiere Daten, entwerfe Architekturen, schreibe Dokumentation
> - Speichere Ergebnisse als Dateien
> - Keine destruktiven Operationen

**Für den Main Agent (Dings):**
- Spawnt den passenden Sub-Agent je nach Aufgabe
> - Modelle: `gpt-5.2-codex`, `gpt-5.3-codex`, `gemini-3-pro-preview`, `gemini-3-flash-preview`:
- Schreibe dem Subagenten: "Lese `/home/maxim/.openclaw/workspace/projects/dings-trader/docs/TRAINING_ROADMAP.md` und mache den nächsten Zwischenschritt aus `## 📋 TODO Liste`."
- Überwacht den Fortschritt
- Integriert Ergebnisse
- Bei Bedarf: Browser-Automation für Google Colab
- Fertige zwischen Tasks in diesem Dokument mit einem grünen Haken (✅) markieren

### Google Colab Integration

1. **Lokale Entwicklung** (Codex Sub-Agent):
   - Code wird lokal geschrieben und getestet
   - `.ipynb` Notebooks werden generiert

2. **Transfer zu Colab** (Main Agent oder manuell):
   - Code wird in Google Colab kopiert
   - A100 GPU für Training genutzt
   - Ergebnisse werden zurückgespielt

### Iterativer Prozess

Dieses Projekt wird **nicht in einem Rutsch** umgesetzt:
- Modulare Entwicklung (Data-Loader → Model → Training → Evaluation)
- Mehrere Iterationen und Anläufe
- Kontinuierliches Refinement
- in kleinen zwischen Schritten bearbeiten

---

## 🧠 Zwei-Modell-Architektur

**Ziel:** Ein System aus zwei kooperierenden ML-Modellen

1. **Preis-Vorhersage-Modell** (Predictor)
   - Vorhersage zukünftiger Kursbewegungen
   - Output: Erwartete Preis-Range / Richtung

2. **Entscheidungs-Modell** (Actor/Trader)
   - Tradiert auf Basis der Vorhersagen
   - Output: Long/Short/Flat + Position-Size

**Training:** Beide Modelle über Google Colab (A100 GPU)

---

## 📋 TODO Liste


### 🔄 Status‑Update (2026-02-10) — strict fail-fast (keine Mock-Fallbacks)

**Code/Contracts ✅ (implementiert + gehärtet):**
- [x] Step 4 Dataset Builder vorhanden (`build_dataset.py`, slot_15m, missing flags, funding mapping)
- [x] Step 5 Feature Engine strict (Scaler-Fit nur 2019–2023, keine Dummy-Scaler)
- [x] Step 6 Env Fixes (slot_15m mapping, funding, missing/NaN handling)
- [x] Step 8 Forecast Pipeline Fixes (Inputs aus `features.parquet`, Output `forecast_0..34`)
- [x] Step 9 PPO Merge strict (Forecast required, fail-fast; legacy rename `fc_feat_*` → `forecast_*` ok)

**Training/Artefakte ⬜ (noch offen / blocked):**
- [ ] Step 3 Multi‑Year Binance Daten (2019+) laden (sonst Lookback512/Buffer800 & Scaler-Fit unmöglich)
- [ ] Step 8 PatchTST **trainieren** → `models/forecast_model.pt`
- [ ] Step 8 Precompute (erfordert `forecast_model.pt`) → `data_processed/forecast_features.parquet`
- [ ] Step 9 PPO **trainieren** → Policy Artefakte (z.B. `ppo_policy.zip`)

"Smoke-Run mit 4‑Tage Mock-Daten" war nur historisch; seit strict-mode bricht das (gewollt) ab.



# Roter Faden v5 (final) — BTCUSDT Perp Bot (Forecast + PPO) mit 15m Decision / 3m Intrabar + Loss/Feedback + Trade-Limits

---

## Schritt 0 — Fixe Spezifikation (nicht mehr anfassen)

### Instrument
- BTCUSDT Perp (USDT-M), isolated

### Taktung
- Decision timeframe: **15m**
- Intrabar-Simulation timeframe (Backtest/Offline): **3m** (5×3m pro 15m)
- Lookback fürs Modell: **512×15m ≈ 5,3 Tage**
- Buffer fürs Bootstrapping + Longterm Stats: **800×15m ≈ 8,3 Tage**

### Limits / Risk
- Max Hold: **48h** ⇒ **192** Decision-Steps (15m) pro Trade
- Max Exposure gleichzeitig offen: **10% Equity**
- Max gleichzeitig offene Positionen (Lots): **10**
- Leverage: **1–10**
- **Long/Short Exclusion (v1, enforced):** **NIE long und short gleichzeitig**  
  → Wenn bereits Long-Lots offen sind, wird Short-Open geblockt (und umgekehrt). Das reduziert Chaos + Overtrading.

### Workspace / Datenablage
✅ **ERLEDIGT:**
- **Alle Daten, Modelle, Code und Artefakte** werden in `/dings-trader/TraderHimSelf/` abgelegt
- Dies ist das zentrale Arbeitsverzeichnis für alle Sub-Agents
- Struktur: `data/`, `models/`, `notebooks/`, `logs/`, `checkpoints/`

### Fees
- Taker: **0.0006** (Market)
- Maker: **0.0002** (später, nicht v1)
- v1 nutzt Market-Entries/Exits ⇒ taker only

### SL/TP Regel
- SL/TP als ATR-Multiples gesetzt
- Intrabar Trigger:
  - Wenn SL und TP innerhalb derselben 3m-Bar getroffen: **SL-first** (konservativ)

---

## Schritt 1 — Setup (Training vs Execution)

### 1A) Training in Google Colab (A100)
- Install: torch, stable-baselines3, gymnasium, numpy, pandas, pyarrow
- Projektstruktur:
  - data_raw/
  - data_processed/
  - features/
  - env/
  - forecast/
  - policy/
  - eval/
  - runs/
  - live/

### 1B) Bot Execution auf deinem PC (Ubuntu VM empfohlen)
- Inference alle 15m: Feature → Forecast → PPO → Risk → Action
- Kein High-End Rechner nötig (Training ist der teure Teil)

**Artefakte (müssen exportiert werden)**
- scaler.pkl (Normalisierung)
- forecast_model.pt (PatchTST)
- ppo_policy.zip (Stable-Baselines3 PPO)
- config.json (Konstanten + Feature-Order + Obs-Order + Action-Mapping)

---

## Schritt 2 — Data Contract (Training = Live)

**CandleRecord**
- open_time_ms, open, high, low, close, volume

**FundingRecord**
- time_ms, funding_rate

**Regel**
- Backtest/Shadow/Demo/Live nutzen exakt dasselbe Schema + denselben Feature-Code.

---

## Schritt 3 — Datenquellen (historisch vs live) ⚠️ (blocked: Multi‑Year Historie fehlt aktuell)

### 3A) Historie (Training/Backtest) = Binance (Multi-Year möglich)
- OHLCV 15m: 2019 → heute
- OHLCV 3m: 2019 → heute (oder soweit verfügbar)
- Speichern:
  - data_raw/btcusdt_15m.parquet
  - data_raw/btcusdt_3m.parquet

### 3B) Live/Demo/Execution = Bitget (Parity zum echten Handel)
- Candles + Funding live von Bitget
- API-Key nötig erst für Demo/Live Trading (nicht für reines Market Data)

---

## Schritt 4 — Dataset Builder (einmal sauber bauen)

Script: build_dataset.py

1) Load OHLCV 15m + 3m
2) Align:
   - jeder 15m Slot hat idealerweise exakt 5×3m Subbars
   - missing Subbars markieren (keine Future-Leaks)
3) Funding-Serie:
   - Funding auf Timeline mappen (step-wise gehalten)
4) Save:
   - data_processed/aligned_15m.parquet
   - data_processed/aligned_3m.parquet
   - data_processed/funding.parquet

Checks:
- keine Time-Travel Bugs
- konsistente Zeitzonen/timestamps

---

## Schritt 5 — Feature Engine (fixe Feature-Liste + Reihenfolge) ✅

**Prinzip**
- Eine Feature-Funktion compute_features(buf_15m) für Backtest UND Live.
- buf_15m Länge: 800 (für 7d stats). Model-Lookback bleibt 512.
- **Implementiert in:** `TraderHimSelf/feature_engine.py`
- **Scaler:** `data_processed/scaler.pkl` (Fit auf 2019-2023)
- **Unit-Test:** Parity-Check (historisch vs live)

### 5.1 Core Feature Vector (fest, Reihenfolge!)

#### A) Returns & Range (Basis)
1. ret_1 = log(close_t / close_{t-1})
2. ret_4 (1h) = log(close_t / close_{t-4})
3. ret_16 (4h) = log(close_t / close_{t-16})
4. ret_48 (12h) = log(close_t / close_{t-48})
5. hl_range_pct = (high - low) / close
6. oc_range_pct = (close - open) / open

#### B) Volatilität & ATR
7.  vol_16  = rolling_std(ret_1, 16)   (~4h)
8.  vol_96  = rolling_std(ret_1, 96)   (~1d)
9.  vol_672 = rolling_std(ret_1, 672)  (~7d)  (aus buf_15m >= 672)
10. atr_14  = ATR(14)

#### C) Trend / Mean Reversion
11. ema_20_dist  = (close - EMA20)/EMA20
12. ema_50_dist  = (close - EMA50)/EMA50
13. ema_200_dist = (close - EMA200)/EMA200
14. ema_20_slope = (EMA20 - EMA20_prev)/EMA20_prev
15. ema_50_slope = (EMA50 - EMA50_prev)/EMA50_prev
16. adx_14       = ADX(14)

#### D) Momentum
17. rsi_14
18. macd        = EMA12 - EMA26
19. macd_signal = EMA(macd, 9)
20. macd_hist   = macd - macd_signal

#### E) Volume (robust)
21. vol_log  = log(1 + volume)
22. vol_z_96 = zscore(vol_log, 96)

#### F) Zeit-Features
23. hour_sin
24. hour_cos
25. dow_sin
26. dow_cos

#### G) Funding Features
27. funding_rate_now
28. time_to_next_funding_steps (in 15m steps, capped 0..32)

Core Features Dimension: 28

### 5.2 Normalisierung (fest)
- Fit StandardScaler nur auf Train (2019–2023)
- Apply identisch auf Val/Test/Live
- Save scaler.pkl
- Live niemals neu fitten

### 5.3 Parity Unit-Test (Pflicht)
- historische Candles als Stream simulieren
- Feature-Vektor muss identisch zu offline gerechnetem Vektor sein (float tolerance)

---

## Schritt 6 — Trading Environment (15m Decision + 3m Intrabar + Multi-Position bis 10 Lots) ✅

File: env/perp_env.py (Gymnasium Env)

### 6.1 Portfolio-State (fest)
✅ **Implementiert:** `TraderHimSelf/env/perp_env.py`
- equity
- open_positions: Liste von Positions-Lots (0..10), jedes Lot enthält:
  - side ∈ {long, short}
  - margin_used
  - leverage L
  - notional = margin_used * L
  - entry_price
  - qty = notional / entry_price
  - sl_price, tp_price
  - open_time_ms
  - time_in_trade_steps_15m

**Konservativ v1:**
- Wenn open_positions nicht leer:
  - Neue Position darf nur in derselben Richtung geöffnet werden
  - Gegenseitiges Hedging wird geblockt (reduziert Overtrading/Chaos)

### 6.2 uPnL (USDT-M linear) pro Lot
✅ **Implementiert**

### 6.3 Fees (taker only v1)
✅ **Implementiert** (0.0006)

### 6.4 Funding (event-basiert)
✅ **Implementiert**

### 6.5 Slippage (konservativ, gegen dich)
✅ **Implementiert** (ATR-basiert)

### 6.6 Liquidation (isolated, konservativer Proxy)
✅ **Implementiert**

### 6.7 SL/TP setzen (ATR-Multiples) pro Lot
✅ **Implementiert**

### 6.8 Intrabar Simulation innerhalb 15m (5×3m)
✅ **Implementiert** (SL-first)

---

## Schritt 7 — Risk Manager Wrapper (hardcoded, vor ML!) + Overtrading-Controls ✅

File: env/risk_manager.py

### 7.1 Hard Caps (fest)
✅ **Implementiert:** `TraderHimSelf/env/risk_manager.py`
1) **Exposure cap**
- exposure_open_margin = Summe(margin_used aller offenen Lots)
- available_exposure = 0.10*equity - exposure_open_margin
- clamp new_margin_used zu available_exposure
- wenn available_exposure <= 0 → force flat (keine neue Position)

2) **Max offene Positionen**
- max_open_positions = 10
- wenn len(open_positions) >= 10 → force flat (keine neue Position)

3) **Leverage clamp**
- L ∈ [1, 10]

4) **SL/TP clamp**
- sl_mult ∈ [0.5, 3.0]
- tp_mult ∈ [0.5, 6.0]
- tp_mult >= sl_mult

5) **No-hedge rule (v1, konservativ)**
- wenn offene Lots existieren:
  - wenn action direction ≠ Richtung der offenen Lots → force flat

### 7.2 Soft Controls (damit er nicht 1000 Trades ballert)
✅ **Implementiert** (Entry Penalty: 0.0002 * equity)

---

## Schritt 8 — Forecast Modell (PatchTST) + Forecast Feature Block + Forecast Loss (Supervised) ✅

File: forecast/train_patchtst.py

### 8.1 Input
✅ **Implementiert:** `TraderHimSelf/forecast/train_patchtst.py`
- Lookback = 512
- Input channels: 28 Core Features (normalisiert)

### 8.2 Targets (multi-horizon)
✅ **Implementiert** (q10, q50, q90 für 1h, 4h, 12h, 24h, 48h)

### 8.3 Forecast Feature Block (fix)
✅ **Implementiert** (35 Dimensions: Horizon Block, Path Block, Curve Stats)

### 8.4 Forecast Loss (Pinball / Quantile Loss)
✅ **Implementiert** (Horizon weights: w_4=1.0, w_16=1.0, w_48=0.8, w_96=0.6, w_192=0.4)

### 8.5 Forecast Evaluation (Val/Test)
✅ **Vorbereitet**

### 8.6 Precompute Pflicht
✅ **CLI-Modus implementiert** (`precompute`)
- Output: `data_processed/forecast_features.parquet` (Spalten `forecast_0..34`, NaN‑Padding für Lookback)
- **Strict:** erfordert `models/forecast_model.pt` (sonst Abbruch) — erst `train`, dann `precompute`
- Forecast weights anschließend **freezen** (für PPO Training)

---

## Schritt 9 — PPO Policy Training + PPO Loss / Credit Assignment (RL) ✅

File: policy/train_ppo.py
✅ **Implementiert:** `TraderHimSelf/policy/train_ppo.py`
⚠️ Training‑Artefakte (z.B. `ppo_policy.zip`) entstehen erst nach echtem PPO‑Training.

### 9.1 Testen aller Bausteine (System-Check) ✅
- Alles nach Roadmap gemacht? ✅
- Kommunikation zwischen Modulen prüfen (Data -> Feature -> Env -> Policy) ✅
- Schnittstellen-Validierung ✅
- Logik-Review ✅
- Bericht erstellt: `TraderHimSelf/system_check_report.md` ✅

### 9.2 Bug-Hunting (Sub-Agent Audit) ✅
- Neue Sub-Agent Instanz spawnen ✅
- Codebase nach logischen Fehlern, Edge-Cases und Performance-Bottlenecks scannen ✅
- **Status:** Audit abgeschlossen, 3 kritische Fehler in `perp_env.py` gefunden. Bericht in `audit_report_9.2.md`.

### 9.2.1 Bug-Fixing (Refactoring) ✅
- Behebung des Time-Travel-Bugs in `perp_env.py`. ✅
- Korrektur der Liquidation-Logik. ✅
- Performance-Optimierung der Intrabar-Simulation (O(1) Zugriff). ✅

### 9.3 Planung: Umzug nach Google Colab ✅
- [x] Plan + Artefakt-Matrix: `docs/COLAB_MIGRATION_PLAN.md`
- [x] Entscheidung: Variante A (empfohlen) = **git clone im Colab**
- [x] `download_binance_data.py` Pfade relativ/parametrisierbar gemacht (`--data-dir`, `--start-date`, `--end-date`, `--symbol`)

### 9.4 Umzug vorbereiten (Notebooks + Bündelung) ✅
- [x] Notebook-Kette angelegt: `00_setup.ipynb` … `07_eval.ipynb`
- [x] `00_setup.ipynb`: Drive mount + Store-Ordner + Symlinks (`data_raw`, `data_processed`, `models`, `logs`, `runs`, `checkpoints`)
- [x] Notebook-Zellen rufen die bestehenden Scripts auf (kein doppelter Code)
- [x] Pro Step klare Status-Ausgabe `OK:` / `ERROR:` + Logs unter `logs/colab/*.log`
- [x] Finale Report-Zelle in `07_eval.ipynb` (`REPORT_START` / `REPORT_END` via `report_status.py`)

### 9.5 Anleitung schreiben (User Guide) ✅
- [x] Step-by-step Guide: `docs/COLAB_USER_GUIDE.md`

### 9.6 Fokus Stop & Manueller Umzug (User Action) ⬜
- [ ] Fokus-Mode für alle Agenten beenden (wenn wir wirklich rübergehen)
- [ ] **User führt Umzug durch:** nach `docs/COLAB_USER_GUIDE.md`
- [ ] Erster Start des Trainings in der Cloud

---

## Schritt 10 — Evaluation (Walk-forward)

Splits:
- Train: 2019–2023
- Val:   2024
- Test:  2025

KPIs:
- net PnL after costs
- max drawdown
- liquidation count
- fee share
- exposure time
- avg open positions (soll klein sein)
- trades per week (soll klein sein)

Baselines:
- flat always
- EMA trend + ATR SL/TP

---

## Schritt 11 — Warmstart / Bootstrapping (fest)

Beim Bot-Start (Shadow/Demo/Live identisch):
1) Ziehe mindestens **800×15m** Candles (Buffer) (mindestens 512, aber Ziel 800)
2) Rechne Features:
   - Model Input: letzte 512 Candles
   - Longterm Stats: aus 800er Buffer (z.B. vol_672)
3) Lade scaler + forecast + ppo
4) Erst wenn Buffer voll & Candle final valid → Decision Loop starten

Gap Handling:
- Wenn Candles fehlen (Bot offline):
  - Missing nachziehen
  - wenn nicht möglich: force flat bis Buffer wieder konsistent

**Wichtig:** Bootstrapping ist live schnell machbar:
- 800×15m ≈ 8,3 Tage. Das wird einmalig nachgeladen, danach läuft’s mit Live-Candles weiter.

---

## Schritt 12 — Shadow Live (kein Trading, kein Key nötig)

File: live/shadow_runner.py
- alle 15m (close + 3s):
  1) neue 15m Candle holen, Buffer append
  2) Features + Forecast + PPO Action
  3) RiskManager check
  4) nur loggen, keine Orders

Optional:
- 3m Candles der letzten 15m holen für Debug/Monitoring

---

## Schritt 13 — Demo Live (Bitget Demo API-Key nötig)

- Bootstrapping wie Schritt 11
- Policy → Risk → Demo Orders + SL/TP setzen
- Outcomes loggen
- Demo ist Pflicht-Gate, bevor echtes Geld überhaupt angefasst wird

---

## Schritt 14 — Live scharf schalten (echtes Geld)

Nur wenn:
- Shadow stabil
- Demo stabil
- KPIs okay

Dann:
- Subaccount + kleines Kapital (empfohlen)
- Trade-Key ohne Withdraw/Transfer
- Start mit kleiner Exposure (1–2%), langsam hoch bis 10%
- (später) VPS mit fixer IP für IP-Whitelist

---

## Was du JETZT programmieren lässt (exakte Reihenfolge)
1) build_dataset.py (15m+3m, Alignment, Funding Schema)
2) feature_engine.py (28 Core Features + scaler + Parity Test)
3) perp_env.py (15m decision + 3m intrabar + SL-first + Multi-positions bis 10)
4) risk_manager.py (Exposure cap + max 10 positions + no-hedge + entry penalty)
5) PatchTST training + forecast precompute (35 Features + Pinball Loss)
6) PPO training (Obs dim 72, reward shaping inkl. entry penalty)
7) Evaluation (walk-forward + baseline)
8) bootstrapping + gap handling
9) shadow live
10) demo live
11) live


