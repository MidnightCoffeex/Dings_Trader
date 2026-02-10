# Dings‑Trader — Verifikation bis Schritt 9.2 + Bug-/Logik‑Audit

## 🔄 Status‑Update (2026-02-10 10:15) — Fixes eingespielt + Artefakte neu erzeugt

Seit dem initialen Audit wurden konkrete Fixes in `TraderHimSelf/` eingespielt (Forecast‑Pipeline, PPO‑Merge, Env slot_15m‑Mapping + Funding + Missing‑Handling) und Smoke‑Runs gemacht.

**Neu vorhanden in `TraderHimSelf/data_processed/` (Mock/Short‑Data):**
- ✅ `features.parquet`
- ✅ `scaler.pkl`
- ✅ `forecast_features.parquet` (Spalten: `forecast_0..forecast_34`, mit NaN‑Padding für Lookback)

**Weiterhin offen / blocked:**
- ⬜ **Multi‑Year Binance Historie (2019+)** laden (aktuell nur 4 Tage Mock‑Data → Lookback512/Buffer800 nicht sinnvoll)
- ⬜ `models/forecast_model.pt` (PatchTST Training) fehlt → **strict mode:** Precompute bricht ohne dieses Artefakt ab (erst trainieren, dann precompute)
- ⬜ PPO Training Artefakte (z.B. `ppo_policy.zip`) fehlen

---


**Datum:** 2026‑02‑10  
**Scope (User‑Request):** Prüfen, ob *alles* bis inkl. **Schritt 9.2** aus `docs/TRAINING_ROADMAP.md` wirklich erledigt ist (jede Teilaufgabe), plus zusätzliche Fehleranalyse (Bugs/Edge‑Cases/Logik). Ergebnis als Report + Lösungsansätze.

> Hinweis: In der Chat‑Nachricht standen Pfade (`.../TRAINING_ROADMAP.md` im Root, `src/`), die so **nicht existieren**. Die tatsächliche Roadmap liegt hier:
> - `projects/dings-trader/docs/TRAINING_ROADMAP.md`

---

## 0) Repo‑Layout (relevant)

### A) Roadmap / Doku
- `projects/dings-trader/docs/TRAINING_ROADMAP.md`  ✅ (Quelle der Anforderungen)

### B) „Roadmap v5 final“ Implementierung (entscheidend)
- `projects/dings-trader/TraderHimSelf/` ✅ (hier liegt der Roadmap‑Code)

Wichtigste Files:
- Schritt 4: `TraderHimSelf/build_dataset.py`
- Schritt 5: `TraderHimSelf/feature_engine.py`
- Schritt 6: `TraderHimSelf/env/perp_env.py`
- Schritt 7: `TraderHimSelf/env/risk_manager.py`
- Schritt 8: `TraderHimSelf/forecast/train_patchtst.py`
- Schritt 9: `TraderHimSelf/policy/train_ppo.py`

### C) Sonstiges / Legacy
- `projects/dings-trader/ml/` existiert, wirkt wie ein älteres/anderes Experiment‑Setup. Dieser Report bewertet **primär TraderHimSelf**, weil das in der Roadmap explizit genannt wird.

---

## 1) Artefakt‑Check (harte Realität vs. „Code existiert“)

Roadmap verlangt, dass bestimmte Artefakte existieren (damit Step 9.* überhaupt sinnvoll trainierbar ist).

### 1.1 Vorhandene Artefakte (Status: ✅ existiert)
In `TraderHimSelf/data_raw/` (Mock‑Daten, nur 2024‑01‑01 → 2024‑01‑04):
- `btcusdt_15m.parquet` (384 rows)
- `btcusdt_3m.parquet` (1819 rows)
- `btcusdt_funding.parquet` (12 rows)

In `TraderHimSelf/data_processed/`:
- `aligned_15m.parquet` ✅ (384×8, UTC DatetimeIndex)
- `aligned_3m.parquet` ✅ (1920×8, UTC DatetimeIndex, enthält `slot_15m`)
- `funding.parquet` ✅ (384×2, UTC DatetimeIndex)

Reports:
- `TraderHimSelf/system_check_report.md` ✅ (Roadmap 9.1)
- `TraderHimSelf/audit_report_9.2.md` ✅ (Bug‑Hunt 9.2)
- `TraderHimSelf/final_bug_audit_report.md` ✅ (weiteres Audit)

### 1.2 Artefakte‑Status (nach Fix‑Run)
In `TraderHimSelf/data_processed/` sind jetzt vorhanden (Mock/Short‑Data):
- ✅ `features.parquet` (Output von Schritt 5)
- ✅ `scaler.pkl` (Output von Schritt 5)
- ✅ `forecast_features.parquet` (Output von Schritt 8 Precompute; `forecast_0..34`)

In `TraderHimSelf/models/` fehlen weiterhin:
- ❌ `forecast_model.pt` (Output Step 8 Training)
- ❌ PPO Policy `.zip` / SB3 Artifacts (Output Step 9 Training)

**Konsequenz:** Die Pipeline ist jetzt *smoke‑lauffähig* bis inkl. Precompute/Merge, aber „echtes Training“ ist weiterhin **blocked** durch fehlende Multi‑Year Daten + fehlende trainierte Modell‑Gewichte.

---

## 2) Roadmap‑Verifikation bis Schritt 9.2 (Soll/Ist)

### Schritt 0 — Fixe Spezifikation
**Soll:** BTCUSDT Perp, 15m Decision / 3m Intrabar, Limits, Fees, SL‑first, no‑hedge.  
**Ist:** In `TradingConfig` (Fallback in `perp_env.py` und/oder `data_contract.py`) sind viele Konstanten vorhanden.

**Auffälligkeit:** Es existieren zwei „Config‑Welten“:
- `data_contract.py` (intended)
- Fallback‑`TradingConfig` in `perp_env.py` bei ImportError

➡️ **Risiko:** Notebook/Colab‑Runs können unbemerkt im Fallback laufen → Drift.

**Status:** ⚠️ teilweise ok, aber Konfig‑Single‑Source fehlt.

---

### Schritt 1 — Setup (Training vs Execution)
**Soll:** klare Ordnerstruktur + Artefakt‑Exports.  
**Ist:** `TraderHimSelf/` hat viele der Ordner (data_raw, data_processed, env, forecast, policy, runs, notebooks …).

**Status:** ✅ Struktur vorhanden.

---

### Schritt 2 — Data Contract
**Soll:** CandleRecord/FundingRecord Schema + gleiche Feature‑Engine offline/live.

**Ist:**
- `data_contract.py` existiert. ✅
- Feature‑Parity Script existiert: `test_feature_engine_parity.py` ✅

**Status:** ✅ Code‑seitig vorhanden.

---

### Schritt 3 — Datenquellen
**Soll:** Multi‑Year Binance 15m/3m (2019→heute), Live später Bitget.

**Ist:** Aktuell liegen nur **4 Tage Mock‑Daten (2024‑01‑01 → 2024‑01‑04)** vor.

**Status:** ❌ Roadmap‑Ziel nicht erfüllt (Datenumfang zu klein).

---

### Schritt 4 — Dataset Builder
**Soll:** Alignment 15m↔3m (5 Subbars), Funding mapping, No‑Leak, UTC, Save processed.

**Ist:** `build_dataset.py` implementiert strict grid, `is_missing`, `slot_15m`, Funding step‑wise mapping.

**Verifiziert (Artefakte):** aligned_15m/aligned_3m/funding existieren ✅

**Status:** ✅ implementiert + ausgeführt (für kleine Mock‑Daten).

---

### Schritt 5 — Feature Engine (28 Features + Scaler + Parity)
**Soll:** `features.parquet` + `scaler.pkl` + Parity‑Unit‑Test.

**Ist:** `feature_engine.py` enthält:
- `FEATURE_COLUMNS` (28) exakt nach Roadmap ✅
- Build‑Pfad: `data_processed/features.parquet` ✅
- Scaler‑Pfad: `data_processed/scaler.pkl` ✅

**Update:** Artefakte wurden inzwischen erzeugt (`features.parquet`, `scaler.pkl`) — aktuell allerdings nur auf Short/Mock‑Daten.

**Status:** ✅ Code fertig + Smoke‑Run ok; ⚠️ für Roadmap‑Ziel (Fit 2019–2023) braucht’s echte Historie.

---

### Schritt 6 — Trading Environment (PerpEnv)
**Soll:** 15m decision + 3m intrabar (SL‑first), fees, slippage, liquidation, funding, multi‑lots.

**Ist (per `env/perp_env.py`):**
- T+1 Execution (Trade am nächsten Open) ✅ (Time‑travel Bug aus 9.2 Report adressiert)
- Liquidation: Margin wird einmalig abgezogen (Double‑count fix) ✅
- Intrabar Simulation: O(1) Slice `idx_start = next_step_idx*5` ✅
- SL‑first implementiert ✅
- Fees (taker) ✅
- Slippage (Entry + Timeout exit) ✅

**Update (Fix‑Run):**
1) Funding ist jetzt implementiert (per‑step über `funding_rate`, fallback via merge_asof aus funding_df). ✅
2) Intrabar‑Mapping nutzt jetzt `slot_15m` (statt rein positional `*5`) und ist damit robust gegen Slicing/Intersection. ✅
3) Missing/NaN Handling: wenn 15m/3m Daten fehlen → konservatives Verhalten (keine neuen Opens, fallback checks). ✅

**Status:** ✅ Roadmap‑Features sind implementiert; ⚠️ correctness muss mit realer Historie + längeren Runs validiert werden.

---

### Schritt 7 — Risk Manager Wrapper
**Soll:** hard caps + no‑hedge + exposure clamp + entry penalty.

**Ist:** `env/risk_manager.py` implementiert validate_action + entry penalty.

**Lücke:** Exposure‑Cap **blockt nur**, clamp der **Positionsgröße** (margin) wird nicht sauber zurückgegeben. In `perp_env.py` wird margin „indirekt“ geclampt über `available_exposure*size_pct`, aber RiskManager alleine liefert keine size clamp. ⚠️

**Status:** ⚠️ brauchbar, aber nicht vollständig Roadmap‑konform als *Wrapper*.

---

### Schritt 8 — Forecast Modell (PatchTST) + Precompute
**Soll:** Input = 28 normalisierte Core‑Features (Lookback 512), Targets multi‑horizon quantiles, Precompute 35D Forecast‑Feature‑Block.

**Ist (train_patchtst.py):**
- 512 lookback / 192 horizon / Quantile loss / Feature‑Block 35D ✅ (Konzept)

**Update (Fix‑Run):**
1) Input‑Quelle ist jetzt korrekt: Core‑Inputs kommen aus `features.parquet` (28D, `FEATURE_COLUMNS`), Close‑Serie wird aus `aligned_15m.parquet` aligned. ✅
2) Spaltennamen sind vereinheitlicht: `forecast_0..forecast_34` (kein `fc_feat_*` mehr). ✅
3) Precompute schreibt full‑length Output mit NaN‑Padding für die Lookback‑Rows. ✅

**Weiterhin Limitierung:** Daten sind aktuell zu kurz (Mock/4 Tage), und `forecast_model.pt` fehlt → die erzeugten Forecast‑Features sind ohne Training inhaltlich nicht brauchbar.

**Status:** ✅ Pipeline‑Code lauffähig; ⚠️ Training/Signalqualität blocked.

---

### Schritt 9 — PPO Training (Policy)
**Soll:** PPO trainiert auf Obs=72 (28 core + 35 forecast + 9 account), saubere Pipeline, system‑check, bug‑hunt.

**Ist (train_ppo.py):**
- Obs‑Dim 72 stimmt im Wrapper ✅
- Merged Core+Forecast in df_15m vorgesehen ✅

**Update (Fix‑Run):**
1) `features.parquet` + `scaler.pkl` wurden erzeugt (Smoke‑Run). ✅
2) Forecast‑Merge ist gehärtet: Rename `fc_feat_*`→`forecast_*` + fail‑fast wenn Spalten fehlen (optional `--allow-dummy-forecast`). ✅
3) 3m‑Daten werden via `slot_15m` auf den 15m‑Zeitraum gefiltert (reduziert Mapping‑Drift). ✅

**Status (Training):** ✅ load/merge‑Pfad lauffähig; ⚠️ echtes PPO‑Training weiterhin blocked (Multi‑Year Daten + trainierte Forecast‑Weights fehlen).

---

### Schritt 9.1 — System Check
**Ist:** `system_check_report.md` existiert ✅

**Aber:** Der Report enthält Punkte, die inzwischen überholt sind (z.B. Dependencies in requirements). `requirements.txt` enthält scikit‑learn/joblib bereits ✅.

**Status:** ✅ report vorhanden.

---

### Schritt 9.2 — Bug‑Hunting
**Ist:** `audit_report_9.2.md` + `final_bug_audit_report.md` existieren ✅

**Status:** ✅ Audit‑Doku vorhanden.

**Fazit Step 0‑9.2:**
- „Bearbeitet“ im Sinne von: **Code + Reports existieren**: größtenteils ja.
- „Jede kleinste Aufgabe gemacht“ im Sinne von: **Pipeline lauffähig + Artefakte + Roadmap‑Parity**: **nein**.

---

## 3) Zusätzliche Bug‑/Logik‑Analyse (über die Reports hinaus)

### 3.1 Blocker‑Bugs (müssen vor RL‑Training weg)

#### (B1) Forecast‑Pipeline (Step 8) — ✅ gefixt (Fix‑Run 2026‑02‑10)
- Input‑Quelle ist jetzt korrekt: `features.parquet` + `FEATURE_COLUMNS` (28D) + Close aligned aus `aligned_15m.parquet`.
- Keine „erste 28 Spalten“ Heuristik mehr (kein Dim‑Mismatch).

#### (B2) Forecast‑Feature‑Spaltennamen mismatch — ✅ gefixt
- Precompute schreibt jetzt `forecast_0..forecast_34` (kein `fc_feat_*`).
- PPO akzeptiert `forecast_*` und kann optional legacy `fc_feat_*` sauber umbenennen.

#### (B3) Dataset‑Länge / Lookback‑Unmöglichkeit — ⚠️ weiterhin Blocker
Mit aktuellen Mock‑Daten (384×15m) kann Lookback 512 nicht sinnvoll laufen.

**Fix:** echte Multi‑Year Historie laden (2019→), dann Pipeline neu bauen.

---

### 3.2 Env‑Korrektheit (Step 6)

#### (E1) Funding — ✅ implementiert
Funding wird jetzt pro Step angewandt (aus `funding_rate` in df_15m oder via `funding_df` asof‑Mapping).

#### (E2) Missing‑Data / NaNs — ✅ konservativ gehandhabt
Wenn 15m/3m Daten fehlen oder `is_missing`/NaNs auftreten → keine neuen Opens; Intrabar‑Fallback (konservativ) statt NaN‑Propagation.

#### (E3) 15m↔3m Mapping — ✅ robust über `slot_15m`
Intrabar‑Subbars werden über `slot_15m` gemappt (statt positional `next_step_idx*5`).

---

## 4) Empfohlene Next‑Steps (damit’s wirklich „e2e“ + trainierbar wird)

1) ⬜ **Multi‑Year Binance Daten (2019+) laden** (Step 3) — aktuell größter Blocker.
2) ⬜ `build_dataset.py` nochmal auf echter Historie laufen lassen (aligned_15m/aligned_3m/funding groß genug).
3) ⬜ `feature_engine.py build` nochmal laufen lassen (Scaler Fit 2019–2023, keine Mock‑Fallbacks).
4) ⬜ PatchTST **trainieren** → `models/forecast_model.pt`.
5) ⬜ `train_patchtst.py precompute` erneut (Forecast‑Features dann sinnvoll).
6) ⬜ PPO **trainieren** → SB3‑Artefakte (z.B. `ppo_policy.zip`).

---

## 5) Minimal‑Tests (Pflicht, bevor wir irgendwas glauben)

1) **Alignment‑Test (15m↔3m):**
   - Für random step k: alle 5 subbars müssen zu slot_15m==k gehören.

2) **Forecast‑Contract‑Test:**
   - `forecast_features.parquet` muss genau 35 Spalten haben, exakt benannt.

3) **PnL‑Invarianten:**
   - open+close bei gleichem Preis → Verlust = fees + slippage (konservativ)
   - liquidation → max loss ~= margin_used (+ fees), nicht mehr

4) **NaN‑Test:**
   - Keine NaNs in Obs (core+forecast). Sonst Training unbrauchbar.

---

## 6) Schlussfazit (brutal ehrlich)

- Bis **Schritt 9.2** ist viel „bearbeitet“ (Code + Reports existieren). ✅
- Aber: „jede kleinste Aufgabe gemacht“ (Roadmap‑Parity + Artefakte + lauffähige End‑to‑End Pipeline) ist **nicht** erfüllt. ❌

**Die drei größten Blocker aktuell:**
1) Daten zu kurz (4 Tage) vs Lookback 512/Buffer 800
2) Forecast Step 8 ist im aktuellen Zustand **nicht lauffähig** (Input‑Spalten / Dim mismatch)
3) Env: Funding fehlt + Mapping fragil

Wenn wir diese 3 Dinge fixen, ist Step 9 PPO Training erst „realistisch“.
