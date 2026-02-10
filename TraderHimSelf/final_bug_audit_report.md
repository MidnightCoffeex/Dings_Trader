# Final Bug Audit Report (Schritt 9.2 — Bug-Hunting)

Pfad: `TraderHimSelf/final_bug_audit_report.md`

Kontext-Basis: `docs/TRAINING_ROADMAP.md` (v5 final)

## Executive Summary
Bei der erneuten Prüfung (Roadmap-konform: SL-first, 15m Decision + 3m Intrabar, Multi-Lots bis 10, isolated linear USDT-M, taker fees) sind mehrere **kritische Inkonsistenzen** und **mathematische/logische Fehler** aufgefallen – primär im Zusammenspiel von `build_dataset.py` ↔ `train_ppo.py` ↔ `perp_env.py` sowie im Forecast-Pipeline-Skript.

Die größten Risiken:
1. **Zeitachsen-/Alignment-Bug:** `perp_env.py` nimmt implizit an, dass `df_3m` nach `reset_index(drop=True)` exakt in 5×-Reihenfolge zu `df_15m` steht (idx_start = (t+1)*5). Das ist nach dem Merge/Intersection in `train_ppo.py` sehr wahrscheinlich **falsch** ⇒ Intrabar-Simulation nutzt falsche 3m-Bars (Time-Travel/Random-Mismatch).
2. **Funding komplett nicht implementiert** in `perp_env.py` (Roadmap 6.4/6.4. funding event-basiert). Das verändert Reward/PnL signifikant.
3. **Observation/Feature-Contract-Bruch:** `train_ppo.py` erwartet Forecast-Feature-Spalten `forecast_0..forecast_34`, aber `train_patchtst.py` schreibt `fc_feat_0..fc_feat_34` ⇒ Forecast-Features werden i.d.R. **nicht gemerged** (still & silent), PPO trainiert auf Dummy/fehlenden Forecast.
4. **Forecast Input-Channel Bug:** `train_patchtst.py` nimmt `df.columns[:28]` als Features; `aligned_15m.parquet` beginnt aber typischerweise mit OHLCV (+ Flags) und nicht mit den 28 skalierten Core-Features ⇒ Modell bekommt falsche Inputs.
5. **Positions-Margin Accounting inkonsistent (Balance/Equity):** Beim Öffnen wird Margin **nicht** vom Balance abgezogen, Exposure wird aber via margin_used begrenzt. Das ist ok als „isolated margin not deducted“-Modell, aber dann muss Liquidation/close accounting konsistent sein. Aktuell wird bei Liquidation `balance -= margin_used` gemacht, obwohl Margin nie abgezogen wurde. Das ist konsistent *nur wenn* man Margin als Teil der Balance betrachtet. Gleichzeitig werden Fees aber sofort von Balance abgezogen. Diese Mischform ist möglich, aber erfordert saubere Definition und Tests.

Im Folgenden: detaillierte Findings + Fix-Vorschläge.

---

## Roadmap Requirements Check (Soll vs. Ist)

### SL-first
- `data_contract.Position.check_exit()` korrekt SL-first.
- `perp_env.py` Intrabar: SL-check vor TP-check ⇒ korrekt SL-first.

### 15m Decision / 3m Intrabar (5 Subbars)
- `build_dataset.py` baut strikten 3m-Grid inkl. NaNs, zusätzlich `slot_15m`.
- `perp_env.py` **ignoriert** `slot_15m` und **assumiert** 5× Sequenz-Alignment.
- `train_ppo.py` schneidet `df_15m` per `common_idx`, `df_3m` bleibt unverändert ⇒ Alignment bricht.

### Multi-Position bis 10 Lots
- `perp_env.py`: Liste `open_positions`, max check vorhanden.
- No-hedge rule in `perp_env.py`: blockt opposite direction wenn Positionen offen.

### Fees taker-only
- `perp_env.py`: entry fee und exit fee mit `taker_fee` umgesetzt.
- Edge: Close-on-timeout nutzt ebenfalls `taker_fee`.

### Funding (event-based)
- `build_dataset.py` mappt funding step-wise.
- `feature_engine.py` nutzt funding_rate im Feature-Vektor.
- `perp_env.py`: Funding block ist `pass` ⇒ **nicht umgesetzt**.

---

## Findings per File

## 1) `data_contract.py`
### (A) Contract drift vs Env
- Enthält `Position` Dataclass (side LONG/SHORT, size, leverage, stop_loss/take_profit) die **nicht** direkt der EnvLot-Struktur entspricht.
- Nicht kritisch, aber **potenziell gefährlich**, wenn später Live/Env/Backtest Position-Objekte verwechselt werden.

### (B) `CandleRecord.from_dataframe_row`
- nutzt `row.get` und `row['open']...` gemischt. Bei pandas Series ist `.get` ok, aber Zugriff über `row['open']` wirft KeyError wenn Spalten fehlen.
- Minor.

---

## 2) `build_dataset.py`
### (A) Funding mapping: potentielle Schema-Mismatch
- Input erwartet `btcusdt_funding.parquet` mit `time_ms`/`funding_rate`. Binance API Rohdaten haben oft `fundingTime`/`fundingRate`.
- Wenn upstream nicht angepasst: Load/prepare_funding bricht.

### (B) 3m Range End
- 3m grid geht bis `end_time + 12m`. Das ist korrekt für „letzte 15m Candle braucht 5 subbars“.
- Aber `validate_alignment` checkt `slot_15m.max() > aligned_15m.max() + 15m` (komisch toleriert) – hier könnte ein Randfall knapp sein.

### (C) Reindex preserves NaNs but env not handling
- Missing candles bleiben NaN und `is_missing` flag gesetzt.
- Downstream (`perp_env.py`) **nutzt** `is_missing` nicht; könnte NaN high/low/close verursachen (SL/TP/Liq math -> NaN propagate) ⇒ Reward/Equity kann NaN werden.

**Fix:** In Env beim Lesen von subbars/15m Bars Missing-Flags prüfen und entweder:
- Step skip / force flat
- oder fehlende subbars ersetzen (konservativ) und Flag ins Obs aufnehmen

---

## 3) `feature_engine.py`
### (A) Funding time_to_next_funding_steps off-by-one
- `steps_left = steps_total - steps_in_cycle` bei step_in_cycle==0 ergibt 32. „Time to next funding“ an Funding-Time selbst sollte **0** sein, nicht 32.
- Beispiel: 00:00 UTC, hour_mod=0, minute_steps=0 => steps_in_cycle=0 => steps_left=32.
- Roadmap: capped 0..32, aber semantisch eher 0..32 inclusive. Trotzdem: *nächstes Funding in 8h* = 32 Schritte; wenn genau am Funding-Zeitpunkt, dann „bis zum nächsten Funding“ tatsächlich 32 (weil nächstes Event in 8h). Wenn Feature „time_to_next_funding“ meint, ist 32 ok. Wenn es „time_to_current funding“ meint, wäre 0.
- Empfehlung: in Roadmap definieren. Aktuell plausibel, aber muss konsistent mit Live.

### (B) Train window tz-awareness
- `train_start = pd.Timestamp(cfg.train_start)` erzeugt tz-naive Timestamp.
- Features Index ist tz-aware (UTC). Vergleich tz-aware vs naive kann warnings/Fehler liefern.

**Fix:** `pd.Timestamp(..., tz='UTC')` oder `.tz_localize('UTC')`.

---

## 4) `env/perp_env.py` (kritisch)

### (A) KRITISCH: 3m Alignment Annahme bricht nach common_idx merge
- Env macht `df_15m.reset_index(drop=True)` und `df_3m.reset_index(drop=True)`.
- Intrabar range: `idx_start = next_step_idx * 5`.
- In `train_ppo.py` wird `df_15m = df_15m.loc[common_idx]` (Index-Schnitt) => reduziert/verschiebt 15m Reihen.
- `df_3m` wird **nicht** analog gefiltert. Dadurch referenziert `idx_start` völlig falsche 3m Bars.

**Impact:**
- SL/TP/Liq triggers basieren auf falschen Preisen ⇒ Reward/PnL unbrauchbar.
- Potenzieller Time-Travel, wenn 3m-Bars aus anderer Zeit genutzt werden.

**Fix (robust):**
- Nicht `reset_index(drop=True)`; stattdessen DatetimeIndex behalten.
- In env: aus `next_candle.open_time_ms` einen UTC Timestamp machen und 5 erwartete 3m timestamps holen oder per `slot_15m` joinen.
- Alternativ: `build_dataset.py` soll eine explizite Mapping-Struktur schreiben (z.B. `subbar_start_idx` pro 15m-row) und env nutzt diese.

### (B) Funding in Env nicht implementiert
- Block `# Funding (Event based) pass`.

**Impact:** PnL/Reward drift vs Roadmap.

**Fix:**
- Funding pro Lot abhängig von side/qty/notional und funding_rate anwenden, wenn Funding Event zwischen prev_step und next_step liegt (oder step-wise je 15m, aber event-based).

### (C) Time-in-trade handling unsauber
- Während 3m loop: `lot.time_in_trade_steps_15m += 0.2` und danach `math.ceil`.
- Ceil nach jedem step macht 0.2->1 sofort; effektiv zählt jedes 15m-step als 1 (ok), aber die intrabar increments sind nutzlos und können Off-by-one in Timeout erzeugen.

**Fix:** pro 15m-step `+=1` einmal am Ende; intrabar keine fractional steps.

### (D) Missing data / NaN propagations
- Keine Guards gegen NaN in `sub_bar` oder `next_close`.

**Fix:** wenn NaNs: konservativ force-close/hold und reward=0 oder -penalty.

### (E) SL/TP Distanz kann negativ/unsinnig werden
- ATR fallback `current_close*0.01` ok.
- Aber falls ATR NaN/0: sl_dist=0, tp_dist=0 ⇒ SL/TP gleich entry => sofortige triggers.

**Fix:** `current_atr = max(eps, ...)`.

### (F) Reward uses `prev_equity` state with brittle init
- `prev_equity = getattr(self,'prev_equity',10000.0)` und special-case if step==BUFFER_STEPS.
- Wenn reset mit start_step option != BUFFER_STEPS, init kann falsch sein.

**Fix:** set `self.prev_equity = self.equity` in reset.

### (G) Risk penalty uses ATR/price scaling
- `risk_penalty = 0.1 * abs(notional_open_pct) * (current_atr/next_close)`.
- notional_open_pct bereits >=0, `abs` redundant.
- ok, aber ensure next_close>0.

### (H) Close at SL/TP uses trigger price without adverse slippage
- Roadmap: slippage conservatively against you. Entry slippage is against you, exits are not.

**Fix:** Apply adverse slippage on exits too (esp. SL).

### (I) Liquidation model simplification
- Uses MMR=0.005 and condition `margin_used + pnl_at_low <= mmr*notional`.
- This is a proxy. But ensure sign correctness.
- Seems plausible.

---

## 5) `env/risk_manager.py`
### (A) Exposure clamp vs env behavior mismatch
- Roadmap 7.1.1: clamp new_margin_used to available_exposure.
- RiskManager currently only sets direction to flat if available_exposure<=0; does **not** clamp size.
- Env itself clamps by `desired_margin = available_exposure * size_pct` (so it clamps). If RiskManager used in live, clamp missing.

**Fix:** RiskManager should also output `max_allowed_margin` or clamp `size_pct` given desired margin.

---

## 6) `forecast/train_patchtst.py` (kritisch)

### (A) KRITISCH: Uses wrong input features
- `DATA_FILE_15M = aligned_15m.parquet` (raw candles) and `self.feature_cols = df.columns[:28]`.
- This almost certainly includes OHLCV/open_time_ms/is_missing etc, not the **28 scaled core features**.

**Fix:** Load `data_processed/features.parquet` (scaled) and explicitly use `FEATURE_COLUMNS` from `feature_engine.py`.

### (B) Forecast feature column naming mismatch
- Saves `fc_feat_{i}` but PPO expects `forecast_{i}`.

**Fix:** Standardize names in both places.

### (C) Dummy close/target logic mismatch risk
- In Dataset: uses `current_close = close_prices[t-1]`, future closes = `close[t:t+H]`. Comment acknowledges potential off-by-one.
- Roadmap: targets y_{t,h} = log(close_{t+h}/close_t). Here it uses close_{t-1} for denominator. That is **off by one**.

**Fix:** Define t as last index in input; if x uses [t-lookback+1..t], denominator close[t]. Otherwise shift accordingly.

### (D) quantile_loss weighting broadcast bug
- `loss_q.unsqueeze(-1) * w` where loss_q is (B,H) and w is (1,H,1) -> results (B,H,H?) likely broadcast incorrectly.

**Fix:** make w shape (1,H) and multiply loss_q * w then mean.

### (E) PatchTST head extremely large
- flatten_dim = 28*n_patches*d_model; with lookback512 patch16 stride8 => n_patches= (512-16)//8+1=63. flatten_dim=28*63*128=225,792; head linear to 512 heavy but ok.
- not a bug, but memory.

---

## 7) `policy/train_ppo.py`

### (A) KRITISCH: Forecast merge mismatch
- Expects `forecast_0..forecast_34`; precompute writes `fc_feat_0..`.
- In merge loop: if column missing does `pass` silently.

**Impact:** PPO likely never sees forecast features.

**Fix:** fail fast if expected columns missing; or auto-map first 35 columns from forecast parquet.

### (B) Index alignment mismatch with env reset_index
- `common_idx` intersection may shrink/skip; env resets index => loses true timestamps.
- Combined with 3m assumption => major.

### (C) df_3m not aligned to common_idx span
- Should be sliced to matching time range (and ideally strict 5 per 15m slot). Currently not.

### (D) SB3 action space mismatch risk
- Env action_space Box(-1,1,5) is okay.
- RiskManager wrapper not used; env enforces rules.

### (E) Missing data handling
- If features have NaNs (rolling windows), they are included. PPO will ingest NaNs (bad).

**Fix:** trim dataset to valid rows where core+forecast non-NaN; or impute.

---

## Cross-Module Inconsistencies

1) **Timestamp handling:** build_dataset/feature_engine operate on UTC DatetimeIndex; env resets to RangeIndex and uses positional mapping.
2) **Forecast pipeline uses different input source and column naming.**
3) **Funding** is in dataset+features but not in environment PnL.

---

## Recommended Fix Order (highest ROI)

1) **Fix env 15m↔3m mapping** (no positional assumptions; use timestamps / slot_15m / precomputed mapping). Add asserts that each 15m step has 5 subbars (or handle missing conservatively).
2) **Implement funding in env** with event-based logic consistent with dataset funding.
3) **Standardize forecast feature column names and merge logic** (fail fast; no silent pass). Use explicit `FEATURE_COLUMNS` for PatchTST inputs.
4) **Fix PatchTST target indexing (off-by-one)** and quantile_loss weighting broadcasting.
5) **NaN policy**: remove initial 672-window rows or fill; ensure PPO never sees NaNs.

---

## Minimal Safety Tests to Add

1) Alignment test:
- For random k, verify that env sub_candles timestamps match [T+0..T+12] minutes and belong to slot_15m==T.

2) PnL invariants:
- Open then immediately close at entry (no movement) => loss == fees + slippage.

3) Liquidation invariant:
- If liquidated, realized loss approx == margin_used (+ fees) and position removed.

4) Forecast features contract:
- Assert forecast_features parquet contains exactly 35 columns with expected names.

---

## Conclusion
Der Code ist in vielen Teilen nah an der Roadmap, aber die **Pipeline-Integrität** ist aktuell nicht gegeben: Intrabar-Simulation und Forecast-Features laufen mit hoher Wahrscheinlichkeit auf falschen Daten/Spalten. Ohne Fix dieser Contract/Alignment-Themen sind Trainingsresultate nicht vertrauenswürdig.
