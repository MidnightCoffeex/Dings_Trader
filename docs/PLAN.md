# dings-trader — Plan

Private research project (Maxim + Dings).

**Canonical GitHub Repo:** https://github.com/MidnightCoffeex/Dings_Trader

## Goal
Build a private tool for BTC trading research:
- ingest public candle data
- engineer useful features
- train a model that suggests **long / short / hold** (with confidence)
- backtest with fees+slippage and **max leverage x10**
- present everything in a modern dark UI (purple accent)

## Non-goals / guardrails
- no guaranteed profits
- no autonomous trading by default
- ship as a private tool first; productization later only if it actually works

## Repo layout
- `ml/` — data ingest, features, training, backtests (modell-projekt / training lives here)
- `ui/` — web dashboard (zeigt nur ergebnisse; kein modell-auswahl/training in der UI)
- `data/` — parquet datasets
- `docs/` — this plan + todos + decisions

## Current status
- scaffolded project dirs + `ml/fetch_candles.py` (Binance public klines → parquet)
- UI skeleton work started via subagent (Next.js/Tailwind/shadcn concept)
