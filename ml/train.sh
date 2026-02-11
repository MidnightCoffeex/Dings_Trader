#!/bin/bash
# Wrapper to run LightGBM training
cd "$(dirname "$0")"
python3 train_lgbm.py --data ../data/btcusdt_5m.parquet --out ../data/preds_lgbm_5m.parquet
