# Colab User Guide

## Hardware Profiles

The training scripts support two hardware profiles: `default` and `high-util`.

### Default Mode
Designed for safety and compatibility (e.g., standard Colab GPU, T4).
- Safe batch sizes (32/64).
- Conservative worker counts.
- No AMP by default.

### High-Util Mode (`--profile high-util`)
Designed for high-performance GPUs (A100, L4) to maximize throughput.
- Increased batch sizes (256/512).
- AMP (Automatic Mixed Precision) enabled.
- Persistent workers and pin memory enabled.
- `torch.compile` enabled (if supported).

**Usage:**
```bash
python forecast/train_patchtst.py train --profile high-util
python policy/train_ppo.py --profile high-util
```

**Warning:**
If you encounter Out-Of-Memory (OOM) errors, try:
1. Reverting to `default` profile.
2. Manually reducing batch size: `--batch-size 128` (or lower).
