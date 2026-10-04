# Configs

- `stage1_base.json`, `stage2_phys.json`, `stage3_thermal.json`: the exact
  command-line arguments recorded in the paper's training checkpoints, plus the
  selected epoch and validation loss. The defaults of `scripts/02_train_base.py`,
  `scripts/03_finetune_phys.py` and `scripts/04_finetune_thermal.py` equal these values.
- `inference.json`: the inference settings behind each table (also available as
  `--preset` in `scripts/05_infer_full_texas.py`).
