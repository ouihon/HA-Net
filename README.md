# Beam Tracking Reference Implementation

This directory contains a compact PyTorch implementation for sequence-based beam tracking experiments prepared for paper artifact review. The code focuses on the training pipeline used in our internal study and is released in a limited form for academic evaluation.

## Scope

- Included: model definition, configuration loader, teacher-forcing training script, and checkpoint export.
- Not included: full dataset, preprocessing pipeline, cluster scripts, and broader experiment management utilities.
- The repository is intended as a reference implementation rather than a turnkey reproduction package.

## Files

- `beam_tracking_config.py`: experiment configuration and RSRP normalization utilities.
- `beam_tracking_model_m.py`: two-layer LSTM predictor with optional temporal attention and station embeddings.
- `beam_tracking_train_TF.py`: dataset wrapper, loss definition, training loop, validation logic, and CLI entry point.

## Environment

Recommended environment:

- Python 3.10+
- PyTorch
- NumPy
- scikit-learn
- tqdm

Example installation:

```bash
pip install torch numpy scikit-learn tqdm
```

## Data Interface

The training script expects:

- a directory of `.npz` trajectory files passed by `--train_data_dir`
- a neighbor mapping JSON passed by `--neighbors_path`

Each `.npz` file must contain an `rsrp_dbm` array with time-major beam measurements. The script derives historical windows and future supervision targets directly from that tensor.

## Training

Example command:

```bash
python beam_tracking_train_TF.py \
  --train_data_dir /path/to/data \
  --neighbors_path /path/to/neighbors.json \
  --save_dir checkpoints
```

By default, the script trains the LSTM-based beam tracker with:

- top-`K` supervised regression on normalized beam scores
- auxiliary link-quality regression
- KL regularization on the beam distribution
- rollout-based validation metrics

Run `python beam_tracking_train_TF.py --help` for the complete list of options.

## Outputs

The training script writes:

- intermediate checkpoints every 50 epochs
- `best_roll_hit_model.pth`
- `best_val_loss_model.pth`
- `final_model.pth`
- `training_history.json`

## Release Note

This code release is intentionally limited. It is suitable for inspecting the model structure and training procedure reported in the paper, but it does not expose proprietary datasets, internal tooling, or the full experimental stack.

If you plan to cite this implementation in a submission, treat it as the paper-aligned reference code snapshot associated with the reported method.
