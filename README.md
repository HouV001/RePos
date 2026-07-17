# RePos

Relative-to-absolute output factorization for cross-environment WiFi CSI-based 3D human pose estimation.

RePos factors 3D pose prediction into two parts: a root-relative pose, decoded from a structured body-part-aware representation, and a root position, estimated by a separate amplitude-based branch. Keeping absolute-position supervision out of the structure branch improves generalization to unseen environments. A single-stage variant, RePos-D, targets in-domain deployment where direct absolute regression is sufficient.

## Installation

```bash
conda create -n repos python=3.10 -y
conda activate repos

# Install PyTorch for your CUDA version (see https://pytorch.org)
pip install torch

pip install -r requirements.txt
```

## Data preparation

### MM-Fi (cross-environment)

Download the [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi) dataset. The raw layout is
`E0X/SXX/AXX/wifi-csi/*.mat` with a `ground_truth.npy` per sequence.

Preprocess one set of arrays per environment for leave-one-environment-out training:

```bash
python scripts/preprocess_per_env.py \
    --data_root /path/to/MMFi_Dataset \
    --output_dir ./data/per_env \
    --protocol protocol3
```

### Person-in-WiFi-3D (in-domain)

```bash
python scripts/preprocess_wifipose.py \
    --data_root /path/to/PersonInWiFi3D \
    --output_dir ./data/wifipose3d
```

## Training

RePos is trained in two stages: Stage 1 learns the root-relative pose, and Stage 2 learns
the root position with Stage 1 frozen.

### Cross-environment (MM-Fi, held-out E04)

```bash
# Stage 1: structured relative pose
python train_v2.py --phase 1 --dataset mmfi --data_root ./data/per_env \
    --loeo_test_env E04 --epochs 100

# Stage 2: root localization
python train_v2.py --phase 2 --dataset mmfi --data_root ./data/per_env \
    --loeo_test_env E04 --phase1_ckpt pretrained/<stage1_run>/v2_phase1_best.pth --epochs 50
```

Set `--loeo_test_env` to `E01`, `E02`, `E03`, or `E04` to choose the held-out environment.

### In-domain (Person-in-WiFi-3D, RePos-D)

```bash
python train_v2.py --phase 1 --dataset wifipose --data_root ./data/wifipose3d --epochs 100
python train_v2.py --phase 2 --dataset wifipose --data_root ./data/wifipose3d \
    --phase1_ckpt pretrained/<stage1_run>/v2_phase1_best.pth --epochs 50
```

## Evaluation

```bash
python evaluate_v2.py --checkpoint pretrained/<stage2_run>/v2_phase2_best.pth \
    --data_root ./data/per_env --loeo_test_env E04 --phase 2
```

Add `--latex` for a LaTeX table, `--per_activity` for a per-activity breakdown, or
`--info_only` to print the metrics stored in a checkpoint without re-running.

## Pretrained models

The four main-table checkpoints are available on
[Google Drive](https://drive.google.com/file/d/1dVdqI_03vdd7MigaQoHUrr5l3XVMFzDg/view?usp=drive_link)
(`RePos_main_checkpoints.zip`, 190 MB). Each MM-Fi checkpoint contains the
complete model (structure branch + ASPN), so no separate Stage-1 file is needed.

| Checkpoint | Benchmark | MPJPE | PA-MPJPE | PCK@50 |
|---|---|---|---|---|
| `mmfi_cross_scene/repos_mmfi_p1.pth` | MM-Fi Protocol 1 (E01–E03 → E04) | 254.4 | 100.5 | 80.0 |
| `mmfi_cross_scene/repos_mmfi_p2.pth` | MM-Fi Protocol 2 (E01–E03 → E04) | 284.1 | 99.3 | 64.9 |
| `mmfi_cross_scene/repos_mmfi_p3.pth` | MM-Fi Protocol 3 (E01–E03 → E04) | 296.1 | 102.0 | 66.8 |
| `piw3d_repos_d/repos_d_piw3d.pth` | Person-in-WiFi-3D (RePos-D, in-domain) | 86.9 | 57.5 | 91.4 |

```bash
# verify stored metrics without re-running
python evaluate_v2.py --checkpoint repos_mmfi_p3.pth --info_only
```

Note: loading with the current code reports a few missing/unexpected keys
(SGA attention-mask buffers); these are deterministic, rebuilt at init, and
do not affect results.

## Repository layout

```
models/       RePos model (structured pose decoder, skeleton graph attention,
              amplitude-based position branch, losses)
datasets/     MM-Fi and Person-in-WiFi-3D loaders
scripts/      dataset preprocessing
utils/        evaluation metrics (MPJPE, PA-MPJPE, PCK)
configs/      reference hyperparameters
train_v2.py   training entry point
evaluate_v2.py evaluation entry point
```

## Citation

```bibtex
@article{hou2026repos,
  title   = {RePos: Relative-to-Absolute Output Factorization for Cross-Environment WiFi-Based 3D Human Pose Estimation},
  author  = {Hou, Zhangcheng and Ohtsuki, Tomoaki},
  year    = {2026}
}
```
