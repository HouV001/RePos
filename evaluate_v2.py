"""
Evaluation Script for CSI-StructPose V2

Supports Phase 1 (root-relative) and Phase 2 (absolute with position prediction).

Metrics:
- MPJPE (Mean Per Joint Position Error)
- PA-MPJPE (Procrustes-Aligned MPJPE)
- PCK@10, 20, 30, 40, 50 (Percentage of Correct Keypoints)
- Root position error (Phase 2 only)
- Per-joint MPJPE table
- Body part error breakdown

Usage:
    # Evaluate Phase 1 (root-relative)
    python evaluate_v2.py --checkpoint pretrained/v2_phase1_best.pth --data_root ./data/cross_scene

    # Evaluate Phase 2 (absolute with physics-guided position)
    python evaluate_v2.py --checkpoint pretrained/v2_phase2_best.pth --data_root ./data/cross_scene --phase 2

    # Quick check saved metrics
    python evaluate_v2.py --checkpoint pretrained/v2_best.pth --info_only
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.csi_structpose_v2 import (
    CSIStructPoseV2,
    CSIStructPoseV2Phase2,
    build_csi_structpose_v2,
    build_csi_structpose_v2_phase2,
    load_phase1_into_phase2,
)
from collections import defaultdict
from utils.metrics import (
    PoseMetrics,
    format_results_table,
    format_latex_table,
    JOINT_NAMES,
    BODY_PARTS,
)

def print_checkpoint_info(ckpt_path: str):
    """Print saved metrics from checkpoint without running evaluation."""
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print('='*60)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    print(f"\nKeys in checkpoint: {list(ckpt.keys())}")
    print(f"Epoch: {ckpt.get('epoch', 'N/A')}")

    if 'best_mpjpe' in ckpt:
        print(f"\nBest MPJPE: {ckpt['best_mpjpe']:.2f} mm")

    if 'test_metrics' in ckpt:
        print(f"\nSaved Test Metrics:")
        for key, val in ckpt['test_metrics'].items():
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.2f}")
            elif isinstance(val, dict):
                print(f"  {key}: {val}")

    if 'config' in ckpt:
        print(f"\nConfig:")
        for k, v in ckpt['config'].items():
            print(f"  {k}: {v}")

    print('='*60)
    return ckpt

@torch.no_grad()
def evaluate_phase1(model, dataloader, device, verbose=True):
    """
    Evaluate V2 Phase 1 model (root-relative pose).

    Returns comprehensive metrics including PCK, MPJPE, PA-MPJPE.
    """
    model.eval()
    metrics = PoseMetrics()

    iterator = tqdm(dataloader, desc="Evaluating V2 Phase 1") if verbose else dataloader

    for batch in iterator:
        if len(batch) == 4:
            csi, _, pose_gt, _ = batch
        elif len(batch) == 3:
            csi, pose_gt, _ = batch
        else:
            csi, pose_gt = batch[:2]

        csi = csi.to(device)

        output = model(csi)
        pose_pred = output['pose'].cpu()

        metrics.update(pose_pred, pose_gt)

    results = metrics.compute()
    return results

@torch.no_grad()
def evaluate_phase2(model, dataloader, device, verbose=True):
    """
    Evaluate V2 Phase 2 model (CSIStructPoseV2Phase2 with dual-path architecture).

    Returns comprehensive metrics including:
    - Absolute pose metrics (MPJPE, PA-MPJPE, PCK)
    - Root position error
    - Relative pose metrics (Phase 1 structure, for comparison)
    """
    model.eval()

    metrics_absolute = PoseMetrics()
    metrics_relative = PoseMetrics()

    root_errors = []

    iterator = tqdm(dataloader, desc="Evaluating V2 Phase 2") if verbose else dataloader

    for batch in iterator:
        if len(batch) == 4:
            csi, _, pose_gt, _ = batch
        elif len(batch) == 3:
            csi, pose_gt, _ = batch
        else:
            csi, pose_gt = batch[:2]

        csi = csi.to(device)
        pose_gt_np = pose_gt.numpy()

        output = model(csi)
        pose_pred = output['pose'].cpu()
        phase1_pose = output['phase1_pose'].cpu()
        root_pred = output['root_position']

        gt_root = pose_gt_np[:, 0, :]

        root_pred_np = root_pred.cpu().numpy()
        root_error = np.linalg.norm(root_pred_np - gt_root, axis=-1) * 1000
        root_errors.append(root_error)

        metrics_absolute.update(pose_pred, pose_gt)

        pose_gt_rel = pose_gt_np - pose_gt_np[:, 0:1, :]
        metrics_relative.update(phase1_pose, torch.from_numpy(pose_gt_rel).float())

    results_absolute = metrics_absolute.compute()
    results_relative = metrics_relative.compute()

    all_root_errors = np.concatenate(root_errors)

    return {
        'absolute': results_absolute,
        'relative': results_relative,
        'root_error': {
            'mean': float(np.mean(all_root_errors)),
            'std': float(np.std(all_root_errors)),
            'median': float(np.median(all_root_errors)),
            'max': float(np.max(all_root_errors)),
        },
        'num_samples': results_absolute['num_samples']
    }

@torch.no_grad()
def evaluate_per_activity(model, dataloader, device, sample_ids, phase=1, verbose=True):
    """
    Evaluate model and group results by activity.

    Args:
        model: Trained model
        dataloader: Test dataloader (must have shuffle=False)
        device: torch device
        sample_ids: np.ndarray of sample IDs like "S01_A02_000"
        phase: 1 or 2

    Returns:
        Dict mapping activity_id -> {'mpjpe': float, 'std': float, 'count': int}
    """
    model.eval()

    all_errors = []
    iterator = tqdm(dataloader, desc="Per-activity eval") if verbose else dataloader

    for batch in iterator:
        if len(batch) == 4:
            csi, _, pose_gt, _ = batch
        elif len(batch) == 3:
            csi, pose_gt, _ = batch
        else:
            csi, pose_gt = batch[:2]

        csi = csi.to(device)
        output = model(csi)
        pose_pred = output['pose'].cpu().numpy()
        pose_gt_np = pose_gt.numpy()

        error = np.linalg.norm(pose_pred - pose_gt_np, axis=-1)
        per_sample_mpjpe = np.mean(error, axis=-1) * 1000
        all_errors.append(per_sample_mpjpe)

    all_errors = np.concatenate(all_errors)
    assert len(all_errors) == len(sample_ids), \
        f"Mismatch: {len(all_errors)} errors vs {len(sample_ids)} sample IDs"

    import re
    act_re = re.compile(r'^A\d+$')
    activity_errors = defaultdict(list)
    for i, sid in enumerate(sample_ids):
        parts = str(sid).split('_')
        activity = next((p for p in parts if act_re.match(p)), 'unknown')
        activity_errors[activity].append(all_errors[i])

    results = {}
    for act in sorted(activity_errors.keys()):
        errs = np.array(activity_errors[act])
        results[act] = {
            'mpjpe': float(np.mean(errs)),
            'std': float(np.std(errs)),
            'count': len(errs),
        }

    return results

ACTIVITY_NAMES = {
    'A01': 'Stretching',
    'A02': 'Walking',
    'A03': 'Jumping',
    'A04': 'Waving',
    'A05': 'Pointing',
    'A06': 'Shoulder rotation',
    'A07': 'Elbow flexion',
    'A08': 'Wrist rotation',
    'A09': 'Arm raises',
    'A10': 'Shoulder abduction',
    'A11': 'Hip abduction',
    'A12': 'Knee raises',
    'A13': 'Sitting',
    'A14': 'Standing up',
    'A15': 'Squat',
    'A16': 'Lunge',
    'A17': 'Calling',
    'A18': 'Drinking',
    'A19': 'Waving hand',
    'A20': 'Clapping',
    'A21': 'Rubbing hands',
    'A22': 'Thumb up',
    'A23': 'Turning around',
    'A24': 'Head rotation',
    'A25': 'Trunk rotation',
    'A26': 'Trunk flexion',
    'A27': 'Lateral bending',
}

PROTOCOL_ACTIVITIES = {
    1: ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18',
        'A19', 'A20', 'A21', 'A22', 'A23', 'A27'],
    2: ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
        'A15', 'A16', 'A24', 'A25', 'A26'],
}

def print_per_activity_results(results: dict):
    """Pretty print per-activity MPJPE results."""
    print(f"\n{'='*70}")
    print("Per-Activity MPJPE Breakdown")
    print('='*70)
    print(f"{'Activity':<6} {'Name':<22} {'Type':<8} {'MPJPE':>8} {'Std':>8} {'Count':>7}")
    print('-'*70)

    daily_acts = set(PROTOCOL_ACTIVITIES[1])
    rehab_acts = set(PROTOCOL_ACTIVITIES[2])

    daily_mpjpes, rehab_mpjpes = [], []

    for act_id, stats in sorted(results.items()):
        name = ACTIVITY_NAMES.get(act_id, '?')
        if act_id in daily_acts:
            act_type = 'Daily'
            daily_mpjpes.append(stats['mpjpe'])
        elif act_id in rehab_acts:
            act_type = 'Rehab'
            rehab_mpjpes.append(stats['mpjpe'])
        else:
            act_type = '?'
        print(f"{act_id:<6} {name:<22} {act_type:<8} {stats['mpjpe']:>8.2f} {stats['std']:>8.2f} {stats['count']:>7}")

    print('-'*70)
    if daily_mpjpes:
        print(f"{'Daily avg':<37} {np.mean(daily_mpjpes):>8.2f}")
    if rehab_mpjpes:
        print(f"{'Rehab avg':<37} {np.mean(rehab_mpjpes):>8.2f}")
    print(f"{'Overall avg':<37} {np.mean([s['mpjpe'] for s in results.values()]):>8.2f}")
    print('='*70)

def print_phase1_results(results: dict):
    """Pretty print Phase 1 evaluation results."""
    print(format_results_table(results, "CSI-StructPose V2 Phase 1"))
    print(f"\nRoot Error (Pelvis): {results['per_joint_mpjpe'][0]:.2f} mm")

def print_phase2_results(results: dict):
    """Pretty print Phase 2 evaluation results."""
    print(f"\n{'='*80}")
    print("CSI-StructPose V2 Phase 2 Evaluation Results (Absolute Pose)")
    print('='*80)
    print(f"Total samples: {results['num_samples']}")

    if 'root_error' in results:
        root = results['root_error']
        print("\n--- Root Position Error (mm) ---")
        print(f"  Mean:   {root['mean']:.2f}")
        print(f"  Std:    {root['std']:.2f}")
        print(f"  Median: {root['median']:.2f}")
        print(f"  Max:    {root['max']:.2f}")

    abs_r = results['absolute']
    print("\n--- Absolute Pose Metrics (World Coordinates) ---")
    print(f"  MPJPE:           {abs_r['mpjpe']:.2f} mm  <- For comparison with DT-Pose etc.")
    print(f"  MPJPE (aligned): {abs_r['mpjpe_aligned']:.2f} mm")
    print(f"  PA-MPJPE:        {abs_r['pa_mpjpe']:.2f} mm")
    print(f"  Root Error (Pelvis): {abs_r['per_joint_mpjpe'][0]:.2f} mm")

    pck = abs_r['pck']
    print("\n--- PCK (%) - Absolute Pose ---")
    pck_str = "  ".join([f"{k}: {v:.1f}" for k, v in pck.items()])
    print(f"  {pck_str}")

    rel_r = results['relative']
    print("\n--- Relative Pose Metrics (Root-Relative, for comparison with Phase 1) ---")
    print(f"  MPJPE:    {rel_r['mpjpe']:.2f} mm")
    print(f"  PA-MPJPE: {rel_r['pa_mpjpe']:.2f} mm")

    print("\n--- Per-Joint MPJPE (mm) - Absolute Pose ---")
    print("-" * 60)
    print(f"{'Joint':<15} {'MPJPE':>10} {'PA-MPJPE':>10} {'MPJPE-rel':>12}")
    print("-" * 60)

    for i, name in enumerate(JOINT_NAMES):
        mpjpe = abs_r['per_joint_mpjpe'][i]
        pa_mpjpe = abs_r['per_joint_pa_mpjpe'][i]
        mpjpe_rel = rel_r['per_joint_mpjpe'][i]
        print(f"{name:<15} {mpjpe:>10.2f} {pa_mpjpe:>10.2f} {mpjpe_rel:>12.2f}")

    print("-" * 60)

    print("\n--- Body Part MPJPE (mm) - Absolute ---")
    for part_name, error in abs_r['body_part_mpjpe'].items():
        print(f"  {part_name:<10}: {error:.2f}")

    print('='*80)

def build_model_phase1(cfg: dict) -> CSIStructPoseV2:
    """Build V2 Phase 1 model from config."""
    return build_csi_structpose_v2(cfg)

def build_model_phase2(cfg: dict) -> CSIStructPoseV2Phase2:
    """Build V2 Phase 2 model from config."""
    return build_csi_structpose_v2_phase2(cfg)

def main(args):
    if args.info_only:
        print_checkpoint_info(args.checkpoint)
        return

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    cfg = ckpt.get('config', ckpt.get('cfg', {}))
    if cfg:
        print(f"Config: {cfg}")
    else:
        print("WARNING: No config found in checkpoint, using defaults")

    state_dict = ckpt.get('model_state_dict', ckpt)

    phase2_indicators = ['holistic_encoder', 'holistic_point_generator', 'global_pose_decoder', 'pose_fusion']
    has_phase2_keys = any(
        any(indicator in k for indicator in phase2_indicators)
        for k in state_dict.keys()
    )

    phase2_keys_found = [k for k in state_dict.keys() if any(ind in k for ind in phase2_indicators)]
    if phase2_keys_found:
        print(f"Found Phase 2 keys: {phase2_keys_found[:3]}...")

    has_position_predictor = any('position_predictor' in k for k in state_dict.keys())

    if args.phase:
        phase = args.phase
    elif has_phase2_keys:
        phase = 2
    elif has_position_predictor:
        phase = 2
    else:
        phase = 1

    print(f"Detected Phase: {phase}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if phase == 1:
        model = build_model_phase1(cfg)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        model = build_model_phase2(cfg)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
            print(f"    First missing: {missing[:5]}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
            print(f"    First unexpected: {unexpected[:5]}")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    from datasets.mmfi_csi_mmwave import MMFiPreprocessedCSIOnly

    if args.absolute:
        root_relative = False
    else:
        root_relative = (phase == 1)

    if getattr(args, 'loeo_test_env', None):
        from datasets import MMFiLOEOCSIOnly
        print(f"[LOEO eval] test env = {args.loeo_test_env}; "
              f"per-env data root = {args.data_root}")
        test_dataset = MMFiLOEOCSIOnly(
            per_env_root=args.data_root,
            envs=[args.loeo_test_env],
            split='test',
            root_relative=root_relative,
        )
    else:
        test_dataset = MMFiPreprocessedCSIOnly(
            data_root=args.data_root,
            split='test',
            root_relative=root_relative
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Test samples: {len(test_dataset)}")
    coord_mode = "root-relative" if root_relative else "absolute"
    print(f"Coordinate mode: {coord_mode}")

    if phase == 1:
        results = evaluate_phase1(model, test_loader, device)
        print_phase1_results(results)

        if args.latex:
            print("\n" + "="*60)
            print("LaTeX Output:")
            print(format_latex_table(results, "CSI-StructPose V2"))
            print("="*60)

    else:
        results = evaluate_phase2(model, test_loader, device)
        print_phase2_results(results)

        if args.latex:
            print("\n" + "="*60)
            print("LaTeX Output (Absolute):")
            print(format_latex_table(results['absolute'], "CSI-StructPose V2"))
            print("="*60)

    if args.per_activity:
        sample_ids = None
        if getattr(test_dataset, 'sample_ids', None) is not None:
            sample_ids = np.asarray(test_dataset.sample_ids)
        else:
            sample_ids_path = os.path.join(args.data_root, 'sample_ids_test.npy')
            if os.path.exists(sample_ids_path):
                sample_ids = np.load(sample_ids_path, allow_pickle=True)
            else:
                print(f"WARNING: {sample_ids_path} not found and dataset has no "
                      f"in-memory sample_ids; cannot compute per-activity breakdown")

        if sample_ids is not None:
            if len(sample_ids) == len(test_dataset):
                act_results = evaluate_per_activity(
                    model, test_loader, device, sample_ids, phase=phase)
                print_per_activity_results(act_results)
                if isinstance(results, dict):
                    results['per_activity'] = act_results
            else:
                print(f"WARNING: sample_ids ({len(sample_ids)}) != dataset ({len(test_dataset)}), skipping per-activity")

    if 'test_metrics' in ckpt:
        print("\n--- Comparison with Saved Metrics ---")
        saved = ckpt['test_metrics']
        current = results if phase == 1 else results['absolute']

        for key in ['mpjpe', 'mpjpe_aligned', 'pa_mpjpe']:
            if key in saved and key in current:
                saved_val = saved[key]
                curr_val = current[key]
                diff = curr_val - saved_val
                print(f"  {key}: saved={saved_val:.2f}, eval={curr_val:.2f}, diff={diff:+.2f}")

    if args.output:
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_to_serializable(x) for x in obj]
            return obj

        save_results = convert_to_serializable(results)

        if args.output.endswith('.npy'):
            np.save(args.output, save_results, allow_pickle=True)
        else:
            import json
            with open(args.output, 'w') as f:
                json.dump(save_results, f, indent=2)

        print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CSI-StructPose V2')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data/cross_scene',
                        help='Path to preprocessed data directory')
    parser.add_argument('--phase', type=int, choices=[1, 2],
                        help='Force evaluation phase (auto-detected if not specified). '
                             '1=Root-relative, 2=Absolute with position prediction')
    parser.add_argument('--absolute', action='store_true',
                        help='Use absolute coordinates even for Phase 1 models '
                             '(for ablation: w/o Stage 2, w/o BP-LQs trained on absolute)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output', type=str,
                        help='Path to save results (.npy or .json)')
    parser.add_argument('--info_only', action='store_true',
                        help='Only print checkpoint info (no evaluation)')
    parser.add_argument('--latex', action='store_true',
                        help='Also output LaTeX table format')
    parser.add_argument('--loeo_test_env', type=str, default=None,
                        choices=[None, 'E01', 'E02', 'E03', 'E04'],
                        help='Leave-one-environment-out: env to evaluate as held-out test. '
                             'When set, --data_root must point to a per-env directory '
                             'produced by scripts/preprocess_per_env.py.')
    parser.add_argument('--per_activity', action='store_true',
                        help='Also report per-activity MPJPE breakdown')

    args = parser.parse_args()
    main(args)
