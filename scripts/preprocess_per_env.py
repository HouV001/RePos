#!/usr/bin/env python3
"""
Per-environment MM-Fi preprocessing for leave-one-environment-out (LOEO) experiments.

Writes one set of .npy files per environment:

    output_dir/
        E01/csi_array.npy, joint_gt_array.npy, sample_ids.npy
        E02/...
        E03/...
        E04/...
        metadata.json

This is consumed by `datasets.mmfi_loeo.MMFiLOEODataset`, which assembles a
{train_envs} -> {test_env} split on demand. Compared with running the existing
`preprocess.py` once per LOEO split (4 outputs, each containing the same train
data three times), this saves ~22 GB and keeps cache locality.

Compatible with the same CSI normalization used by `scripts/preprocess.py`
(min-max per frame, GraphPose-Fi style).

Dataset layout expected:

    data_root/
        E01/S01/A01/wifi-csi/frame001.mat ... ground_truth.npy
        E01/S02/...
        ...
        E04/S31/...

Each .mat has CSIamp [3 RX, 114 sub, 10 packets]; we take the first 3 packets
to match the [T, 3, 3, 114] format the model expects.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import scipy.io as scio
from tqdm import tqdm

ALL_ACTIONS = [f"A{i:02d}" for i in range(1, 28)]
PROTOCOL1_ACTIONS = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18',
                     'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
PROTOCOL2_ACTIONS = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
                     'A15', 'A16', 'A24', 'A25', 'A26']

def load_csi_graphposefi_style(mat_path: str) -> np.ndarray:
    """Same per-frame load + min-max normalize as scripts/preprocess.py."""
    data = scio.loadmat(mat_path)
    csi = data.get('CSIamp', data.get('csi_amp', None))
    if csi is None:
        raise KeyError(f"Could not find CSI data in {mat_path}")

    csi = np.array(csi, dtype=np.float32)
    csi[np.isinf(csi)] = np.nan
    for i in range(csi.shape[2]):
        col = csi[:, :, i]
        nan_mask = np.isnan(col)
        if np.any(nan_mask):
            valid = col[~nan_mask]
            col[nan_mask] = valid.mean() if len(valid) else 0.0

    cmin, cmax = float(np.min(csi)), float(np.max(csi))
    if cmax - cmin > 1e-8:
        csi = (csi - cmin) / (cmax - cmin)
    else:
        csi = np.zeros_like(csi)

    csi = csi[:, :, :3].transpose(0, 2, 1)
    return csi

def list_subjects(env_dir: str) -> List[str]:
    """Return sorted SXX subject dirs in an env."""
    return sorted([d for d in os.listdir(env_dir)
                   if d.startswith('S') and os.path.isdir(os.path.join(env_dir, d))])

def process_env(data_root: str, env: str, output_dir: str, actions: List[str],
                num_frames: int = 5) -> Dict:
    """Process one environment to npy. Returns small stats dict."""
    env_dir = os.path.join(data_root, env)
    if not os.path.isdir(env_dir):
        raise FileNotFoundError(f"Missing env dir: {env_dir}")

    subjects = list_subjects(env_dir)
    out_dir = os.path.join(output_dir, env)
    os.makedirs(out_dir, exist_ok=True)

    csi_list, pose_list, sid_list = [], [], []
    skipped = 0

    pbar = tqdm(total=len(subjects) * len(actions), desc=f"[{env}]", leave=False)
    for subject in subjects:
        for action in actions:
            action_path = os.path.join(env_dir, subject, action)
            gt_path = os.path.join(action_path, 'ground_truth.npy')
            csi_dir = os.path.join(action_path, 'wifi-csi')
            if not (os.path.exists(gt_path) and os.path.isdir(csi_dir)):
                pbar.update(1); skipped += 1; continue

            poses = np.load(gt_path)
            n = len(poses)

            sample_counter = 0
            for center in range(num_frames - 1, n):
                frames, ok = [], True
                for k in range(num_frames):
                    fi = max(0, center - num_frames + 1 + k)
                    fp = os.path.join(csi_dir, f"frame{fi+1:03d}.mat")
                    if not os.path.exists(fp):
                        ok = False; break
                    try:
                        frames.append(load_csi_graphposefi_style(fp))
                    except Exception:
                        ok = False; break
                if not ok or len(frames) != num_frames:
                    continue
                csi_list.append(np.stack(frames, axis=0))
                pose_list.append(poses[center].astype(np.float32))
                sid_list.append(f"{env}_{subject}_{action}_{sample_counter:03d}")
                sample_counter += 1
            pbar.update(1)
    pbar.close()

    if not csi_list:
        return {'env': env, 'n_samples': 0, 'skipped_pairs': skipped}

    csi_arr = np.stack(csi_list, axis=0).astype(np.float32)
    pose_arr = np.stack(pose_list, axis=0).astype(np.float32)
    sid_arr = np.asarray(sid_list)

    np.save(os.path.join(out_dir, 'csi_array.npy'), csi_arr)
    np.save(os.path.join(out_dir, 'joint_gt_array.npy'), pose_arr)
    np.save(os.path.join(out_dir, 'sample_ids.npy'), sid_arr)

    return {
        'env': env,
        'n_samples': int(csi_arr.shape[0]),
        'csi_shape': list(csi_arr.shape),
        'pose_shape': list(pose_arr.shape),
        'subjects': subjects,
        'actions': actions,
        'skipped_pairs': skipped,
    }

def main():
    ap = argparse.ArgumentParser(description='Per-environment MM-Fi preprocessing for LOEO')
    ap.add_argument('--data_root', required=True, help='Path to MMFi_Dataset')
    ap.add_argument('--output_dir', required=True, help='Output dir (will create per-env subdirs)')
    ap.add_argument('--protocol', default='protocol3',
                    choices=['protocol1', 'protocol2', 'protocol3', 'all'])
    ap.add_argument('--num_frames', type=int, default=5)
    ap.add_argument('--envs', nargs='+', default=['E01', 'E02', 'E03', 'E04'])
    ap.add_argument('--limit_subjects', type=int, default=None,
                    help='For smoke testing: process at most N subjects per env.')
    ap.add_argument('--limit_actions', type=int, default=None,
                    help='For smoke testing: process at most N actions per subject.')
    args = ap.parse_args()

    actions = {
        'protocol1': PROTOCOL1_ACTIONS,
        'protocol2': PROTOCOL2_ACTIONS,
        'protocol3': ALL_ACTIONS,
        'all': ALL_ACTIONS,
    }[args.protocol]
    if args.limit_actions is not None:
        actions = actions[:args.limit_actions]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Data root : {args.data_root}")
    print(f"Output    : {args.output_dir}")
    print(f"Protocol  : {args.protocol} ({len(actions)} actions)")
    print(f"Envs      : {args.envs}")
    print(f"Frames/win: {args.num_frames}")
    if args.limit_subjects or args.limit_actions:
        print(f"!! SMOKE-TEST MODE: limit_subjects={args.limit_subjects} "
              f"limit_actions={args.limit_actions}")

    all_stats = {}
    for env in args.envs:
        if args.limit_subjects is not None:
            orig = list_subjects(os.path.join(args.data_root, env))
            sel = orig[:args.limit_subjects]
            print(f"  [{env}] subjects limited to: {sel}")
            stats = _process_env_with_subjects(
                args.data_root, env, args.output_dir, actions, args.num_frames, sel)
        else:
            stats = process_env(args.data_root, env, args.output_dir, actions, args.num_frames)
        all_stats[env] = stats
        print(f"  [{env}] -> {stats.get('n_samples', 0)} samples "
              f"(skipped {stats.get('skipped_pairs', 0)} pairs)")

    metadata = {
        'protocol': args.protocol,
        'num_frames': args.num_frames,
        'csi_format': '[T, RX, TX, Sub] = [5, 3, 3, 114]',
        'pose_format': '[J, 3] = [17, 3]',
        'normalization': 'minmax_per_frame',
        'env_stats': all_stats,
        'sample_id_format': 'EXX_SXX_AXX_NNN',
    }
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nWrote metadata to {os.path.join(args.output_dir, 'metadata.json')}")

def _process_env_with_subjects(data_root, env, output_dir, actions, num_frames, subjects):
    """Variant of process_env that accepts an explicit subject list (smoke-test)."""
    env_dir = os.path.join(data_root, env)
    out_dir = os.path.join(output_dir, env)
    os.makedirs(out_dir, exist_ok=True)

    csi_list, pose_list, sid_list = [], [], []
    skipped = 0
    pbar = tqdm(total=len(subjects) * len(actions), desc=f"[{env}/limited]", leave=False)
    for subject in subjects:
        for action in actions:
            action_path = os.path.join(env_dir, subject, action)
            gt_path = os.path.join(action_path, 'ground_truth.npy')
            csi_dir = os.path.join(action_path, 'wifi-csi')
            if not (os.path.exists(gt_path) and os.path.isdir(csi_dir)):
                pbar.update(1); skipped += 1; continue
            poses = np.load(gt_path)
            n = len(poses)
            sample_counter = 0
            for center in range(num_frames - 1, n):
                frames, ok = [], True
                for k in range(num_frames):
                    fi = max(0, center - num_frames + 1 + k)
                    fp = os.path.join(csi_dir, f"frame{fi+1:03d}.mat")
                    if not os.path.exists(fp):
                        ok = False; break
                    try:
                        frames.append(load_csi_graphposefi_style(fp))
                    except Exception:
                        ok = False; break
                if not ok or len(frames) != num_frames:
                    continue
                csi_list.append(np.stack(frames, axis=0))
                pose_list.append(poses[center].astype(np.float32))
                sid_list.append(f"{env}_{subject}_{action}_{sample_counter:03d}")
                sample_counter += 1
            pbar.update(1)
    pbar.close()

    if not csi_list:
        return {'env': env, 'n_samples': 0, 'skipped_pairs': skipped}

    csi_arr = np.stack(csi_list, axis=0).astype(np.float32)
    pose_arr = np.stack(pose_list, axis=0).astype(np.float32)
    sid_arr = np.asarray(sid_list)

    np.save(os.path.join(out_dir, 'csi_array.npy'), csi_arr)
    np.save(os.path.join(out_dir, 'joint_gt_array.npy'), pose_arr)
    np.save(os.path.join(out_dir, 'sample_ids.npy'), sid_arr)
    return {
        'env': env,
        'n_samples': int(csi_arr.shape[0]),
        'csi_shape': list(csi_arr.shape),
        'pose_shape': list(pose_arr.shape),
        'subjects': subjects,
        'actions': actions,
        'skipped_pairs': skipped,
    }

if __name__ == '__main__':
    main()
