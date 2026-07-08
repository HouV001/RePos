#!/usr/bin/env python3
"""
Preprocess MM-Fi dataset with GraphPose-Fi compatible settings.

This script ensures CSI-StructPose uses IDENTICAL data processing as GraphPose-Fi:
1. CSI normalization: Min-max per frame (same as GraphPose-Fi)
2. Inf/NaN handling: Replace inf with nan, fill nan with column mean
3. Random split: Subject-level (cleaner than GraphPose-Fi's sample-level)
4. Pose coordinates: Root-relative by default (same as GraphPose-Fi)

Usage:
    python scripts/preprocess_graphposefi_style.py \
        --data_root /path/to/MMFi_Dataset \
        --output_dir ./data/graphposefi_style \
        --protocol protocol1 \
        --split random_split
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as scio
from tqdm import tqdm
from collections import defaultdict

PROTOCOL1_ACTIONS = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
PROTOCOL2_ACTIONS = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
ALL_ACTIONS = [f'A{i:02d}' for i in range(1, 28)]
ALL_SUBJECTS = [f'S{i:02d}' for i in range(1, 41)]

def get_scene(subject: str) -> str:
    """Get scene from subject ID (same as GraphPose-Fi)."""
    subject_num = int(subject[1:])
    if subject_num <= 10:
        return 'E01'
    elif subject_num <= 20:
        return 'E02'
    elif subject_num <= 30:
        return 'E03'
    else:
        return 'E04'

def load_csi_graphposefi_style(mat_path: str) -> np.ndarray:
    """
    Load and normalize CSI exactly like GraphPose-Fi.

    GraphPose-Fi code (feeder/mmfi.py lines 610-618):
        data = scio.loadmat(frame)['CSIamp']
        data[np.isinf(data)] = np.nan
        for i in range(10):
            temp_col = data[:, :, i]
            nan_num = np.count_nonzero(temp_col != temp_col)
            if nan_num != 0:
                temp_not_nan_col = temp_col[temp_col == temp_col]
                temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    Returns:
        csi: [3, 3, 114] normalized CSI (we convert to our format)
    """
    data = scio.loadmat(mat_path)
    csi = data.get('CSIamp', data.get('csi_amp', None))
    if csi is None:
        raise KeyError(f"Could not find CSI data in {mat_path}")

    csi = np.array(csi, dtype=np.float32)

    csi[np.isinf(csi)] = np.nan

    for i in range(csi.shape[2]):
        temp_col = csi[:, :, i]
        nan_mask = np.isnan(temp_col)
        if np.any(nan_mask):
            temp_not_nan = temp_col[~nan_mask]
            if len(temp_not_nan) > 0:
                temp_col[nan_mask] = temp_not_nan.mean()
            else:
                temp_col[nan_mask] = 0.0

    csi_min = np.min(csi)
    csi_max = np.max(csi)
    if csi_max - csi_min > 1e-8:
        csi = (csi - csi_min) / (csi_max - csi_min)
    else:
        csi = np.zeros_like(csi)

    csi = csi[:, :, :3]
    csi = csi.transpose(0, 2, 1)

    return csi

def get_split_form(split_type: str, actions: list, random_seed: int = 0, ratio: float = 0.8):
    """
    Get train/val form dicts (per-action split, matching DT-Pose/GraphPose-Fi EXACTLY).

    For random_split: Uses per-action seed incrementing (rs += 1), causing the same
    subject to potentially be in train for some actions and test for others.
    This matches DT-Pose/GraphPose-Fi for fair comparison.

    Returns:
        train_form: {subject: [actions]}
        val_form: {subject: [actions]}
    """
    train_form = {}
    val_form = {}

    if split_type == 'random_split':
        rs = random_seed
        for action in actions:
            np.random.seed(rs)
            idx = np.random.permutation(len(ALL_SUBJECTS))
            idx_train = idx[:int(np.floor(ratio * len(ALL_SUBJECTS)))]
            idx_val = idx[int(np.floor(ratio * len(ALL_SUBJECTS))):]
            subjects_train = [ALL_SUBJECTS[i] for i in idx_train]
            subjects_val = [ALL_SUBJECTS[i] for i in idx_val]

            for subject in ALL_SUBJECTS:
                if subject in subjects_train:
                    if subject in train_form:
                        train_form[subject].append(action)
                    else:
                        train_form[subject] = [action]
                if subject in subjects_val:
                    if subject in val_form:
                        val_form[subject].append(action)
                    else:
                        val_form[subject] = [action]
            rs += 1

    elif split_type == 'cross_scene_split':
        train_subjects = [f'S{i:02d}' for i in range(1, 31)]
        val_subjects = [f'S{i:02d}' for i in range(31, 41)]
        for subject in train_subjects:
            train_form[subject] = actions
        for subject in val_subjects:
            val_form[subject] = actions

    elif split_type == 'cross_subject_split':
        train_subjects = ['S01', 'S02', 'S03', 'S04', 'S06', 'S07', 'S08', 'S09',
                          'S11', 'S12', 'S13', 'S14', 'S16', 'S17', 'S18', 'S19',
                          'S21', 'S22', 'S23', 'S24', 'S26', 'S27', 'S28', 'S29',
                          'S31', 'S32', 'S33', 'S34', 'S36', 'S37', 'S38', 'S39']
        val_subjects = ['S05', 'S10', 'S15', 'S20', 'S25', 'S30', 'S35', 'S40']
        for subject in train_subjects:
            train_form[subject] = actions
        for subject in val_subjects:
            val_form[subject] = actions

    else:
        raise ValueError(f"Unknown split type: {split_type}")

    return train_form, val_form

def find_data_root(data_root: str) -> tuple:
    """Find actual data root and structure type."""
    if os.path.exists(os.path.join(data_root, 'E01')):
        return data_root, 'scene_based'

    if os.path.exists(os.path.join(data_root, 'S01')):
        return data_root, 'flat'

    raise ValueError(f"Could not find valid data structure in {data_root}")

def get_action_path(data_root: str, structure: str, subject: str, action: str) -> str:
    """Get path to action directory."""
    if structure == 'scene_based':
        scene = get_scene(subject)
        return os.path.join(data_root, scene, subject, action)
    else:
        return os.path.join(data_root, subject, action)

def process_dataset(data_root: str, output_dir: str, protocol: str, split_type: str,
                    num_frames: int = 5, root_relative: bool = True, random_seed: int = 0):
    """Process MM-Fi dataset with GraphPose-Fi compatible settings."""

    data_root, structure = find_data_root(data_root)
    print(f"Data root: {data_root}")
    print(f"Structure: {structure}")

    if protocol == 'protocol1':
        actions = PROTOCOL1_ACTIONS
    elif protocol == 'protocol2':
        actions = PROTOCOL2_ACTIONS
    elif protocol in ['protocol3', 'all']:
        actions = ALL_ACTIONS
    else:
        actions = ALL_ACTIONS

    print(f"Protocol: {protocol} ({len(actions)} actions)")

    train_form, val_form = get_split_form(split_type, actions, random_seed=random_seed)
    train_pairs = sum(len(acts) for acts in train_form.values())
    val_pairs = sum(len(acts) for acts in val_form.values())
    print(f"Split: {split_type} (per-action, matching DT-Pose/GraphPose-Fi)")
    print(f"  Train subject-action pairs: {train_pairs}")
    print(f"  Val subject-action pairs: {val_pairs}")
    if split_type == 'random_split':
        print(f"  Random seed (start): {random_seed}")
    print(f"Coordinates: {'root-relative' if root_relative else 'ABSOLUTE'}")

    os.makedirs(output_dir, exist_ok=True)

    for split_name, form in [('train', train_form), ('test', val_form)]:
        print(f"\nProcessing {split_name} split...")

        csi_list = []
        pose_list = []
        sample_id_list = []

        total_pairs = sum(len(acts) for acts in form.values())
        pbar = tqdm(total=total_pairs, desc=f"Processing ({split_name})")

        for subject, subject_actions in form.items():
            for action in subject_actions:
                action_path = get_action_path(data_root, structure, subject, action)

                gt_path = os.path.join(action_path, 'ground_truth.npy')
                csi_dir = os.path.join(action_path, 'wifi-csi')

                if not os.path.exists(gt_path) or not os.path.exists(csi_dir):
                    pbar.update(1)
                    continue

                poses = np.load(gt_path)
                n_frames = len(poses)

                sample_counter = 0

                for center_idx in range(num_frames - 1, n_frames):
                    csi_frames = []
                    valid = True

                    for i in range(num_frames):
                        frame_idx = center_idx - num_frames + 1 + i
                        frame_idx = max(0, frame_idx)
                        frame_path = os.path.join(csi_dir, f'frame{frame_idx + 1:03d}.mat')

                        if os.path.exists(frame_path):
                            try:
                                csi = load_csi_graphposefi_style(frame_path)
                                csi_frames.append(csi)
                            except Exception as e:
                                valid = False
                                break
                        else:
                            valid = False
                            break

                    if not valid or len(csi_frames) != num_frames:
                        continue

                    csi_window = np.stack(csi_frames, axis=0)

                    pose = poses[center_idx].astype(np.float32)

                    if root_relative:
                        pose = pose - pose[0:1, :]

                    csi_list.append(csi_window)
                    pose_list.append(pose)
                    sample_id_list.append(f"{subject}_{action}_{sample_counter:03d}")
                    sample_counter += 1

                pbar.update(1)

        pbar.close()

        if len(csi_list) == 0:
            print(f"  WARNING: No samples found for {split_name}!")
            continue

        csi_array = np.stack(csi_list, axis=0).astype(np.float32)
        pose_array = np.stack(pose_list, axis=0).astype(np.float32)

        print(f"  {split_name}: {len(csi_array)} samples")
        print(f"    CSI shape: {csi_array.shape}")
        print(f"    Pose shape: {pose_array.shape}")

        np.save(os.path.join(output_dir, f'csi_array_{split_name}.npy'), csi_array)
        np.save(os.path.join(output_dir, f'joint_gt_array_{split_name}.npy'), pose_array)
        np.save(os.path.join(output_dir, f'sample_ids_{split_name}.npy'), np.array(sample_id_list))
        print(f"    Sample IDs: {len(sample_id_list)}")

    metadata = {
        'protocol': protocol,
        'split': split_type,
        'split_level': 'per_action',
        'num_frames': num_frames,
        'root_relative': root_relative,
        'normalization': 'minmax_per_frame',
        'csi_format': '[T, RX, TX, Sub] = [5, 3, 3, 114]',
        'pose_format': '[J, 3] = [17, 3]',
        'train_form': train_form,
        'val_form': val_form,
        'random_seed': random_seed if split_type == 'random_split' else None,
    }
    np.save(os.path.join(output_dir, 'metadata.npy'), metadata)

    print(f"\nDone! Data saved to {output_dir}")
    print(f"Coordinate mode: {'root-relative' if root_relative else 'ABSOLUTE'}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess MM-Fi with GraphPose-Fi style')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to MM-Fi dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--protocol', type=str, default='protocol1',
                        choices=['protocol1', 'protocol2', 'protocol3', 'all'],
                        help='Action protocol (protocol3 = all 27 actions)')
    parser.add_argument('--split', type=str, default='random_split',
                        choices=['random_split', 'cross_scene_split', 'cross_subject_split'],
                        help='Data split type')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of frames per sample')
    parser.add_argument('--absolute', action='store_true',
                        help='Use absolute coordinates (default: root-relative)')
    parser.add_argument('--random_seed', type=int, default=0,
                        help='Random seed for random_split (default: 0, matching DT-Pose/GraphPose-Fi)')

    args = parser.parse_args()

    process_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        protocol=args.protocol,
        split_type=args.split,
        num_frames=args.num_frames,
        root_relative=not args.absolute,
        random_seed=args.random_seed,
    )

if __name__ == '__main__':
    main()
