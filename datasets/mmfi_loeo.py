"""
Leave-one-environment-out (LOEO) dataset for MM-Fi.

Reads per-environment npy files produced by `scripts/preprocess_per_env.py` and
assembles a {train_envs} -> {test_env} split on demand. This avoids the ~22 GB
of duplicated train data that would result from running the existing
`scripts/preprocess.py` separately for each LOEO target.

Drop-in API match for `MMFiPreprocessedCSIOnly`: returns (csi, pose, limb_len).

Usage in train_v2.py:
    from datasets import MMFiLOEOCSIOnly
    train = MMFiLOEOCSIOnly(per_env_root='data/per_env',
                            envs=['E02','E03','E04'],
                            split='train', root_relative=True)
    test  = MMFiLOEOCSIOnly(per_env_root='data/per_env',
                            envs=['E01'], split='test', root_relative=True)
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_LIMB_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

def _compute_limb_lengths(poses: np.ndarray) -> torch.Tensor:
    n = len(poses)
    lens = np.zeros((n, len(_LIMB_PAIRS)), dtype=np.float32)
    for i, (a, b) in enumerate(_LIMB_PAIRS):
        lens[:, i] = np.sqrt(((poses[:, a] - poses[:, b]) ** 2).sum(-1))
    return torch.from_numpy(lens)

def _load_env(per_env_root: str, env: str):
    csi = np.load(os.path.join(per_env_root, env, 'csi_array.npy')).astype(np.float32)
    pose = np.load(os.path.join(per_env_root, env, 'joint_gt_array.npy')).astype(np.float32)
    sid_path = os.path.join(per_env_root, env, 'sample_ids.npy')
    sids = np.load(sid_path, allow_pickle=True) if os.path.exists(sid_path) else None
    return csi, pose, sids

def _normalize_csi_inplace(csi: np.ndarray) -> np.ndarray:
    """Per-sample z-score + clip, matching MMFiPreprocessed."""
    flat = csi.reshape(len(csi), -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True) + 1e-8
    flat = (flat - mean) / std
    csi = flat.reshape(csi.shape)
    np.clip(csi, -10.0, 10.0, out=csi)
    return csi

class MMFiLOEOCSIOnly(Dataset):
    """CSI-only LOEO dataset assembled from per-env npy files.

    Returns (csi, pose, limb_len) — same signature as MMFiPreprocessedCSIOnly.

    Args:
        per_env_root: dir containing E01/, E02/, E03/, E04/ subdirs each with
            csi_array.npy / joint_gt_array.npy / sample_ids.npy
        envs: list of environments to include in this dataset (e.g. ['E02','E03','E04']
            for the train side of {test=E01}).
        split: 'train' or 'test' (only used for logging).
        root_relative: subtract pelvis to put it at origin (default True).
    """

    def __init__(self,
                 per_env_root: str,
                 envs: List[str],
                 split: str = 'train',
                 root_relative: bool = True,
                 subjects: List[str] = None):
        if not envs:
            raise ValueError("MMFiLOEOCSIOnly: `envs` must be non-empty")
        self.per_env_root = per_env_root
        self.envs = list(envs)
        self.split = split
        self.root_relative = root_relative
        self.subjects = list(subjects) if subjects is not None else None

        csis, poses, sids = [], [], []
        for env in self.envs:
            csi_e, pose_e, sid_e = _load_env(per_env_root, env)
            csis.append(csi_e)
            poses.append(pose_e)
            if sid_e is not None:
                sids.append(sid_e)
        csi = np.concatenate(csis, axis=0)
        pose = np.concatenate(poses, axis=0)
        sample_ids = np.concatenate(sids, axis=0) if sids else None

        if self.subjects is not None:
            if sample_ids is None:
                raise ValueError(
                    "subjects= filter requires per-env sample_ids.npy on disk")
            wanted = set(self.subjects)
            keep = np.array(
                [sid.split('_')[1] in wanted for sid in sample_ids])
            csi = csi[keep]
            pose = pose[keep]
            sample_ids = sample_ids[keep]

        if root_relative:
            pose = pose - pose[:, 0:1, :]

        csi = _normalize_csi_inplace(csi)

        self.csi = torch.from_numpy(csi)
        self.poses = torch.from_numpy(pose)
        self.limb_lengths = _compute_limb_lengths(pose)
        self.sample_ids = sample_ids

        sub_str = f" subjects={self.subjects}" if self.subjects else ""
        print(f"[MMFiLOEOCSIOnly] split={split} envs={self.envs}{sub_str} "
              f"-> {len(self.csi)} samples; CSI {tuple(self.csi.shape)} "
              f"Pose {tuple(self.poses.shape)} root_relative={root_relative}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.csi[idx], self.poses[idx], self.limb_lengths[idx]

    def __len__(self) -> int:
        return len(self.csi)

def make_loeo_split(per_env_root: str, test_env: str,
                    all_envs: List[str] = ('E01', 'E02', 'E03', 'E04'),
                    root_relative: bool = True):
    """Build (train_dataset, test_dataset) for one LOEO target."""
    test_env = test_env.upper()
    if test_env not in all_envs:
        raise ValueError(f"test_env {test_env!r} not in {all_envs}")
    train_envs = [e for e in all_envs if e != test_env]
    train = MMFiLOEOCSIOnly(per_env_root, train_envs, split='train', root_relative=root_relative)
    test = MMFiLOEOCSIOnly(per_env_root, [test_env], split='test', root_relative=root_relative)
    return train, test
