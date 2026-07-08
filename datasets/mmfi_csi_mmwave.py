"""
MM-Fi Dataset with paired CSI and mmWave for Phase 1 Training

Supports three split protocols:
1. cross_scene: Train on E01+E02+E03, Test on E04 (most strict, tests environment generalization)
2. cross_subject: Train on 32 subjects, Test on 8 subjects (tests person generalization)
3. random_split: 80/20 random split (has data leakage risk)

Data structure:
    data_root/
        E01/S01/A01/
            wifi-csi/frame001.mat, frame002.mat, ...
            mmwave/frame001.bin, frame002.bin, ...
            ground_truth.npy
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import scipy.io as scio
from glob import glob

SCENE_SUBJECT_MAP = {
    'E01': [f'S{i:02d}' for i in range(1, 11)],
    'E02': [f'S{i:02d}' for i in range(11, 21)],
    'E03': [f'S{i:02d}' for i in range(21, 31)],
    'E04': [f'S{i:02d}' for i in range(31, 41)],
}

ALL_SUBJECTS = [f'S{i:02d}' for i in range(1, 41)]
ALL_ACTIONS = [f'A{i:02d}' for i in range(1, 28)]

SPLIT_PROTOCOLS = {
    'cross_subject': {
        'train': [s for s in ALL_SUBJECTS if s not in
                  ['S05', 'S10', 'S15', 'S20', 'S25', 'S30', 'S35', 'S40']],
        'test': ['S05', 'S10', 'S15', 'S20', 'S25', 'S30', 'S35', 'S40']
    },
    'cross_scene': {
        'train_scenes': ['E01', 'E02', 'E03'],
        'test_scenes': ['E04']
    },
    'random_split': {
        'ratio': 0.8,
        'seed': 42
    }
}

def sanitize_data(data: np.ndarray) -> np.ndarray:
    """Sanitize data by replacing inf/NaN values."""
    finite_mask = np.isfinite(data)
    if finite_mask.any():
        min_val = np.min(data[finite_mask])
        max_val = np.max(data[finite_mask])
    else:
        min_val, max_val = 0.0, 0.0

    data = np.where(np.isneginf(data), min_val, data)
    data = np.where(np.isposinf(data), max_val, data)
    data = np.nan_to_num(data, nan=0.0)
    return data

class MMFiCSImmWavePaired(Dataset):
    """
    MM-Fi Dataset for Phase 1 training with mmWave supervision.

    Supports two data structures:
    - scene_based: E01/S01/A01/ (official MM-Fi structure with scene hierarchy)
    - flat: S01/A01/ (flattened structure with global subject IDs S01-S40)

    Supports three split protocols:
    - cross_scene: Train on E01+E02+E03 (or S01-S30), Test on E04 (or S31-S40)
    - cross_subject: Train on 32 subjects, Test on 8 subjects
    - random_split: 80/20 random split (has data leakage risk)

    Returns: (csi, mmwave, pose, limb_len) tuple

    Args:
        data_root: Root directory of MM-Fi dataset
        split: 'train' or 'test'
        protocol: Split protocol ('cross_scene', 'cross_subject', 'random_split')
        num_frames: Number of temporal frames (default: 5)
        max_points: Max points per frame (default: 150)
        random_seed: Random seed for random_split protocol
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 protocol: str = 'cross_scene',
                 num_frames: int = 5,
                 max_points: int = 150,
                 random_seed: int = 42,
                 root_relative: bool = True):

        self.data_root = self._find_data_root(data_root)
        self.split = split
        self.protocol = protocol
        self.num_frames = num_frames
        self.max_points = max_points
        self.random_seed = random_seed
        self.root_relative = root_relative

        self.data_structure = self._detect_data_structure()

        self.samples = self._build_sample_list()
        coord_mode = "root-relative" if root_relative else "absolute"
        print(f"MMFiCSImmWavePaired ({protocol}, {self.data_structure}, {coord_mode}): {len(self.samples)} samples for {split}")

    def _find_data_root(self, data_root: str) -> str:
        """Auto-detect data root directory."""
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data root not found: {data_root}")

        first_level = os.listdir(data_root)
        has_scenes = any(d.startswith('E') and os.path.isdir(os.path.join(data_root, d)) for d in first_level)
        has_subjects = any(d.startswith('S') and os.path.isdir(os.path.join(data_root, d)) for d in first_level)

        if has_scenes or has_subjects:
            return data_root

        for subdir in ['MMFi_Dataset', 'Zipfiles', 'filtered_mmwave', 'data']:
            subpath = os.path.join(data_root, subdir)
            if os.path.isdir(subpath):
                try:
                    sub_contents = os.listdir(subpath)
                    if any(d.startswith('E') or d.startswith('S') for d in sub_contents):
                        print(f"  Found data in: {subdir}")
                        return subpath
                except PermissionError:
                    continue

        return data_root

    def _detect_data_structure(self) -> str:
        """
        Auto-detect data structure:
        - 'scene_based': E01/S01/A01/ (official MM-Fi with scene hierarchy)
        - 'flat': S01/A01/ (global subject IDs S01-S40)
        """
        try:
            first_level = os.listdir(self.data_root)
        except Exception as e:
            raise ValueError(f"Cannot read data root: {self.data_root}") from e

        has_scenes = any(d.startswith('E') and len(d) >= 2 and d[1:3].isdigit()
                        for d in first_level if os.path.isdir(os.path.join(self.data_root, d)))
        has_subjects = any(d.startswith('S') and len(d) >= 2 and d[1:3].isdigit()
                         for d in first_level if os.path.isdir(os.path.join(self.data_root, d)))

        if has_scenes:
            print(f"  Detected scene-based structure (E01/S01/A01)")
            return 'scene_based'
        elif has_subjects:
            print(f"  Detected flat structure (S01/A01)")
            return 'flat'
        else:
            raise ValueError(f"Cannot detect data structure in {self.data_root}. "
                           f"Expected E01/... or S01/... directories.")

    def _build_sample_list(self) -> List[dict]:
        """Build list of samples based on protocol and data structure."""
        if self.data_structure == 'flat':
            if self.protocol == 'cross_scene':
                return self._build_flat_cross_scene_samples()
            elif self.protocol == 'cross_subject':
                return self._build_flat_cross_subject_samples()
            elif self.protocol == 'random_split':
                return self._build_flat_random_split_samples()
        else:
            if self.protocol == 'cross_scene':
                return self._build_scene_cross_scene_samples()
            elif self.protocol == 'cross_subject':
                return self._build_scene_cross_subject_samples()
            elif self.protocol == 'random_split':
                return self._build_scene_random_split_samples()

        raise ValueError(f"Unknown protocol: {self.protocol}. "
                        f"Choose from: cross_scene, cross_subject, random_split")

    def _build_flat_cross_scene_samples(self) -> List[dict]:
        """
        Cross-Scene for flat structure (S01-S40 directly).
        Train: S01-S30 (virtual E01+E02+E03), Test: S31-S40 (virtual E04)
        """
        samples = []

        if self.split == 'train':
            target_subjects = [f'S{i:02d}' for i in range(1, 31)]
        else:
            target_subjects = [f'S{i:02d}' for i in range(31, 41)]

        for subject in sorted(os.listdir(self.data_root)):
            subject_path = os.path.join(self.data_root, subject)
            if not os.path.isdir(subject_path) or not subject.startswith('S'):
                continue
            if subject not in target_subjects:
                continue

            subject_num = int(subject[1:])
            virtual_scene = f'E{(subject_num - 1) // 10 + 1:02d}'

            for action in sorted(os.listdir(subject_path)):
                action_path = os.path.join(subject_path, action)
                if os.path.isdir(action_path) and action.startswith('A'):
                    self._add_action_samples(samples, action_path, virtual_scene, subject)

        return samples

    def _build_flat_cross_subject_samples(self) -> List[dict]:
        """
        Cross-Subject for flat structure.
        Train: 32 subjects, Test: S05,S10,S15,S20,S25,S30,S35,S40
        """
        samples = []
        protocol = SPLIT_PROTOCOLS['cross_subject']

        if self.split == 'train':
            target_subjects = protocol['train']
        else:
            target_subjects = protocol['test']

        for subject in sorted(os.listdir(self.data_root)):
            subject_path = os.path.join(self.data_root, subject)
            if not os.path.isdir(subject_path) or not subject.startswith('S'):
                continue
            if subject not in target_subjects:
                continue

            subject_num = int(subject[1:])
            virtual_scene = f'E{(subject_num - 1) // 10 + 1:02d}'

            for action in sorted(os.listdir(subject_path)):
                action_path = os.path.join(subject_path, action)
                if os.path.isdir(action_path) and action.startswith('A'):
                    self._add_action_samples(samples, action_path, virtual_scene, subject)

        return samples

    def _build_flat_random_split_samples(self) -> List[dict]:
        """
        Random split for flat structure.
        ⚠️ WARNING: Same subject's data may appear in both train and test!
        """
        all_samples = []

        for subject in sorted(os.listdir(self.data_root)):
            subject_path = os.path.join(self.data_root, subject)
            if not os.path.isdir(subject_path) or not subject.startswith('S'):
                continue

            subject_num = int(subject[1:])
            virtual_scene = f'E{(subject_num - 1) // 10 + 1:02d}'

            for action in sorted(os.listdir(subject_path)):
                action_path = os.path.join(subject_path, action)
                if os.path.isdir(action_path) and action.startswith('A'):
                    self._add_action_samples(all_samples, action_path, virtual_scene, subject)

        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(all_samples))
        split_idx = int(len(all_samples) * SPLIT_PROTOCOLS['random_split']['ratio'])

        if self.split == 'train':
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        return [all_samples[i] for i in selected_indices]

    def _build_scene_cross_scene_samples(self) -> List[dict]:
        """
        Cross-Scene split for scene-based structure.
        Train on E01+E02+E03, Test on E04.
        """
        samples = []
        protocol = SPLIT_PROTOCOLS['cross_scene']

        if self.split == 'train':
            target_scenes = protocol['train_scenes']
        else:
            target_scenes = protocol['test_scenes']

        for scene in sorted(os.listdir(self.data_root)):
            scene_path = os.path.join(self.data_root, scene)
            if not os.path.isdir(scene_path) or not scene.startswith('E'):
                continue
            if scene not in target_scenes:
                continue

            for subject in sorted(os.listdir(scene_path)):
                subject_path = os.path.join(scene_path, subject)
                if not os.path.isdir(subject_path) or not subject.startswith('S'):
                    continue

                scene_num = int(scene[1:])
                local_num = int(subject[1:])
                global_subject = f'S{(scene_num - 1) * 10 + local_num:02d}'

                for action in sorted(os.listdir(subject_path)):
                    action_path = os.path.join(subject_path, action)
                    if os.path.isdir(action_path) and action.startswith('A'):
                        self._add_action_samples(samples, action_path, scene, global_subject)

        return samples

    def _build_scene_cross_subject_samples(self) -> List[dict]:
        """
        Cross-Subject split for scene-based structure.
        Train on 32 subjects, Test on 8 subjects (global IDs).
        """
        samples = []
        protocol = SPLIT_PROTOCOLS['cross_subject']

        if self.split == 'train':
            target_subjects = protocol['train']
        else:
            target_subjects = protocol['test']

        for scene in sorted(os.listdir(self.data_root)):
            scene_path = os.path.join(self.data_root, scene)
            if not os.path.isdir(scene_path) or not scene.startswith('E'):
                continue

            scene_num = int(scene[1:])

            for subject in sorted(os.listdir(scene_path)):
                subject_path = os.path.join(scene_path, subject)
                if not os.path.isdir(subject_path) or not subject.startswith('S'):
                    continue

                local_num = int(subject[1:])
                global_subject = f'S{(scene_num - 1) * 10 + local_num:02d}'

                if global_subject not in target_subjects:
                    continue

                for action in sorted(os.listdir(subject_path)):
                    action_path = os.path.join(subject_path, action)
                    if os.path.isdir(action_path) and action.startswith('A'):
                        self._add_action_samples(samples, action_path, scene, global_subject)

        return samples

    def _build_scene_random_split_samples(self) -> List[dict]:
        """
        Random split for scene-based structure.
        ⚠️ WARNING: Same subject's data may appear in both train and test!
        """
        all_samples = []

        for scene in sorted(os.listdir(self.data_root)):
            scene_path = os.path.join(self.data_root, scene)
            if not os.path.isdir(scene_path) or not scene.startswith('E'):
                continue

            scene_num = int(scene[1:])

            for subject in sorted(os.listdir(scene_path)):
                subject_path = os.path.join(scene_path, subject)
                if not os.path.isdir(subject_path) or not subject.startswith('S'):
                    continue

                local_num = int(subject[1:])
                global_subject = f'S{(scene_num - 1) * 10 + local_num:02d}'

                for action in sorted(os.listdir(subject_path)):
                    action_path = os.path.join(subject_path, action)
                    if os.path.isdir(action_path) and action.startswith('A'):
                        self._add_action_samples(all_samples, action_path, scene, global_subject)

        np.random.seed(self.random_seed)
        indices = np.random.permutation(len(all_samples))
        split_idx = int(len(all_samples) * SPLIT_PROTOCOLS['random_split']['ratio'])

        if self.split == 'train':
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]

        return [all_samples[i] for i in selected_indices]

    def _add_action_samples(self, samples: List, action_path: str,
                           scene: str = None, subject: str = None):
        """Add samples from action directory."""
        if not os.path.isdir(action_path):
            return

        csi_dir = os.path.join(action_path, 'wifi-csi')
        mmwave_dir = os.path.join(action_path, 'mmwave')
        gt_path = os.path.join(action_path, 'ground_truth.npy')

        if not all(os.path.exists(p) for p in [csi_dir, mmwave_dir, gt_path]):
            return

        csi_files = sorted(glob(os.path.join(csi_dir, 'frame*.mat')))
        mmwave_files = sorted(glob(os.path.join(mmwave_dir, 'frame*.bin')))
        gt = np.load(gt_path)

        n_frames = min(len(csi_files), len(mmwave_files), len(gt))

        if n_frames < self.num_frames:
            return

        for idx in range(self.num_frames - 1, n_frames):
            samples.append({
                'frame_idx': idx,
                'action_path': action_path,
                'scene': scene,
                'subject': subject,
            })

    def _load_csi_frames(self, action_path: str, center_idx: int) -> np.ndarray:
        """Load CSI frames [T, 3, 3, 114]."""
        csi_dir = os.path.join(action_path, 'wifi-csi')
        frames = []

        for i in range(self.num_frames):
            frame_0idx = center_idx - self.num_frames + 1 + i
            frame_0idx = max(0, frame_0idx)
            frame_1idx = frame_0idx + 1

            path = os.path.join(csi_dir, f'frame{frame_1idx:03d}.mat')

            loaded = False
            if os.path.exists(path):
                try:
                    data = scio.loadmat(path)
                    csi_amp = data.get('CSIamp', data.get('csi_amp', None))
                    if csi_amp is not None:
                        csi_amp = np.array(csi_amp, dtype=np.float32)

                        if csi_amp.ndim == 3 and csi_amp.shape == (3, 114, 10):
                            csi_amp = csi_amp[:, :, :3].transpose(0, 2, 1)

                        csi_amp = sanitize_data(csi_amp)
                        frames.append(csi_amp)
                        loaded = True
                except:
                    pass

            if not loaded:
                frames.append(np.zeros((3, 3, 114), dtype=np.float32))

        csi = np.stack(frames, axis=0)

        mean = csi.mean()
        std = csi.std() + 1e-8
        csi = (csi - mean) / std
        csi = np.clip(csi, -10.0, 10.0)

        return csi

    def _load_mmwave_frames(self, action_path: str, center_idx: int) -> np.ndarray:
        """Load mmWave frames [T, max_points, 5]."""
        mmwave_dir = os.path.join(action_path, 'mmwave')
        frames = []

        for i in range(self.num_frames):
            frame_0idx = center_idx - self.num_frames + 1 + i
            frame_0idx = max(0, frame_0idx)
            frame_1idx = frame_0idx + 1

            path = os.path.join(mmwave_dir, f'frame{frame_1idx:03d}.bin')

            loaded = False
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        data = np.frombuffer(f.read(), dtype=np.float64)
                    pc = data.reshape(-1, 5)
                    pc = self._pad_or_truncate(pc, self.max_points)
                    frames.append(pc.astype(np.float32))
                    loaded = True
                except:
                    pass

            if not loaded:
                frames.append(np.zeros((self.max_points, 5), dtype=np.float32))

        return np.stack(frames, axis=0)

    def _pad_or_truncate(self, pc: np.ndarray, target_n: int) -> np.ndarray:
        """Pad or truncate point cloud."""
        n = len(pc)
        if n >= target_n:
            return pc[:target_n]
        else:
            padding = np.zeros((target_n - n, pc.shape[1]))
            return np.concatenate([pc, padding], axis=0)

    def _compute_limb_lengths(self, pose: np.ndarray) -> np.ndarray:
        """Compute limb lengths from pose."""
        limb_pairs = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
            (0, 7), (7, 8), (8, 9), (9, 10),
            (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)
        ]
        limb_lengths = []
        for src, dst in limb_pairs:
            length = np.sqrt(((pose[src] - pose[dst]) ** 2).sum())
            limb_lengths.append(length)
        return np.array(limb_lengths, dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns tuple: (csi, mmwave, pose, limb_len)

        Matches CSIDiff's format exactly.
        """
        sample = self.samples[idx]
        action_path = sample['action_path']
        frame_idx = sample['frame_idx']

        csi = self._load_csi_frames(action_path, frame_idx)

        mmwave = self._load_mmwave_frames(action_path, frame_idx)

        gt_path = os.path.join(action_path, 'ground_truth.npy')
        gt = np.load(gt_path)
        pose = gt[frame_idx]

        if self.root_relative:
            pose = pose - pose[0:1, :]

        limb_len = self._compute_limb_lengths(pose)

        csi = torch.from_numpy(csi.copy()).float()
        mmwave = torch.from_numpy(mmwave.copy()).float()
        pose = torch.from_numpy(pose.copy()).float()
        limb_len = torch.from_numpy(limb_len.copy()).float()

        return csi, mmwave, pose, limb_len

    def __len__(self) -> int:
        return len(self.samples)

class MMFiCSIOnly(Dataset):
    """
    MM-Fi Dataset for CSI-only training (Phase 2 - no mmWave needed).

    Returns tuple: (csi, pose, limb_len)
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 protocol: str = 'cross_scene',
                 num_frames: int = 5,
                 random_seed: int = 42):

        self._base = MMFiCSImmWavePaired(
            data_root=data_root,
            split=split,
            protocol=protocol,
            num_frames=num_frames,
            random_seed=random_seed
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (csi, pose, limb_len)"""
        csi, _, pose, limb_len = self._base[idx]
        return csi, pose, limb_len

    def __len__(self) -> int:
        return len(self._base)

class MMFiPreprocessed(Dataset):
    """
    Fast dataset loading from preprocessed .npy files.

    Use this with data from scripts/preprocess.py for much faster training.

    Expected files in data_root:
        csi_array_train.npy / csi_array_test.npy     [N, T, 3, 3, 114]
        joint_gt_array_train.npy / joint_gt_array_test.npy  [N, 17, 3]
        mmwave_array_train.npy / mmwave_array_test.npy (optional)  [N, T, 150, 5]

    Returns: (csi, mmwave, pose, limb_len) tuple for Phase 1
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 with_mmwave: bool = True,
                 root_relative: bool = True):
        """
        Args:
            data_root: Directory containing preprocessed .npy files
            split: 'train' or 'test'
            with_mmwave: Whether to load mmWave data (for Phase 1)
            root_relative: If True, use root-relative coordinates (pelvis at origin).
                          If False, use absolute coordinates.
                          Default: True (recommended for pose estimation).
        """
        self.data_root = data_root
        self.split = split
        self.with_mmwave = with_mmwave

        csi_path = os.path.join(data_root, f'csi_array_{split}.npy')
        if not os.path.exists(csi_path):
            raise FileNotFoundError(f"CSI file not found: {csi_path}")

        print(f"Loading {csi_path}...")
        csi = np.load(csi_path).astype(np.float32)

        pose_path = os.path.join(data_root, f'joint_gt_array_{split}.npy')
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

        print(f"Loading {pose_path}...")
        poses = np.load(pose_path).astype(np.float32)

        self.root_relative = root_relative
        if root_relative:
            print("  Using root-relative coordinates (pelvis at origin)")
            poses = poses - poses[:, 0:1, :]
        else:
            print("  Using absolute coordinates")

        print("Normalizing CSI...")
        csi_flat = csi.reshape(len(csi), -1)
        mean = csi_flat.mean(axis=1, keepdims=True)
        std = csi_flat.std(axis=1, keepdims=True) + 1e-8
        csi_flat = (csi_flat - mean) / std
        csi = csi_flat.reshape(csi.shape)
        csi = np.clip(csi, -10.0, 10.0)

        print("Converting to Tensor...")
        self.csi = torch.from_numpy(csi)
        self.poses = torch.from_numpy(poses)

        self.limb_lengths = self._compute_all_limb_lengths(poses)

        self.mmwave = None
        if with_mmwave:
            mmwave_path = os.path.join(data_root, f'mmwave_array_{split}.npy')
            if os.path.exists(mmwave_path):
                print(f"Loading {mmwave_path} (full load to RAM)...")
                mmwave = np.load(mmwave_path).astype(np.float32)
                self.mmwave = torch.from_numpy(mmwave)
                del mmwave
            else:
                print(f"Warning: mmWave file not found: {mmwave_path}")
                self.with_mmwave = False

        print(f"MMFiPreprocessed ({split}): {len(self)} samples")
        print(f"  CSI: {self.csi.shape}, Pose: {self.poses.shape}")
        if self.mmwave is not None:
            print(f"  mmWave: {self.mmwave.shape}")

    def _compute_all_limb_lengths(self, poses: np.ndarray) -> torch.Tensor:
        """Compute limb lengths for all samples."""
        limb_pairs = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
            (0, 7), (7, 8), (8, 9), (9, 10),
            (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)
        ]

        limb_lengths = np.zeros((len(poses), len(limb_pairs)), dtype=np.float32)
        for i, (src, dst) in enumerate(limb_pairs):
            limb_lengths[:, i] = np.sqrt(((poses[:, src] - poses[:, dst]) ** 2).sum(axis=-1))

        return torch.from_numpy(limb_lengths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (csi, mmwave, pose, limb_len)"""
        csi = self.csi[idx]
        pose = self.poses[idx]
        limb_len = self.limb_lengths[idx]

        if self.mmwave is not None:
            mmwave = self.mmwave[idx]
        else:
            mmwave = torch.zeros(5, 150, 5)

        return csi, mmwave, pose, limb_len

    def __len__(self) -> int:
        return len(self.csi)

class MMFiPreprocessedCSIOnly(Dataset):
    """
    Fast CSI-only dataset for Phase 2 training.

    Returns: (csi, pose, limb_len) tuple
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 root_relative: bool = True):
        self._base = MMFiPreprocessed(data_root, split, with_mmwave=False, root_relative=root_relative)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (csi, pose, limb_len)"""
        csi, _, pose, limb_len = self._base[idx]
        return csi, pose, limb_len

    def __len__(self) -> int:
        return len(self._base)

class MMFiPreprocessedWithEnvId(Dataset):
    """
    CSI-only dataset that also returns environment ID for room-wise normalization.

    Returns: (csi, pose, env_id) tuple
        - csi: [5, 3, 3, 114] CSI tensor
        - pose: [17, 3] pose tensor (absolute or root-relative)
        - env_id: int (0=E01, 1=E02, 2=E03, 3=E04)

    Environment mapping (from MM-Fi):
        E01 (env_id=0): S01-S10 (Room1 + Horizontal)
        E02 (env_id=1): S11-S20 (Room1 + Vertical)
        E03 (env_id=2): S21-S30 (Room2 + Horizontal)
        E04 (env_id=3): S31-S40 (Room2 + Vertical)
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 root_relative: bool = False):
        import os

        self._base = MMFiPreprocessed(data_root, split, with_mmwave=False, root_relative=root_relative)

        sample_ids_path = os.path.join(data_root, f'sample_ids_{split}.npy')
        if os.path.exists(sample_ids_path):
            self.sample_ids = np.load(sample_ids_path, allow_pickle=True)
            self.env_ids = [self._get_env_id(str(sid)) for sid in self.sample_ids]
            print(f"  Loaded {len(self.env_ids)} sample IDs with env mapping")
        else:
            print(f"  Warning: sample_ids_{split}.npy not found, using default env_id=0")
            self.sample_ids = None
            self.env_ids = [0] * len(self._base)

    def _get_env_id(self, sample_id: str) -> int:
        """
        Get environment ID from sample ID.

        Args:
            sample_id: e.g., "S01_A02_000"

        Returns:
            env_id: 0=E01, 1=E02, 2=E03, 3=E04
        """
        try:
            subject_str = sample_id.split('_')[0]
            subject_num = int(subject_str[1:])
            if 1 <= subject_num <= 10:
                return 0
            elif 11 <= subject_num <= 20:
                return 1
            elif 21 <= subject_num <= 30:
                return 2
            else:
                return 3
        except (ValueError, IndexError):
            return 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Returns: (csi, pose, env_id)"""
        csi, _, pose, _ = self._base[idx]
        return csi, pose, self.env_ids[idx]

    def __len__(self) -> int:
        return len(self._base)

class MMFiCSIWithSceneInfo(Dataset):
    """
    MM-Fi Dataset for CSI with scene information for physics-aware training.

    Supports both:
    1. Preprocessed data (npy files) - uses default scene values if not available
    2. Raw data structure (E01-E04 directories) - parses scene from directory structure

    Returns tuple: (csi, pose, limb_len, orientation_id, room_id)

    Scene configuration (from MM-Fi paper):
        E01: Room 1 + Horizontal sensor deployment
        E02: Room 1 + Vertical sensor deployment
        E03: Room 2 + Horizontal sensor deployment
        E04: Room 2 + Vertical sensor deployment

    Args:
        orientation_id: 0=horizontal, 1=vertical
        room_id: 0=Room1, 1=Room2
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 protocol: str = 'cross_scene',
                 num_frames: int = 5,
                 random_seed: int = 42,
                 root_relative: bool = True):
        self.data_root = data_root
        self.split = split
        self._is_preprocessed = self._check_preprocessed(data_root)

        if self._is_preprocessed:
            self._base = MMFiPreprocessed(data_root, split, with_mmwave=False, root_relative=root_relative)
            self._scene_ids = self._load_scene_ids(data_root, split)
            print(f"  Using preprocessed data with scene info (has_scene_ids={self._scene_ids is not None})")
        else:
            self._base = MMFiCSImmWavePaired(
                data_root=data_root,
                split=split,
                protocol=protocol,
                num_frames=num_frames,
                random_seed=random_seed,
                root_relative=root_relative
            )
            self._scene_ids = None
            print(f"  Using raw data with scene info from directory structure")

        print(f"  Scene info: orientation (0=horizontal, 1=vertical), room (0=Room1, 1=Room2)")

    def _check_preprocessed(self, data_root: str) -> bool:
        """Check if data_root contains preprocessed npy files."""
        import os
        csi_file = os.path.join(data_root, 'csi_array_train.npy')
        return os.path.exists(csi_file)

    def _load_scene_ids(self, data_root: str, split: str) -> Optional[np.ndarray]:
        """Load scene IDs from preprocessed data if available."""
        import os
        scene_file = os.path.join(data_root, f'scene_ids_{split}.npy')
        if os.path.exists(scene_file):
            return np.load(scene_file, allow_pickle=True)

        sample_file = os.path.join(data_root, f'sample_ids_{split}.npy')
        if os.path.exists(sample_file):
            sample_ids = np.load(sample_file, allow_pickle=True)
            scene_ids = []
            for sid in sample_ids:
                if isinstance(sid, str) and sid.startswith('S'):
                    try:
                        subject_num = int(sid[1:3])
                        scene_ids.append(self.subject_to_scene(subject_num))
                    except (ValueError, IndexError):
                        scene_ids.append("E01")
                else:
                    scene_ids.append("E01")
            return np.array(scene_ids)

        return None

    @staticmethod
    def subject_to_scene(subject_num: int) -> str:
        """
        Convert subject number to scene/environment.

        MM-Fi dataset mapping:
            S01-S10 → E01 (Room1 + Horizontal)
            S11-S20 → E02 (Room1 + Vertical)
            S21-S30 → E03 (Room2 + Horizontal)
            S31-S40 → E04 (Room2 + Vertical)

        Args:
            subject_num: Subject number (1-40)

        Returns:
            Scene string "E01", "E02", "E03", or "E04"
        """
        if subject_num <= 10:
            return "E01"
        elif subject_num <= 20:
            return "E02"
        elif subject_num <= 30:
            return "E03"
        else:
            return "E04"

    @staticmethod
    def parse_scene(scene_str: str) -> Tuple[int, int]:
        """
        Parse scene string to physical configuration.

        Args:
            scene_str: "E01", "E02", "E03", "E04" or None

        Returns:
            orientation_id: 0=horizontal, 1=vertical
            room_id: 0=Room1, 1=Room2
        """
        if scene_str is None:
            return 0, 0

        try:
            env_id = int(scene_str[1:]) - 1
            room_id = env_id // 2
            orientation_id = env_id % 2
            return orientation_id, room_id
        except (ValueError, IndexError):
            return 0, 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (csi, pose, limb_len, orientation_id, room_id)
        """
        if self._is_preprocessed:
            csi, _, pose, limb_len = self._base[idx]

            if self._scene_ids is not None:
                scene_str = self._scene_ids[idx]
                orientation_id, room_id = self.parse_scene(scene_str)
            else:
                orientation_id, room_id = 0, 0
        else:
            sample = self._base.samples[idx]
            scene = sample.get('scene', None)
            csi, _, pose, limb_len = self._base[idx]
            orientation_id, room_id = self.parse_scene(scene)

        return (
            csi,
            pose,
            limb_len,
            torch.tensor(orientation_id, dtype=torch.long),
            torch.tensor(room_id, dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self._base)

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Testing MMFiCSImmWavePaired with different protocols")
    print("=" * 60)

    data_root = sys.argv[1] if len(sys.argv) > 1 else "/path/to/MMFi"

    if not os.path.exists(data_root):
        print(f"Data root not found: {data_root}")
        print("Usage: python mmfi_csi_mmwave.py /path/to/MMFi_Dataset")
        exit(1)

    for protocol in ['cross_scene', 'cross_subject', 'random_split']:
        print(f"\n{'='*40}")
        print(f"Protocol: {protocol}")
        print(f"{'='*40}")

        train_set = MMFiCSImmWavePaired(data_root, split='train', protocol=protocol)
        test_set = MMFiCSImmWavePaired(data_root, split='test', protocol=protocol)

        print(f"  Train samples: {len(train_set)}")
        print(f"  Test samples: {len(test_set)}")

        train_paths = set(s['action_path'] for s in train_set.samples)
        test_paths = set(s['action_path'] for s in test_set.samples)
        overlap = train_paths & test_paths

        if overlap:
            print(f"  ⚠️  WARNING: {len(overlap)} overlapping action paths!")
            if protocol != 'random_split':
                print(f"     This is unexpected for {protocol} protocol!")
        else:
            print(f"  ✅ No data leakage (0 overlapping paths)")

        if len(train_set.samples) > 0:
            train_scenes = set(s.get('scene', 'N/A') for s in train_set.samples)
            test_scenes = set(s.get('scene', 'N/A') for s in test_set.samples)
            train_subjects = set(s.get('subject', 'N/A') for s in train_set.samples)
            test_subjects = set(s.get('subject', 'N/A') for s in test_set.samples)

            print(f"  Train scenes: {sorted(train_scenes)}")
            print(f"  Test scenes: {sorted(test_scenes)}")
            print(f"  Train subjects ({len(train_subjects)}): {sorted(train_subjects)[:5]}...")
            print(f"  Test subjects ({len(test_subjects)}): {sorted(test_subjects)}")

            subject_overlap = train_subjects & test_subjects
            if subject_overlap and protocol != 'random_split':
                print(f"  ⚠️  Subject overlap: {sorted(subject_overlap)}")
            elif not subject_overlap:
                print(f"  ✅ No subject overlap")
