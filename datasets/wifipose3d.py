"""
Person-in-WiFi-3D Dataset Loader

Dataset format:
- CSI: .mat files with 'csi_out' key, shape (20, 30, 3, 3) - complex values
  - 20: time packets
  - 30: subcarriers
  - 3x3: MIMO antennas
- Keypoints: .npy files, shape (N, 14, 3) - N persons, 14 joints, xyz

Joint mapping (Person-in-WiFi-3D 14 joints to COCO 17 joints):
  WiFiPose3D (14):          COCO (17):
  0: head                   0: nose
  1: neck                   1: left_eye (interpolate)
  2: left_shoulder          2: right_eye (interpolate)
  3: right_shoulder         3: left_ear (interpolate)
  4: left_elbow             4: right_ear (interpolate)
  5: right_elbow            5: left_shoulder
  6: left_wrist             6: right_shoulder
  7: right_wrist            7: left_elbow
  8: left_hip               8: right_elbow
  9: right_hip              9: left_wrist
  10: left_knee             10: right_wrist
  11: right_knee            11: left_hip
  12: left_ankle            12: right_hip
  13: right_ankle           13: left_knee
                            14: right_knee
                            15: left_ankle
                            16: right_ankle
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Optional
import h5py
import pywt

def load_csi_mat(path: str) -> np.ndarray:
    """Load CSI from MATLAB v7.3 .mat file.

    Returns:
        csi: Complex array of shape (20, 30, 3, 3) - (time, subcarriers, tx, rx)
    """
    with h5py.File(path, 'r') as f:
        csi_raw = f['csi_out'][:]
        csi = csi_raw['real'] + 1j * csi_raw['imag']
        csi = np.array(csi).transpose(3, 2, 1, 0)
        csi = csi.transpose(3, 2, 0, 1)
    return csi.astype(np.complex128)

def csi_sanitization(csi_rx: np.ndarray) -> np.ndarray:
    """
    Phase sanitization for one receiver's CSI.
    Removes timing offset and sampling frequency offset.

    Args:
        csi_rx: shape (3, 30, T) - 3 TX antennas, 30 subcarriers, T packets

    Returns:
        Sanitized CSI with same shape
    """
    one_csi = csi_rx[0, :, :]
    two_csi = csi_rx[1, :, :]
    three_csi = csi_rx[2, :, :]

    pi = np.pi
    M = 3
    N = 30
    T = one_csi.shape[1]
    fi = 312.5 * 2

    csi_phase = np.zeros((M, N, T))

    for t in range(T):
        csi_phase[0, :, t] = np.unwrap(np.angle(one_csi[:, t]))
        csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(two_csi[:, t] * np.conj(one_csi[:, t])))
        csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(three_csi[:, t] * np.conj(two_csi[:, t])))

        ai = np.tile(2 * pi * fi * np.arange(N), M)
        bi = np.ones(M * N)
        ci = np.concatenate([csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t]])

        A = np.dot(ai, ai)
        B = np.dot(ai, bi)
        C = np.dot(bi, bi)
        D = np.dot(ai, ci)
        E = np.dot(bi, ci)

        rho_opt = (B * E - C * D) / (A * C - B ** 2)
        beta_opt = (B * D - A * E) / (A * C - B ** 2)

        temp = np.tile(np.arange(N), M).reshape(M, N)
        csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt

    antennaPair_One = np.abs(one_csi) * np.exp(1j * csi_phase[0, :, :])
    antennaPair_Two = np.abs(two_csi) * np.exp(1j * csi_phase[1, :, :])
    antennaPair_Three = np.abs(three_csi) * np.exp(1j * csi_phase[2, :, :])

    return np.stack([antennaPair_One, antennaPair_Two, antennaPair_Three], axis=0)

def phase_denoise(csi: np.ndarray) -> np.ndarray:
    """
    Apply phase denoising to CSI.

    Args:
        csi: shape (3, 3, 30, 20) - (rx, tx, subcarriers, time)

    Returns:
        Denoised CSI with same shape
    """
    result = []
    for rx in range(3):
        sanitized = csi_sanitization(csi[rx])
        result.append(sanitized)
    return np.stack(result, axis=0)

def dwt_denoise(csi_amp: np.ndarray) -> np.ndarray:
    """
    Apply discrete wavelet transform denoising.

    Args:
        csi_amp: CSI amplitude array

    Returns:
        Denoised amplitude
    """
    w = pywt.Wavelet('db11')
    coeffs = pywt.wavedec(csi_amp, w, mode='symmetric')
    return pywt.waverec(coeffs, w)

def map_14_to_17_joints(joints_14: np.ndarray) -> np.ndarray:
    """
    Map Person-in-WiFi-3D 14 joints to COCO 17 joints format.

    WiFiPose3D 14 joints:
        0: head, 1: neck, 2: left_shoulder, 3: right_shoulder,
        4: left_elbow, 5: right_elbow, 6: left_wrist, 7: right_wrist,
        8: left_hip, 9: right_hip, 10: left_knee, 11: right_knee,
        12: left_ankle, 13: right_ankle

    COCO 17 joints:
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    Args:
        joints_14: shape (14, 3)

    Returns:
        joints_17: shape (17, 3)
    """
    joints_17 = np.zeros((17, 3), dtype=joints_14.dtype)

    joints_17[0] = joints_14[0]
    joints_17[5] = joints_14[2]
    joints_17[6] = joints_14[3]
    joints_17[7] = joints_14[4]
    joints_17[8] = joints_14[5]
    joints_17[9] = joints_14[6]
    joints_17[10] = joints_14[7]
    joints_17[11] = joints_14[8]
    joints_17[12] = joints_14[9]
    joints_17[13] = joints_14[10]
    joints_17[14] = joints_14[11]
    joints_17[15] = joints_14[12]
    joints_17[16] = joints_14[13]

    head = joints_14[0]
    neck = joints_14[1]
    left_shoulder = joints_14[2]
    right_shoulder = joints_14[3]

    head_neck_mid = (head + neck) / 2
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    eye_offset = shoulder_width * 0.15

    joints_17[1] = head_neck_mid + np.array([eye_offset, 0, 0])
    joints_17[2] = head_neck_mid + np.array([-eye_offset, 0, 0])

    ear_offset = shoulder_width * 0.25
    joints_17[3] = head + np.array([ear_offset, 0, 0])
    joints_17[4] = head + np.array([-ear_offset, 0, 0])

    return joints_17

WIFIPOSE_SCENE_CENTER = np.array([1.546, 1.270, 3.204], dtype=np.float32)

class WiFiPose3DDataset(Dataset):
    """
    Person-in-WiFi-3D Dataset loader.

    Converts CSI from (20, 30, 3, 3) to model-expected format and
    maps 14 joints to 17 joints.

    Args:
        data_root: Path to dataset (containing train_data/ or test_data/)
        split: 'train' or 'test'
        num_frames: Number of temporal frames to use (default: 5)
        num_subcarriers: Target number of subcarriers (default: 114 for padding)
        use_phase: Whether to include phase information
        root_relative: Use root-relative coordinates (pelvis at origin)
        joint_format: 'coco17' or 'wifi14'
        normalization: CSI normalization method - 'zscore' (default) or 'minmax' (DT-Pose style)
        pose_center: Pose centering mode:
            - 'none': absolute coordinates (raw, for DT-Pose comparison)
            - 'pelvis': root-relative (pelvis at origin, default if root_relative=True)
            - 'scene': scene-centered (subtract scene center, easier to learn)
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 num_frames: int = 5,
                 num_subcarriers: int = 114,
                 use_phase: bool = True,
                 root_relative: bool = True,
                 joint_format: str = 'coco17',
                 normalization: str = 'zscore',
                 pose_center: str = None):

        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.num_subcarriers = num_subcarriers
        self.use_phase = use_phase
        self.root_relative = root_relative
        self.joint_format = joint_format
        self.normalization = normalization

        if pose_center is not None:
            self.pose_center = pose_center
        elif root_relative:
            self.pose_center = 'pelvis'
        else:
            self.pose_center = 'none'

        if split == 'train':
            self.data_dir = os.path.join(data_root, 'train_data')
        else:
            self.data_dir = os.path.join(data_root, 'test_data')

        list_file = os.path.join(self.data_dir, f'{split}_data_list.txt')
        self.file_list = self._load_file_list(list_file)

        self.num_joints = 17 if joint_format == 'coco17' else 14

        print(f"WiFiPose3DDataset ({split}, {joint_format}, pose_center={self.pose_center}, {normalization}): {len(self.file_list)} samples")

    def _load_file_list(self, file_path: str) -> List[str]:
        """Load file names from list file."""
        file_names = []
        with open(file_path, 'r') as f:
            for line in f:
                name = line.strip()
                if name:
                    file_names.append(name)
        return file_names

    def _load_csi(self, file_name: str) -> np.ndarray:
        """
        Load and preprocess CSI data.

        Returns:
            csi: shape (num_frames, 3, 3, num_subcarriers)
        """
        csi_path = os.path.join(self.data_dir, 'csi', f'{file_name}.mat')

        csi_complex = load_csi_mat(csi_path)

        csi = csi_complex.transpose(2, 3, 1, 0)

        if self.use_phase:
            csi_denoised = phase_denoise(csi)
        else:
            csi_denoised = csi

        csi_amp = np.abs(csi_denoised)

        csi_amp = dwt_denoise(csi_amp)

        csi_amp = csi_amp.transpose(3, 0, 1, 2)

        available_frames = csi_amp.shape[0]

        if self.num_frames >= available_frames:
            pass
        elif self.num_frames < available_frames:
            indices = np.linspace(0, available_frames - 1, self.num_frames, dtype=int)
            csi_amp = csi_amp[indices]

        if csi_amp.shape[0] < self.num_frames:
            pad_size = self.num_frames - csi_amp.shape[0]
            csi_amp = np.pad(csi_amp, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='edge')

        if csi_amp.shape[-1] < self.num_subcarriers:
            from scipy.ndimage import zoom
            scale = self.num_subcarriers / csi_amp.shape[-1]
            csi_amp = zoom(csi_amp, (1, 1, 1, scale), order=1)

        if self.normalization == 'minmax':
            csi_min = csi_amp.min()
            csi_max = csi_amp.max()
            csi_amp = (csi_amp - csi_min) / (csi_max - csi_min + 1e-8)
        else:
            mean = csi_amp.mean()
            std = csi_amp.std() + 1e-8
            csi_amp = (csi_amp - mean) / std
            csi_amp = np.clip(csi_amp, -10.0, 10.0)

        return csi_amp.astype(np.float32)

    def _load_keypoints(self, file_name: str) -> np.ndarray:
        """
        Load keypoints and optionally convert to COCO 17 format.

        Returns:
            keypoints: shape (num_joints, 3)
        """
        kp_path = os.path.join(self.data_dir, 'keypoint', f'{file_name}.npy')
        keypoints = np.load(kp_path)

        kp = keypoints[0]

        if self.joint_format == 'coco17':
            kp = map_14_to_17_joints(kp)

        if self.pose_center == 'pelvis':
            if self.joint_format == 'coco17':
                pelvis = (kp[11] + kp[12]) / 2
            else:
                pelvis = (kp[5] + kp[7]) / 2
            kp = kp - pelvis
        elif self.pose_center == 'scene':
            kp = kp - WIFIPOSE_SCENE_CENTER

        return kp.astype(np.float32)

    def _compute_limb_lengths(self, pose: np.ndarray) -> np.ndarray:
        """Compute limb lengths from pose."""
        if self.joint_format == 'coco17':
            limb_pairs = [
                (0, 5), (0, 6),
                (5, 7), (7, 9),
                (6, 8), (8, 10),
                (5, 11), (6, 12),
                (11, 12),
                (11, 13), (13, 15),
                (12, 14), (14, 16),
            ]
        else:
            limb_pairs = [
                (0, 1),
                (0, 2), (0, 3),
                (2, 4), (4, 8),
                (3, 6), (6, 10),
                (2, 5), (3, 7),
                (5, 9), (9, 12),
                (7, 11), (11, 13),
            ]

        limb_lengths = []
        for src, dst in limb_pairs:
            length = np.sqrt(((pose[src] - pose[dst]) ** 2).sum())
            limb_lengths.append(length)

        while len(limb_lengths) < 16:
            limb_lengths.append(0.0)

        return np.array(limb_lengths[:16], dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            csi: [num_frames, 3, 3, num_subcarriers]
            pose: [num_joints, 3]
            limb_len: [16]
        """
        file_name = self.file_list[idx]

        csi = self._load_csi(file_name)
        pose = self._load_keypoints(file_name)
        limb_len = self._compute_limb_lengths(pose)

        return (
            torch.from_numpy(csi),
            torch.from_numpy(pose),
            torch.from_numpy(limb_len)
        )

    def __len__(self) -> int:
        return len(self.file_list)

class WiFiPose3DPreprocessed(Dataset):
    """
    Fast loading from preprocessed numpy arrays.

    Expected files:
        csi_array_{split}.npy: [N, T, 3, 3, num_subcarriers]
        pose_array_{split}.npy: [N, num_joints, 3] - 14 (wifi14) or 17 (coco17)
    """

    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 root_relative: bool = True):

        self.data_root = data_root
        self.split = split
        self.root_relative = root_relative

        csi_path = os.path.join(data_root, f'csi_array_{split}.npy')
        pose_path = os.path.join(data_root, f'pose_array_{split}.npy')

        print(f"Loading {csi_path}...")
        self.csi = torch.from_numpy(np.load(csi_path).astype(np.float32))

        print(f"Loading {pose_path}...")
        poses = np.load(pose_path).astype(np.float32)

        self.num_joints = poses.shape[1]
        self.joint_format = 'wifi14' if self.num_joints == 14 else 'coco17'

        if root_relative:
            if self.num_joints == 14:
                pelvis = (poses[:, 5:6, :] + poses[:, 7:8, :]) / 2
            else:
                pelvis = (poses[:, 11:12, :] + poses[:, 12:13, :]) / 2
            poses = poses - pelvis

        self.poses = torch.from_numpy(poses)
        self.limb_lengths = self._compute_all_limb_lengths(poses)

        print(f"WiFiPose3DPreprocessed ({split}, {self.joint_format}): {len(self)} samples")

    def _compute_all_limb_lengths(self, poses: np.ndarray) -> torch.Tensor:
        """Compute limb lengths for all samples."""
        if self.num_joints == 14:
            limb_pairs = [
                (0, 1),
                (0, 2), (0, 3),
                (2, 4), (4, 8),
                (3, 6), (6, 10),
                (2, 5), (3, 7),
                (5, 7),
                (5, 9), (9, 12),
                (7, 11), (11, 13),
                (2, 3), (0, 0)
            ]
        else:
            limb_pairs = [
                (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (5, 6), (11, 12), (0, 0)
            ]

        limb_lengths = np.zeros((len(poses), 16), dtype=np.float32)
        for i, (src, dst) in enumerate(limb_pairs):
            if i < 16:
                limb_lengths[:, i] = np.sqrt(((poses[:, src] - poses[:, dst]) ** 2).sum(axis=-1))

        return torch.from_numpy(limb_lengths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.csi[idx], self.poses[idx], self.limb_lengths[idx]

    def __len__(self) -> int:
        return len(self.csi)

if __name__ == "__main__":
    import sys

    data_root = sys.argv[1] if len(sys.argv) > 1 else "./data/wifipose3d"

    print("Testing WiFiPose3DDataset...")

    train_set = WiFiPose3DDataset(data_root, split='train', joint_format='coco17')
    print(f"Train samples: {len(train_set)}")

    csi, pose, limb_len = train_set[0]
    print(f"CSI shape: {csi.shape}")
    print(f"Pose shape: {pose.shape}")
    print(f"Limb lengths shape: {limb_len.shape}")

    test_set = WiFiPose3DDataset(data_root, split='test', joint_format='coco17')
    print(f"Test samples: {len(test_set)}")
