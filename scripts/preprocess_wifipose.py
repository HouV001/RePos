#!/usr/bin/env python3
"""
Preprocess Person-in-WiFi-3D (WiFiPose3D) Dataset

Converts raw .mat CSI files and .npy keypoint files to preprocessed numpy arrays
for fast training with WiFiPose3DPreprocessed dataset loader.

Input structure (raw dataset):
    data_root/
    ├── train_data/
    │   ├── train_data_list.txt
    │   ├── csi/
    │   │   └── *.mat
    │   └── keypoint/
    │       └── *.npy
    └── test_data/
        ├── test_data_list.txt
        ├── csi/
        │   └── *.mat
        └── keypoint/
            └── *.npy

Output structure:
    output_dir/
    ├── csi_array_train.npy   # [N, T, 3, 3, 114]
    ├── csi_array_test.npy
    ├── pose_array_train.npy  # [N, 17, 3] absolute coordinates
    └── pose_array_test.npy

Usage:
    python scripts/preprocess_wifipose.py \
        --data_root /path/to/wifipose_dataset \
        --output_dir ./data/wifipose \
        --num_frames 5 \
        --num_subcarriers 114
"""

import os
import sys
import argparse
import numpy as np
import h5py
import pywt
from tqdm import tqdm
from typing import List, Tuple

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
    """Apply discrete wavelet transform denoising."""
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

def process_csi(csi_path: str, num_frames: int, num_subcarriers: int,
                use_phase: bool = True, normalization: str = 'minmax') -> np.ndarray:
    """
    Load and preprocess a single CSI file.

    Returns:
        csi: shape (num_frames, 3, 3, num_subcarriers)
    """
    from scipy.ndimage import zoom

    csi_complex = load_csi_mat(csi_path)

    csi = csi_complex.transpose(2, 3, 1, 0)

    if use_phase:
        csi_denoised = phase_denoise(csi)
    else:
        csi_denoised = csi

    csi_amp = np.abs(csi_denoised)

    csi_amp = dwt_denoise(csi_amp)

    csi_amp = csi_amp.transpose(3, 0, 1, 2)

    available_frames = csi_amp.shape[0]

    if num_frames < available_frames:
        indices = np.linspace(0, available_frames - 1, num_frames, dtype=int)
        csi_amp = csi_amp[indices]
    elif num_frames > available_frames:
        pad_size = num_frames - available_frames
        csi_amp = np.pad(csi_amp, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='edge')

    if csi_amp.shape[-1] != num_subcarriers:
        scale = num_subcarriers / csi_amp.shape[-1]
        csi_amp = zoom(csi_amp, (1, 1, 1, scale), order=1)

    if normalization == 'minmax':
        csi_min = csi_amp.min()
        csi_max = csi_amp.max()
        csi_amp = (csi_amp - csi_min) / (csi_max - csi_min + 1e-8)
    else:
        mean = csi_amp.mean()
        std = csi_amp.std() + 1e-8
        csi_amp = (csi_amp - mean) / std
        csi_amp = np.clip(csi_amp, -10.0, 10.0)

    return csi_amp.astype(np.float32)

def load_file_list(file_path: str, experiment: str = 'one-person') -> List[str]:
    """Load file names from list file."""
    file_names = []
    with open(file_path, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                pose_number = int(name.split('_')[0][2])
                if experiment == 'all-person':
                    file_names.append(name)
                elif experiment == 'one-person' and pose_number == 1:
                    file_names.append(name)
                elif experiment == 'two-person' and pose_number == 2:
                    file_names.append(name)
                elif experiment == 'three-person' and pose_number == 3:
                    file_names.append(name)
    return file_names

def preprocess_split(data_dir: str, file_list: List[str],
                     num_frames: int, num_subcarriers: int,
                     use_phase: bool, normalization: str,
                     joint_format: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess all samples in a split.

    Returns:
        csi_array: [N, num_frames, 3, 3, num_subcarriers]
        pose_array: [N, num_joints, 3]
    """
    csi_list = []
    pose_list = []

    for file_name in tqdm(file_list, desc="Processing"):
        csi_path = os.path.join(data_dir, 'csi', f'{file_name}.mat')
        if not os.path.exists(csi_path):
            print(f"Warning: CSI file not found: {csi_path}")
            continue

        try:
            csi = process_csi(csi_path, num_frames, num_subcarriers, use_phase, normalization)
        except Exception as e:
            print(f"Error processing {csi_path}: {e}")
            continue

        kp_path = os.path.join(data_dir, 'keypoint', f'{file_name}.npy')
        if not os.path.exists(kp_path):
            print(f"Warning: Keypoint file not found: {kp_path}")
            continue

        keypoints = np.load(kp_path)
        kp = keypoints[0]

        if joint_format == 'coco17':
            kp = map_14_to_17_joints(kp)

        csi_list.append(csi)
        pose_list.append(kp)

    csi_array = np.stack(csi_list, axis=0)
    pose_array = np.stack(pose_list, axis=0).astype(np.float32)

    return csi_array, pose_array

def main():
    parser = argparse.ArgumentParser(description='Preprocess WiFiPose3D dataset')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to raw WiFiPose3D dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for preprocessed files')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of temporal frames (default: 5)')
    parser.add_argument('--num_subcarriers', type=int, default=114,
                        help='Number of subcarriers after interpolation (default: 114)')
    parser.add_argument('--experiment', type=str, default='one-person',
                        choices=['one-person', 'two-person', 'three-person', 'all-person'],
                        help='Experiment setting (default: one-person)')
    parser.add_argument('--joint_format', type=str, default='wifi14',
                        choices=['coco17', 'wifi14'],
                        help='Joint format (default: wifi14 for DT-Pose compatibility)')
    parser.add_argument('--normalization', type=str, default='minmax',
                        choices=['minmax', 'zscore'],
                        help='CSI normalization (default: minmax for DT-Pose compatibility)')
    parser.add_argument('--no_phase', action='store_true',
                        help='Disable phase denoising')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    use_phase = not args.no_phase

    print(f"WiFiPose3D Preprocessing")
    print(f"========================")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Experiment: {args.experiment}")
    print(f"Frames: {args.num_frames}, Subcarriers: {args.num_subcarriers}")
    print(f"Joint format: {args.joint_format}")
    print(f"Normalization: {args.normalization}")
    print(f"Phase denoising: {use_phase}")
    print()

    print("Processing TRAIN split...")
    train_dir = os.path.join(args.data_root, 'train_data')
    train_list_path = os.path.join(train_dir, 'train_data_list.txt')

    if not os.path.exists(train_list_path):
        print(f"Error: Train list not found: {train_list_path}")
        sys.exit(1)

    train_files = load_file_list(train_list_path, args.experiment)
    print(f"Found {len(train_files)} train samples")

    train_csi, train_pose = preprocess_split(
        train_dir, train_files,
        args.num_frames, args.num_subcarriers,
        use_phase, args.normalization, args.joint_format
    )

    print(f"Train CSI shape: {train_csi.shape}")
    print(f"Train Pose shape: {train_pose.shape}")

    np.save(os.path.join(args.output_dir, 'csi_array_train.npy'), train_csi)
    np.save(os.path.join(args.output_dir, 'pose_array_train.npy'), train_pose)
    print(f"Saved train arrays")
    print()

    print("Processing TEST split...")
    test_dir = os.path.join(args.data_root, 'test_data')
    test_list_path = os.path.join(test_dir, 'test_data_list.txt')

    if not os.path.exists(test_list_path):
        print(f"Error: Test list not found: {test_list_path}")
        sys.exit(1)

    test_files = load_file_list(test_list_path, args.experiment)
    print(f"Found {len(test_files)} test samples")

    test_csi, test_pose = preprocess_split(
        test_dir, test_files,
        args.num_frames, args.num_subcarriers,
        use_phase, args.normalization, args.joint_format
    )

    print(f"Test CSI shape: {test_csi.shape}")
    print(f"Test Pose shape: {test_pose.shape}")

    np.save(os.path.join(args.output_dir, 'csi_array_test.npy'), test_csi)
    np.save(os.path.join(args.output_dir, 'pose_array_test.npy'), test_pose)
    print(f"Saved test arrays")
    print()

    print("Pose Statistics (absolute coordinates):")
    all_poses = np.concatenate([train_pose, test_pose], axis=0)
    print(f"  X range: [{all_poses[:,:,0].min():.3f}, {all_poses[:,:,0].max():.3f}] m")
    print(f"  Y range: [{all_poses[:,:,1].min():.3f}, {all_poses[:,:,1].max():.3f}] m")
    print(f"  Z range: [{all_poses[:,:,2].min():.3f}, {all_poses[:,:,2].max():.3f}] m")

    pelvis_idx = 11 if args.joint_format == 'coco17' else 8
    pelvis_all = (all_poses[:, pelvis_idx] + all_poses[:, pelvis_idx + 1]) / 2
    scene_center = pelvis_all.mean(axis=0)
    print(f"  Scene center (pelvis mean): [{scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f}]")

    print()
    print("Done!")
    print(f"Output files saved to: {args.output_dir}")
    print()
    print("Usage:")
    print(f"  python train_v2.py --phase 1 --dataset wifipose --data_root {args.output_dir}")

if __name__ == '__main__':
    main()
