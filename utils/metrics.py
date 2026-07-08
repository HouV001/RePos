"""
Evaluation Metrics for 3D Human Pose Estimation

Implements GraphPose-Fi/DT-Pose compatible evaluation:
- MPJPE with align=False: Global position error + pose error (default for comparison)
- MPJPE with align=True: Hip-aligned, only pose error
- PA-MPJPE: Procrustes-Aligned MPJPE (proper SVD-based alignment)
- PCK: Percentage of Correct Keypoints at various thresholds
- Per-joint metrics with table formatting

Reference:
    GraphPose-Fi: https://github.com/Cirrick/GraphPose-Fi
    DT-Pose: https://github.com/cseeyangchen/DT-Pose
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union

HIP_INDEX_17 = 0
TORSO_JOINTS_17 = (5, 12)

JOINT_NAMES_17 = [
    'Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle',
    'R_Hip', 'R_Knee', 'R_Ankle',
    'Spine', 'Thorax', 'Neck', 'Head',
    'L_Shoulder', 'L_Elbow', 'L_Wrist',
    'R_Shoulder', 'R_Elbow', 'R_Wrist'
]

BODY_PARTS_17 = {
    'Head': [9, 10],
    'Torso': [0, 7, 8],
    'L_Arm': [11, 12, 13],
    'R_Arm': [14, 15, 16],
    'L_Leg': [1, 2, 3],
    'R_Leg': [4, 5, 6],
}

HIP_INDEX_14 = (5, 7)
TORSO_JOINTS_14 = (2, 5)

JOINT_NAMES_14 = [
    'Neck', 'Head', 'Shoulder_A', 'Shoulder_B',
    'Elbow_A', 'Hip_A', 'Elbow_B', 'Hip_B',
    'Wrist_A', 'Knee_A', 'Wrist_B', 'Knee_B',
    'Ankle_A', 'Ankle_B'
]

BODY_PARTS_14 = {
    'Head': [0, 1],
    'Torso': [2, 3, 5, 7],
    'Arm_A': [2, 4, 8],
    'Arm_B': [3, 6, 10],
    'Leg_A': [5, 9, 12],
    'Leg_B': [7, 11, 13],
}

HIP_INDEX_18 = 1
TORSO_JOINTS_18 = (5, 8)
JOINT_NAMES_18 = [
    'Nose', 'Neck', 'R_Sho', 'R_Elb', 'R_Wri', 'L_Sho', 'L_Elb', 'L_Wri',
    'R_Hip', 'R_Knee', 'R_Ank', 'L_Hip', 'L_Knee', 'L_Ank', 'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear'
]
BODY_PARTS_18 = {
    'Head': [0, 14, 15, 16, 17],
    'Torso': [1],
    'R_Arm': [2, 3, 4],
    'L_Arm': [5, 6, 7],
    'R_Leg': [8, 9, 10],
    'L_Leg': [11, 12, 13],
}

HIP_INDEX = HIP_INDEX_17
JOINT_NAMES = JOINT_NAMES_17
BODY_PARTS = BODY_PARTS_17

def get_joint_config(num_joints: int = 17):
    """Get joint configuration based on number of joints."""
    if num_joints == 14:
        return {
            'hip_index': HIP_INDEX_14,
            'torso_joints': TORSO_JOINTS_14,
            'joint_names': JOINT_NAMES_14,
            'body_parts': BODY_PARTS_14,
        }
    elif num_joints == 18:
        return {
            'hip_index': HIP_INDEX_18,
            'torso_joints': TORSO_JOINTS_18,
            'joint_names': JOINT_NAMES_18,
            'body_parts': BODY_PARTS_18,
        }
    else:
        return {
            'hip_index': HIP_INDEX_17,
            'torso_joints': TORSO_JOINTS_17,
            'joint_names': JOINT_NAMES_17,
            'body_parts': BODY_PARTS_17,
        }

def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def compute_mpjpe(pred: Union[torch.Tensor, np.ndarray],
                  gt: Union[torch.Tensor, np.ndarray],
                  align: bool = False,
                  return_per_joint: bool = False,
                  num_joints: int = None) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Compute MPJPE with optional hip alignment.

    Args:
        pred: Predicted poses [B, J, 3]
        gt: Ground truth poses [B, J, 3]
        align: If True, align hip before computing error
        return_per_joint: If True, also return per-joint errors
        num_joints: Number of joints (14 or 17). Auto-detected if None.

    Returns:
        mpjpe: Mean error in mm
        per_joint: (optional) Per-joint errors [J]
    """
    pred = to_numpy(pred).copy()
    gt = to_numpy(gt)

    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        gt = gt[np.newaxis, ...]

    if num_joints is None:
        num_joints = pred.shape[1]

    if align:
        if num_joints == 14:
            pred_hip = (pred[:, 8, :] + pred[:, 9, :]) / 2
            gt_hip = (gt[:, 8, :] + gt[:, 9, :]) / 2
        else:
            pred_hip = pred[:, HIP_INDEX_17, :]
            gt_hip = gt[:, HIP_INDEX_17, :]
        offset = gt_hip - pred_hip
        pred = pred + offset[:, np.newaxis, :]

    error = np.linalg.norm(pred - gt, axis=-1)

    error_mm = error * 1000

    mpjpe = np.mean(error_mm)

    if return_per_joint:
        per_joint = np.mean(error_mm, axis=0)
        return mpjpe, per_joint

    return mpjpe

def calulate_error(preds: np.ndarray, gts: np.ndarray, align: bool = False,
                   num_joints: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    GraphPose-Fi compatible error calculation.

    This function matches GraphPose-Fi's calulate_error() exactly for fair comparison.
    Reference: https://github.com/Cirrick/GraphPose-Fi/blob/main/utils.py

    Args:
        preds: Predicted poses [B, J, 3] in meters
        gts: Ground truth poses [B, J, 3] in meters
        align: If True, translate predictions to align hip with GT hip
        num_joints: Number of joints (14 or 17). Auto-detected if None.

    Returns:
        mpjpe: Per-sample MPJPE [B] in meters
        pampjpe: Per-sample PA-MPJPE [B] in meters
        mpjpe_per_joint: Mean MPJPE per joint [J] in meters
        pampjpe_per_joint: Mean PA-MPJPE per joint [J] in meters
    """
    preds = preds.copy()
    N = preds.shape[0]
    if num_joints is None:
        num_joints = preds.shape[1]

    if align:
        if num_joints == 14:
            preds_hip = (preds[:, 8, :] + preds[:, 9, :]) / 2
            gts_hip = (gts[:, 8, :] + gts[:, 9, :]) / 2
        else:
            preds_hip = preds[:, HIP_INDEX_17, :]
            gts_hip = gts[:, HIP_INDEX_17, :]
        offset = gts_hip - preds_hip
        preds = preds + offset[:, np.newaxis, :]

    mpjpe_joints = np.sqrt(np.sum(np.square(preds - gts), axis=2))
    mpjpe = mpjpe_joints.mean(axis=1)

    pampjpe_joints = np.zeros([N, num_joints])
    for n in range(N):
        frame_pred = preds[n]
        frame_gt = gts[n]
        aligned = procrustes_alignment(frame_pred, frame_gt)
        pampjpe_joints[n] = np.sqrt(np.sum(np.square(aligned - frame_gt), axis=1))

    pampjpe = pampjpe_joints.mean(axis=1)

    mpjpe_per_joint = mpjpe_joints.mean(axis=0)
    pampjpe_per_joint = pampjpe_joints.mean(axis=0)

    return mpjpe, pampjpe, mpjpe_per_joint, pampjpe_per_joint

def procrustes_alignment(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    Align predicted pose to ground truth using Procrustes analysis.

    Ported from DT-Pose implementation for consistency.
    Reference: https://github.com/cseeyangchen/DT-Pose/blob/main/utils.py

    Args:
        pred: Predicted pose [J, 3]
        gt: Ground truth pose [J, 3]

    Returns:
        aligned: Aligned prediction [J, 3]
    """
    mu_pred = pred.mean(axis=0)
    mu_gt = gt.mean(axis=0)

    pred_centered = pred - mu_pred
    gt_centered = gt - mu_gt

    ss_pred = (pred_centered ** 2).sum()
    ss_gt = (gt_centered ** 2).sum()

    norm_pred = np.sqrt(ss_pred)
    norm_gt = np.sqrt(ss_gt)

    pred_normalized = pred_centered / (norm_pred + 1e-8)
    gt_normalized = gt_centered / (norm_gt + 1e-8)

    A = np.dot(gt_normalized.T, pred_normalized)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    det_T = np.linalg.det(T)
    V[:, -1] *= np.sign(det_T)
    s[-1] *= np.sign(det_T)
    T = np.dot(V, U.T)

    trace_TA = s.sum()
    scale = trace_TA * norm_gt / (norm_pred + 1e-8)

    t = mu_gt - scale * np.dot(mu_pred, T)

    aligned = scale * np.dot(pred, T) + t

    return aligned

def compute_pa_mpjpe(pred: Union[torch.Tensor, np.ndarray],
                     gt: Union[torch.Tensor, np.ndarray],
                     return_per_joint: bool = False) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Compute Procrustes-Aligned MPJPE (PA-MPJPE).

    Aligns each prediction to its ground truth using Procrustes analysis
    before computing MPJPE.

    Args:
        pred: Predicted poses [B, J, 3]
        gt: Ground truth poses [B, J, 3]
        return_per_joint: If True, also return per-joint errors

    Returns:
        pa_mpjpe: Mean error in mm after alignment
        per_joint: (optional) Per-joint errors [J]
    """
    pred = to_numpy(pred)
    gt = to_numpy(gt)

    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        gt = gt[np.newaxis, ...]

    B, J, _ = pred.shape

    aligned_pred = np.zeros_like(pred)
    for i in range(B):
        aligned_pred[i] = procrustes_alignment(pred[i], gt[i])

    error = np.linalg.norm(aligned_pred - gt, axis=-1)
    error_mm = error * 1000

    pa_mpjpe = np.mean(error_mm)

    if return_per_joint:
        per_joint = np.mean(error_mm, axis=0)
        return pa_mpjpe, per_joint

    return pa_mpjpe

def compute_pck(pred: Union[torch.Tensor, np.ndarray],
                gt: Union[torch.Tensor, np.ndarray],
                thresholds: List[float] = [0.5, 0.4, 0.3, 0.2, 0.1],
                num_joints: int = None) -> Dict[str, float]:
    """
    Compute Percentage of Correct Keypoints (PCK) at various thresholds.

    Following DT-Pose convention: normalized by torso size (right_shoulder to left_hip).
    A joint is correct if: error / torso_size < threshold

    Args:
        pred: Predicted poses [B, J, 3]
        gt: Ground truth poses [B, J, 3]
        thresholds: List of thresholds as ratio of torso size (e.g., 0.5 = 50%)
        num_joints: Number of joints (14 or 17). Auto-detected if None.

    Returns:
        pck_dict: Dictionary with PCK@threshold values (in percentage)
    """
    pred = to_numpy(pred)
    gt = to_numpy(gt)

    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        gt = gt[np.newaxis, ...]

    if num_joints is None:
        num_joints = pred.shape[1]

    if num_joints == 14:
        torso_j1, torso_j2 = TORSO_JOINTS_14
    else:
        torso_j1, torso_j2 = TORSO_JOINTS_17

    torso_scale = np.sqrt(np.sum(np.square(gt[:, torso_j1, :] - gt[:, torso_j2, :]), axis=1))
    torso_scale = np.maximum(torso_scale, 1e-8)

    error = np.linalg.norm(pred - gt, axis=-1)

    normalized_error = error / torso_scale[:, np.newaxis]

    pck_dict = {}
    for thresh in thresholds:
        correct = (normalized_error < thresh).astype(float)
        pck = np.mean(correct) * 100
        pck_dict[f'PCK@{int(thresh*100)}'] = pck

    return pck_dict

def compute_per_joint_pck(pred: Union[torch.Tensor, np.ndarray],
                          gt: Union[torch.Tensor, np.ndarray],
                          threshold: float = 0.5,
                          num_joints: int = None) -> np.ndarray:
    """
    Compute PCK for each joint at a specific threshold.

    Following DT-Pose convention: normalized by torso size.

    Args:
        pred: Predicted poses [B, J, 3]
        gt: Ground truth poses [B, J, 3]
        threshold: Threshold as ratio of torso size (e.g., 0.5 = 50%)
        num_joints: Number of joints (14 or 17). Auto-detected if None.

    Returns:
        per_joint_pck: PCK for each joint [J] (in percentage)
    """
    pred = to_numpy(pred)
    gt = to_numpy(gt)

    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        gt = gt[np.newaxis, ...]

    if num_joints is None:
        num_joints = pred.shape[1]

    if num_joints == 14:
        torso_j1, torso_j2 = TORSO_JOINTS_14
    else:
        torso_j1, torso_j2 = TORSO_JOINTS_17

    torso_scale = np.sqrt(np.sum(np.square(gt[:, torso_j1, :] - gt[:, torso_j2, :]), axis=1))
    torso_scale = np.maximum(torso_scale, 1e-8)

    error = np.linalg.norm(pred - gt, axis=-1)
    normalized_error = error / torso_scale[:, np.newaxis]

    correct = (normalized_error < threshold).astype(float)
    per_joint_pck = np.mean(correct, axis=0) * 100

    return per_joint_pck

def compute_body_part_error(pred: Union[torch.Tensor, np.ndarray],
                            gt: Union[torch.Tensor, np.ndarray],
                            num_joints: int = None) -> Dict[str, float]:
    """
    Compute MPJPE for each body part.

    Args:
        pred: Predicted poses [B, J, 3]
        gt: Ground truth poses [B, J, 3]
        num_joints: Number of joints (14 or 17). Auto-detected if None.

    Returns:
        part_errors: Dictionary with body part errors in mm
    """
    pred = to_numpy(pred)
    gt = to_numpy(gt)

    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        gt = gt[np.newaxis, ...]

    if num_joints is None:
        num_joints = pred.shape[1]

    body_parts = BODY_PARTS_14 if num_joints == 14 else BODY_PARTS_17

    error = np.linalg.norm(pred - gt, axis=-1) * 1000

    part_errors = {}
    for part_name, joint_indices in body_parts.items():
        part_error = np.mean(error[:, joint_indices])
        part_errors[part_name] = part_error

    return part_errors

class PoseMetrics:
    """
    Comprehensive pose estimation metrics calculator.

    Follows GraphPose-Fi/DT-Pose convention:
    - mpjpe: MPJPE with align=False (global position + pose error)
    - mpjpe_aligned: MPJPE with align=True (pose error only, hip aligned)
    - pa_mpjpe: Procrustes-Aligned MPJPE

    Accumulates predictions and ground truths, then computes all metrics.
    Supports both 17-joint (MM-Fi) and 14-joint (WiFiPose3D) formats.
    """

    def __init__(self, pck_thresholds: List[float] = [0.5, 0.4, 0.3, 0.2, 0.1],
                 num_joints: int = None):
        """
        Args:
            pck_thresholds: List of normalized thresholds (ratio of torso size).
                           Default [0.5, 0.4, 0.3, 0.2, 0.1] -> PCK@50, PCK@40, ...
            num_joints: Number of joints (14 or 17). Auto-detected if None.
        """
        self.pck_thresholds = pck_thresholds
        self.num_joints = num_joints
        self.reset()

    def reset(self):
        """Reset accumulated data."""
        self.predictions = []
        self.ground_truths = []

    def update(self, pred: Union[torch.Tensor, np.ndarray],
               gt: Union[torch.Tensor, np.ndarray]):
        """
        Add batch of predictions and ground truths.

        Args:
            pred: Predicted poses [B, J, 3]
            gt: Ground truth poses [B, J, 3]
        """
        pred = to_numpy(pred)
        gt = to_numpy(gt)

        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
            gt = gt[np.newaxis, ...]

        self.predictions.append(pred)
        self.ground_truths.append(gt)

    def compute(self) -> Dict[str, Union[float, np.ndarray, Dict]]:
        """
        Compute all metrics following GraphPose-Fi convention.

        Returns:
            Dictionary containing:
            - mpjpe: MPJPE with align=False (global + pose error) in mm
            - mpjpe_aligned: MPJPE with align=True (pose error only) in mm
            - pa_mpjpe: PA-MPJPE in mm
            - pck: Dict of PCK@threshold values (computed on hip-aligned predictions)
            - per_joint_mpjpe: Per-joint MPJPE [J] (align=False)
            - per_joint_mpjpe_aligned: Per-joint MPJPE [J] (align=True)
            - per_joint_pa_mpjpe: Per-joint PA-MPJPE [J]
            - body_part_mpjpe: Per body part MPJPE (align=False)
            - num_samples: Total number of samples
            - num_joints: Number of joints (14 or 17)
        """
        if len(self.predictions) == 0:
            raise ValueError("No data accumulated. Call update() first.")

        pred = np.concatenate(self.predictions, axis=0)
        gt = np.concatenate(self.ground_truths, axis=0)

        num_joints = self.num_joints if self.num_joints is not None else pred.shape[1]

        results = {
            'num_samples': len(pred),
            'num_joints': num_joints,
            'mpjpe': compute_mpjpe(pred, gt, align=False, num_joints=num_joints),
            'mpjpe_aligned': compute_mpjpe(pred, gt, align=True, num_joints=num_joints),
            'pa_mpjpe': compute_pa_mpjpe(pred, gt),
            'pck': compute_pck(pred, gt, self.pck_thresholds, num_joints=num_joints),
        }

        _, per_joint_mpjpe = compute_mpjpe(pred, gt, align=False, return_per_joint=True, num_joints=num_joints)
        _, per_joint_mpjpe_aligned = compute_mpjpe(pred, gt, align=True, return_per_joint=True, num_joints=num_joints)
        _, per_joint_pa_mpjpe = compute_pa_mpjpe(pred, gt, return_per_joint=True)

        results['per_joint_mpjpe'] = per_joint_mpjpe
        results['per_joint_mpjpe_aligned'] = per_joint_mpjpe_aligned
        results['per_joint_pa_mpjpe'] = per_joint_pa_mpjpe
        results['body_part_mpjpe'] = compute_body_part_error(pred, gt, num_joints=num_joints)

        return results

def format_results_table(results: Dict, method_name: str = "Ours") -> str:
    """
    Format evaluation results as a table string.

    Args:
        results: Dictionary from PoseMetrics.compute()
        method_name: Name of the method

    Returns:
        Formatted table string
    """
    lines = []

    num_joints = results.get('num_joints', 17)
    joint_names = JOINT_NAMES_14 if num_joints == 14 else JOINT_NAMES_17

    lines.append("=" * 70)
    lines.append(f"Evaluation Results: {method_name}")
    lines.append("=" * 70)
    lines.append(f"Total samples: {results['num_samples']}")
    lines.append(f"Joint format: {num_joints}-joint")

    lines.append("\n--- Overall Metrics (GraphPose-Fi Convention) ---")
    lines.append(f"MPJPE (align=False):    {results['mpjpe']:.2f} mm  <- Use this for comparison")
    if 'mpjpe_aligned' in results:
        lines.append(f"MPJPE (align=True):     {results['mpjpe_aligned']:.2f} mm  (hip-aligned, pose error only)")
    lines.append(f"PA-MPJPE:               {results['pa_mpjpe']:.2f} mm")

    lines.append("\n--- PCK (%) ---")
    pck_str = "  ".join([f"{k}: {v:.1f}" for k, v in results['pck'].items()])
    lines.append(pck_str)

    lines.append("\n--- Per-Joint MPJPE (mm) ---")
    lines.append("-" * 50)
    lines.append(f"{'Joint':<15} {'MPJPE':>10} {'PA-MPJPE':>10}")
    lines.append("-" * 50)

    for i, name in enumerate(joint_names):
        if i < len(results['per_joint_mpjpe']):
            mpjpe = results['per_joint_mpjpe'][i]
            pa_mpjpe = results['per_joint_pa_mpjpe'][i]
            lines.append(f"{name:<15} {mpjpe:>10.2f} {pa_mpjpe:>10.2f}")

    lines.append("-" * 50)

    lines.append("\n--- Body Part MPJPE (mm) ---")
    for part_name, error in results['body_part_mpjpe'].items():
        lines.append(f"  {part_name:<10}: {error:.2f}")

    lines.append("=" * 70)

    return "\n".join(lines)

def format_comparison_table(methods_results: Dict[str, Dict],
                            metric: str = 'per_joint_mpjpe') -> str:
    """
    Format per-joint comparison table for multiple methods.

    Similar to Table 2 in papers comparing methods per joint.

    Args:
        methods_results: Dict[method_name, results_dict]
        metric: 'per_joint_mpjpe' or 'per_joint_pa_mpjpe'

    Returns:
        Formatted comparison table string
    """
    methods = list(methods_results.keys())
    n_methods = len(methods)

    lines = []
    lines.append("=" * (20 + 12 * n_methods))
    lines.append("Table: Per-joint MPJPE (mm) comparison")
    lines.append("=" * (20 + 12 * n_methods))

    header = f"{'Joint':<15}"
    for method in methods:
        header += f" {method:>10}"
    lines.append(header)
    lines.append("-" * (20 + 12 * n_methods))

    for i, name in enumerate(JOINT_NAMES):
        row = f"{name:<15}"
        values = []
        for method in methods:
            val = methods_results[method][metric][i]
            values.append(val)
            row += f" {val:>10.2f}"

        min_val = min(values)
        if values.count(min_val) == 1:
            row_parts = row.split()
            best_idx = values.index(min_val) + 1
            row_parts[best_idx] = f"*{row_parts[best_idx]}*"
            row = " ".join(row_parts)

        lines.append(row)

    lines.append("-" * (20 + 12 * n_methods))

    avg_row = f"{'Average':<15}"
    for method in methods:
        avg = np.mean(methods_results[method][metric])
        avg_row += f" {avg:>10.2f}"
    lines.append(avg_row)

    lines.append("=" * (20 + 12 * n_methods))

    return "\n".join(lines)

def format_latex_table(results: Dict, method_name: str = "Ours") -> str:
    """
    Format results as LaTeX table row for paper.

    Args:
        results: Dictionary from PoseMetrics.compute()
        method_name: Name of the method

    Returns:
        LaTeX table row string
    """
    pck = results['pck']
    row = f"{method_name} & "
    row += " & ".join([f"{pck[f'PCK@{t}']:.1f}" for t in [10, 20, 30, 40, 50]])
    row += f" & {results['mpjpe']:.2f} & {results['pa_mpjpe']:.2f} \\\\"

    return row

def format_per_joint_latex(methods_results: Dict[str, Dict]) -> str:
    """
    Format per-joint MPJPE as LaTeX table for paper.

    Args:
        methods_results: Dict[method_name, results_dict]

    Returns:
        LaTeX table string
    """
    methods = list(methods_results.keys())

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Per-joint MPJPE (mm) $\\downarrow$ comparisons on MM-Fi}")
    lines.append("\\label{tab:per_joint}")
    lines.append("\\resizebox{\\columnwidth}{!}{%")

    cols = "l" + "c" * len(methods)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")

    header = "Joint & " + " & ".join(methods) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for i, name in enumerate(JOINT_NAMES):
        values = [methods_results[m]['per_joint_mpjpe'][i] for m in methods]
        min_val = min(values)

        row_parts = []
        for val in values:
            if val == min_val:
                row_parts.append(f"\\textbf{{{val:.2f}}}")
            else:
                row_parts.append(f"{val:.2f}")

        row = f"{name} & " + " & ".join(row_parts) + " \\\\"
        lines.append(row)

    lines.append("\\midrule")

    avg_values = [np.mean(methods_results[m]['per_joint_mpjpe']) for m in methods]
    min_avg = min(avg_values)
    avg_parts = []
    for val in avg_values:
        if val == min_avg:
            avg_parts.append(f"\\textbf{{{val:.2f}}}")
        else:
            avg_parts.append(f"{val:.2f}")

    lines.append(f"Average & " + " & ".join(avg_parts) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table}")

    return "\n".join(lines)

if __name__ == '__main__':
    print("Testing pose metrics...")

    B, J = 100, 17
    pred = np.random.randn(B, J, 3) * 0.1
    gt = pred + np.random.randn(B, J, 3) * 0.03

    print(f"\nMPJPE: {compute_mpjpe(pred, gt):.2f} mm")
    print(f"PA-MPJPE: {compute_pa_mpjpe(pred, gt):.2f} mm")

    pck = compute_pck(pred, gt)
    print(f"PCK: {pck}")

    metrics = PoseMetrics()
    metrics.update(pred[:50], gt[:50])
    metrics.update(pred[50:], gt[50:])
    results = metrics.compute()

    print("\n" + format_results_table(results, "Test Method"))

    print("\nLaTeX row:")
    print(format_latex_table(results, "Ours"))
