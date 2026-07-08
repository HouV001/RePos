"""
Loss Functions for CSI-StructPose

Includes:
1. Pose Loss (L1 + L2)
2. Consistency Loss (Dual-path agreement)
3. Limb Length Loss (Anatomical constraint)
4. Chamfer Loss (mmWave point cloud supervision)
5. Part Separation Loss (Optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

def chamfer_loss(pred_pc: torch.Tensor, gt_pc: torch.Tensor) -> torch.Tensor:
    """
    Compute Chamfer Distance between predicted and ground truth point clouds.

    Chamfer Distance measures bidirectional point cloud similarity:
    - Forward: min distance from each predicted point to ground truth
    - Backward: min distance from each ground truth point to predicted

    Args:
        pred_pc: Predicted point cloud [B, N, C] or [B, T, N, C]
        gt_pc: Ground truth point cloud [B, M, C] or [B, T, M, C]

    Returns:
        Chamfer distance (scalar)

    Note:
        If pred_pc is [B, N, C] and gt_pc is [B, T, M, C], we use only the
        last frame of gt_pc for supervision (most aligned with current pose).
    """
    if pred_pc.dim() == 3 and gt_pc.dim() == 4:
        gt_pc = gt_pc[:, -1, :, :]

    if pred_pc.dim() == 4:
        B, T, N, C = pred_pc.shape
        pred_pc = pred_pc.reshape(B * T, N, C)
        gt_pc = gt_pc.reshape(B * T, -1, gt_pc.shape[-1])

    pred_xyz = pred_pc[..., :3]
    gt_xyz = gt_pc[..., :3]

    gt_valid = (gt_xyz.abs().sum(dim=-1) > 1e-6)

    batch_has_valid_gt = gt_valid.any(dim=1)

    if not batch_has_valid_gt.any():
        return torch.tensor(0.0, device=pred_pc.device, requires_grad=True)

    valid_batch_indices = batch_has_valid_gt.nonzero(as_tuple=True)[0]

    pred_xyz_valid = pred_xyz[valid_batch_indices]
    gt_xyz_valid = gt_xyz[valid_batch_indices]
    gt_valid_mask = gt_valid[valid_batch_indices]

    diff = pred_xyz_valid.unsqueeze(2) - gt_xyz_valid.unsqueeze(1)
    dist = (diff ** 2).sum(dim=-1)

    dist_for_forward = dist.masked_fill(~gt_valid_mask.unsqueeze(1), float('inf'))
    dist_pred_to_gt = dist_for_forward.min(dim=2)[0]

    dist_pred_to_gt = torch.where(torch.isinf(dist_pred_to_gt),
                                   torch.zeros_like(dist_pred_to_gt),
                                   dist_pred_to_gt)
    loss_forward = dist_pred_to_gt.mean()

    dist_gt_to_pred = dist.min(dim=1)[0]

    loss_backward = dist_gt_to_pred[gt_valid_mask].mean()

    return loss_forward + loss_backward

class LimbLengthLoss(nn.Module):
    """
    Limb length consistency loss.

    Ensures predicted limb lengths match ground truth,
    enforcing anatomical plausibility.

    Supports both MM-Fi 17-joint and WiFiPose3D 14-joint formats.
    """

    LIMB_PAIRS_17 = [
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 4),
        (4, 5),
        (5, 6),
        (0, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (8, 11),
        (11, 12),
        (12, 13),
        (8, 14),
        (14, 15),
        (15, 16),
    ]

    LIMB_PAIRS_14 = [
        (0, 1),
        (0, 2),
        (0, 3),
        (2, 4),
        (4, 8),
        (3, 6),
        (6, 10),
        (2, 5),
        (3, 7),
        (5, 7),
        (5, 9),
        (9, 12),
        (7, 11),
        (11, 13),
    ]

    LIMB_PAIRS_18 = [
        (4, 3), (3, 2), (7, 6), (6, 5), (10, 9), (9, 8), (13, 12), (12, 11),
        (8, 2), (11, 5), (2, 1), (5, 1), (0, 1), (14, 0), (15, 0), (16, 14), (17, 15),
    ]

    LIMB_PAIRS = LIMB_PAIRS_17

    def __init__(self, num_joints: int = 17):
        super().__init__()
        self.num_joints = num_joints
        self.limb_pairs = {14: self.LIMB_PAIRS_14, 17: self.LIMB_PAIRS_17,
                           18: self.LIMB_PAIRS_18}[num_joints]

    def compute_limb_lengths(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Compute all limb lengths from pose.

        Args:
            pose: [B, J, 3] where J is num_joints (14 or 17)
        Returns:
            limb_lengths: [B, num_limbs] (14 for 14-joint, 16 for 17-joint)
        """
        limb_lengths = []
        for (i, j) in self.limb_pairs:
            length = torch.norm(pose[:, i] - pose[:, j], dim=-1)
            limb_lengths.append(length)
        return torch.stack(limb_lengths, dim=1)

    def forward(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_pose: [B, J, 3] predicted pose (J=14 or 17)
            gt_pose: [B, J, 3] ground truth pose

        Returns:
            loss: scalar limb length loss
        """
        pred_lengths = self.compute_limb_lengths(pred_pose)
        gt_lengths = self.compute_limb_lengths(gt_pose)

        loss = F.l1_loss(pred_lengths, gt_lengths)

        return loss

class BoneDirectionLoss(nn.Module):
    """
    Bone-direction (articulation) loss.

    Matches the UNIT direction of each bone, not just its length. The mean pose
    already satisfies joint-L2 and limb LENGTHS, so neither term resists
    collapse-to-mean; matching bone DIRECTIONS penalizes wrong articulation
    directly and pushes the structure branch off the mean pose. Defined on the
    same skeleton edges as LimbLengthLoss (14/17/18-joint).
    """
    def __init__(self, num_joints: int = 17, eps: float = 1e-6):
        super().__init__()
        self.num_joints = num_joints
        self.limb_pairs = {14: LimbLengthLoss.LIMB_PAIRS_14,
                           17: LimbLengthLoss.LIMB_PAIRS_17,
                           18: LimbLengthLoss.LIMB_PAIRS_18}[num_joints]
        self.eps = eps

    def _unit_bones(self, pose: torch.Tensor) -> torch.Tensor:
        bones = []
        for (i, j) in self.limb_pairs:
            v = pose[:, j] - pose[:, i]
            v = v / (v.norm(dim=-1, keepdim=True) + self.eps)
            bones.append(v)
        return torch.stack(bones, dim=1)

    def forward(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
        """1 - mean cosine similarity between predicted and GT unit bone vectors."""
        pb = self._unit_bones(pred_pose)
        gb = self._unit_bones(gt_pose)
        cos = (pb * gb).sum(dim=-1)
        return (1.0 - cos).mean()

class ConsistencyLoss(nn.Module):
    """
    Dual-path consistency loss.

    Encourages agreement between Direct Path and Point Cloud Path predictions.
    Uses stop-gradient to prevent mode collapse.
    """

    def __init__(self, symmetric: bool = True):
        super().__init__()
        self.symmetric = symmetric

    def forward(self,
                pose_direct: torch.Tensor,
                pose_pc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_direct: [B, 17, 3] from Direct Path
            pose_pc: [B, 17, 3] from Point Cloud Path

        Returns:
            loss: scalar consistency loss
        """
        if self.symmetric:
            loss = F.mse_loss(pose_direct, pose_pc.detach()) + \
                   F.mse_loss(pose_pc, pose_direct.detach())
            loss = loss / 2
        else:
            loss = F.mse_loss(pose_pc, pose_direct.detach())

        return loss

class PartSeparationLoss(nn.Module):
    """
    Optional: Encourage points from different parts to be spatially separated.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self,
                point_cloud: torch.Tensor,
                part_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_cloud: [B, N, 5]
            part_labels: [B, N] part index (0-5)

        Returns:
            loss: scalar separation loss
        """
        B, N, _ = point_cloud.shape
        xyz = point_cloud[..., :3]

        loss = 0
        count = 0

        for b in range(B):
            unique_parts = torch.unique(part_labels[b])
            for i in range(len(unique_parts)):
                for j in range(i + 1, len(unique_parts)):
                    part_i = unique_parts[i]
                    part_j = unique_parts[j]

                    mask_i = part_labels[b] == part_i
                    mask_j = part_labels[b] == part_j

                    center_i = xyz[b, mask_i].mean(dim=0)
                    center_j = xyz[b, mask_j].mean(dim=0)

                    dist = torch.norm(center_i - center_j)

                    loss += F.relu(self.margin - dist)
                    count += 1

        return loss / max(count, 1)

class CSIStructPoseLoss(nn.Module):
    """
    Combined loss function for CSI-StructPose.

    Total Loss = pose_loss + λ1*consistency_loss + λ2*limb_loss + λ3*chamfer_loss
    """

    def __init__(self,
                 consistency_weight: float = 0.5,
                 limb_weight: float = 0.1,
                 chamfer_weight: float = 0.0,
                 use_l1: bool = True,
                 use_l2: bool = True):
        super().__init__()

        self.consistency_weight = consistency_weight
        self.limb_weight = limb_weight
        self.chamfer_weight = chamfer_weight
        self.use_l1 = use_l1
        self.use_l2 = use_l2

        self.consistency_loss = ConsistencyLoss(symmetric=True)
        self.limb_loss = LimbLengthLoss()

    def forward(self,
                pose_direct: torch.Tensor,
                pose_pc: torch.Tensor,
                pose_gt: torch.Tensor,
                virtual_pc: Optional[torch.Tensor] = None,
                mmwave_gt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pose_direct: [B, 17, 3] Direct Path prediction
            pose_pc: [B, 17, 3] Point Cloud Path prediction
            pose_gt: [B, 17, 3] Ground truth
            virtual_pc: [B, N, C] or [B, T, N, C] Generated point cloud (optional)
            mmwave_gt: [B, M, C] or [B, T, M, C] Ground truth mmWave points (optional)

        Returns:
            total_loss: scalar total loss
            loss_dict: dictionary of individual losses
        """
        if self.use_l1 and self.use_l2:
            loss_direct = F.l1_loss(pose_direct, pose_gt) + F.mse_loss(pose_direct, pose_gt)
            loss_pc = F.l1_loss(pose_pc, pose_gt) + F.mse_loss(pose_pc, pose_gt)
        elif self.use_l1:
            loss_direct = F.l1_loss(pose_direct, pose_gt)
            loss_pc = F.l1_loss(pose_pc, pose_gt)
        else:
            loss_direct = F.mse_loss(pose_direct, pose_gt)
            loss_pc = F.mse_loss(pose_pc, pose_gt)

        pose_loss = loss_direct + loss_pc

        consistency = self.consistency_loss(pose_direct, pose_pc)

        limb = self.limb_loss(pose_pc, pose_gt)

        total_loss = pose_loss + \
                     self.consistency_weight * consistency + \
                     self.limb_weight * limb

        loss_dict = {
            'total': total_loss.item(),
            'pose_direct': loss_direct.item(),
            'pose_pc': loss_pc.item(),
            'consistency': consistency.item(),
            'limb': limb.item()
        }

        if virtual_pc is not None and mmwave_gt is not None and self.chamfer_weight > 0:
            loss_chamfer = chamfer_loss(virtual_pc, mmwave_gt)
            total_loss = total_loss + self.chamfer_weight * loss_chamfer
            loss_dict['chamfer'] = loss_chamfer.item()
            loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

def compute_mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute Mean Per Joint Position Error.

    Args:
        pred: [B, 17, 3]
        gt: [B, 17, 3]

    Returns:
        mpjpe: scalar (in same units as input, typically meters)
    """
    error = torch.norm(pred - gt, dim=-1)
    mpjpe = error.mean()
    return mpjpe

LIMB_PAIRS_14 = LimbLengthLoss.LIMB_PAIRS_14
LIMB_PAIRS_17 = LimbLengthLoss.LIMB_PAIRS_17

def compute_pampjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Compute Procrustes-Aligned MPJPE (PA-MPJPE).

    Aligns each prediction to its ground truth via SVD-based Procrustes analysis
    (rotation, translation, scale) before computing MPJPE.

    Args:
        pred: [B, J, 3] predicted poses
        gt: [B, J, 3] ground truth poses

    Returns:
        pa_mpjpe: scalar mean error after alignment
    """
    B = pred.shape[0]
    total_error = 0.0

    for i in range(B):
        p = pred[i]
        g = gt[i]

        p_mean = p.mean(dim=0, keepdim=True)
        g_mean = g.mean(dim=0, keepdim=True)
        p_centered = p - p_mean
        g_centered = g - g_mean

        p_scale = torch.sqrt((p_centered ** 2).sum())
        g_scale = torch.sqrt((g_centered ** 2).sum())

        p_normalized = p_centered / (p_scale + 1e-8)
        g_normalized = g_centered / (g_scale + 1e-8)

        H = g_normalized.T @ p_normalized
        U, S, Vh = torch.linalg.svd(H)

        R = Vh.T @ U.T

        if torch.det(R) < 0:
            Vh_fixed = Vh.clone()
            Vh_fixed[-1, :] *= -1
            R = Vh_fixed.T @ U.T

        aligned = g_scale * (p_normalized @ R) + g_mean

        error = torch.norm(aligned - g, dim=-1)
        total_error = total_error + error.mean()

    pa_mpjpe = total_error / B
    return pa_mpjpe

class HipAlignedPoseLoss(nn.Module):
    """
    Hip-aligned pose loss for cross-scene absolute coordinate training.

    Aligns predicted and GT poses at the hip/pelvis before computing loss.
    This allows the model to focus on learning pose structure rather than
    absolute global position, which varies across scenes.

    The loss decomposes into:
    - Aligned pose loss: L1+L2 loss after hip alignment (focuses on pose structure)
    - Root position loss: Separate loss on hip position (optional)
    """

    HIP_INDEX_17 = 0
    HIP_INDEX_14 = (5, 7)

    def __init__(self,
                 num_joints: int = 17,
                 root_weight: float = 0.1,
                 use_l1: bool = True,
                 use_l2: bool = True):
        """
        Args:
            num_joints: Number of joints (14 or 17)
            root_weight: Weight for root position loss (0 to disable)
            use_l1: Use L1 loss
            use_l2: Use L2 (MSE) loss
        """
        super().__init__()
        self.num_joints = num_joints
        self.root_weight = root_weight
        self.use_l1 = use_l1
        self.use_l2 = use_l2

    def get_hip_position(self, pose: torch.Tensor) -> torch.Tensor:
        """Get hip/root position from pose."""
        if self.num_joints == 14:
            return (pose[:, 5, :] + pose[:, 7, :]) / 2
        else:
            return pose[:, 0, :]

    def forward(self,
                pred_pose: torch.Tensor,
                gt_pose: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_pose: [B, J, 3] predicted pose
            gt_pose: [B, J, 3] ground truth pose

        Returns:
            total_loss: scalar
            loss_dict: dictionary with loss components
        """
        pred_hip = self.get_hip_position(pred_pose)
        gt_hip = self.get_hip_position(gt_pose)

        offset = gt_hip - pred_hip
        pred_aligned = pred_pose + offset.unsqueeze(1)

        aligned_loss = torch.tensor(0.0, device=pred_pose.device)
        if self.use_l1:
            aligned_loss = aligned_loss + F.l1_loss(pred_aligned, gt_pose)
        if self.use_l2:
            aligned_loss = aligned_loss + F.mse_loss(pred_aligned, gt_pose)

        root_loss = F.l1_loss(pred_hip, gt_hip) + F.mse_loss(pred_hip, gt_hip)

        total_loss = aligned_loss + self.root_weight * root_loss

        loss_dict = {
            'aligned': aligned_loss.item(),
            'root': root_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict

class VelocityLoss(nn.Module):
    """
    Velocity consistency loss for temporal smoothness.

    Encourages consistent motion by penalizing large frame-to-frame velocity changes.
    Useful when processing sequences of poses.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self,
                pred_poses: torch.Tensor,
                gt_poses: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_poses: [B, T, J, 3] predicted poses over time
            gt_poses: [B, T, J, 3] ground truth poses (optional)

        Returns:
            loss: scalar velocity loss
        """
        if pred_poses.dim() != 4:
            return torch.tensor(0.0, device=pred_poses.device)

        pred_vel = pred_poses[:, 1:] - pred_poses[:, :-1]

        if gt_poses is not None:
            gt_vel = gt_poses[:, 1:] - gt_poses[:, :-1]
            loss = F.l1_loss(pred_vel, gt_vel)
        else:
            loss = pred_vel.abs().mean()

        return self.weight * loss

def random_translation_augmentation(
    pose: torch.Tensor,
    max_offset: float = 0.5,
    p: float = 0.5
) -> torch.Tensor:
    """
    Apply random translation augmentation to poses.

    This helps the model learn relative pose structure rather than
    memorizing specific global positions from training data.

    Args:
        pose: [B, J, 3] pose tensor
        max_offset: Maximum translation offset in meters
        p: Probability of applying augmentation

    Returns:
        augmented_pose: [B, J, 3] (same reference, modified in-place)
    """
    if torch.rand(1).item() > p:
        return pose

    B = pose.shape[0]
    device = pose.device

    offset = (torch.rand(B, 1, 3, device=device) - 0.5) * 2 * max_offset

    return pose + offset

def random_rotation_augmentation(
    pose: torch.Tensor,
    max_angle: float = 30.0,
    p: float = 0.5
) -> torch.Tensor:
    """
    Apply random rotation augmentation around vertical (Y) axis.

    Args:
        pose: [B, J, 3] pose tensor
        max_angle: Maximum rotation angle in degrees
        p: Probability of applying augmentation

    Returns:
        rotated_pose: [B, J, 3]
    """
    if torch.rand(1).item() > p:
        return pose

    B = pose.shape[0]
    device = pose.device

    angles = (torch.rand(B, device=device) - 0.5) * 2 * (max_angle * 3.14159 / 180)

    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    R = torch.zeros(B, 3, 3, device=device)
    R[:, 0, 0] = cos_a
    R[:, 0, 2] = sin_a
    R[:, 1, 1] = 1
    R[:, 2, 0] = -sin_a
    R[:, 2, 2] = cos_a

    hip = pose[:, 0:1, :]
    centered = pose - hip

    rotated = torch.bmm(centered, R.transpose(1, 2))

    return rotated + hip
