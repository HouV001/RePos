"""
CSI-StructPose V2 Training Script

Two-phase architecture:
- Phase 1: Train relative pose (root-relative coordinates) - learns pose structure
- Phase 2: Train absolute pose with learned phase position predictor

Usage:
    # Phase 1: Train relative pose
    python train_v2.py --phase 1 --data_root ./data/cross_scene --epochs 100

    # Phase 2: Position prediction (requires Phase 1 checkpoint)
    python train_v2.py --phase 2 --data_root ./data/cross_scene --phase1_ckpt pretrained/v2_best.pth --epochs 50

    # Phase 2: With joint training (finetune Phase 1)
    python train_v2.py --phase 2 --data_root ./data/cross_scene --phase1_ckpt pretrained/v2_best.pth --finetune --epochs 50
"""

import os
import argparse
import math
from datetime import datetime
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.csi_structpose_v2 import (
    build_csi_structpose_v2,
    build_csi_structpose_v2_phase2,
    load_phase1_into_phase2,
)
from models.losses import LimbLengthLoss, BoneDirectionLoss
from utils.metrics import compute_mpjpe, compute_pa_mpjpe, compute_pck

def compute_global_position_mean(data_root):
    """Global mean root position over the training set (all 3 axes).
    Loads whichever pose array the dataset wrote, and uses the SAME root
    definition as training: pelvis (joint 0) for 17 joints (MM-Fi), hip midpoint
    (joints 5,7) for the 14-joint Person-in-WiFi-3D TRUE order."""
    poses = None
    for fn in ('joint_gt_array_train.npy', 'pose_array_train.npy'):
        p = os.path.join(data_root, fn)
        if os.path.exists(p):
            poses = np.load(p)
            break
    if poses is None:
        raise FileNotFoundError(
            f"No train pose array in {data_root} "
            f"(looked for joint_gt_array_train.npy / pose_array_train.npy)")
    nj = poses.shape[1]
    root = poses[:, 0, :] if nj == 17 else (poses[:, 5, :] + poses[:, 7, :]) / 2.0
    return np.mean(root, axis=0)

def compute_global_position_mean_loeo(per_env_root, test_env):
    """Global mean root over LOEO TRAIN envs (all per-env dirs except test_env).
    MM-Fi 17-joint: root = pelvis = joint 0. Used for --absolute --env_norm + --loeo_test_env."""
    envs = sorted(d for d in os.listdir(per_env_root)
                  if os.path.isdir(os.path.join(per_env_root, d))
                  and d != test_env and d.startswith('E'))
    roots = []
    for e in envs:
        pose = np.load(os.path.join(per_env_root, e, 'joint_gt_array.npy'), mmap_mode='r')
        roots.append(np.asarray(pose[:, 0, :]))
    root = np.concatenate(roots, axis=0)
    return np.mean(root, axis=0), envs

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """
    Warmup + Cosine Annealing scheduler (like GraphPose-Fi/DT-Pose).

    - First `warmup_epochs`: Linear warmup from 0 to target LR
    - Remaining epochs: Cosine annealing decay to ~0

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
    """
    def lr_lambda(cur_epoch):
        if cur_epoch < warmup_epochs:
            return (cur_epoch + 1) / warmup_epochs
        progress = (cur_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_phase1_epoch(model, dataloader, optimizer, device, epoch, n_epochs,
                       limb_loss_fn, limb_weight=0.1, simple_loss=False,
                       env_norm=False, global_mean=None,
                       bone_loss_fn=None, bone_weight=0.0, freeze_encoder=False):
    """Train one epoch for Phase 1 (relative pose, or absolute when env_norm).

    Args:
        simple_loss: If True, use only MPJPE loss like GraphPose-Fi (L2 norm).
                     If False, use L1+MSE+LimbLength loss.
        env_norm: If True (with absolute data), center the absolute pose by the
                  global training mean so the model predicts a centered-absolute
                  pose (feasible regression). Metrics are denormalized.
        global_mean: [3] tensor of the global position mean from training data.
    """
    model.train()
    if freeze_encoder:
        model.csi_encoder.eval()

    loss_meter = AverageMeter()
    mpjpe_meter = AverageMeter()
    pampjpe_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f"Phase1 Epoch {epoch}/{n_epochs}")

    for batch in pbar:
        if len(batch) == 3:
            csi, pose_gt, _ = batch
        else:
            csi, pose_gt = batch[:2]

        csi = csi.to(device)
        pose_gt = pose_gt.to(device)
        B = csi.size(0)

        if env_norm and global_mean is not None:
            pose_gt_target = pose_gt - global_mean.view(1, 1, 3)
        else:
            pose_gt_target = pose_gt

        optimizer.zero_grad()

        output = model(csi)
        pose_pred = output['pose']

        if simple_loss:
            loss = torch.mean(torch.norm(pose_pred - pose_gt_target, dim=-1))
        else:
            pose_loss = F.l1_loss(pose_pred, pose_gt_target) + F.mse_loss(pose_pred, pose_gt_target)
            limb_loss = limb_loss_fn(pose_pred, pose_gt_target)
            loss = pose_loss + limb_weight * limb_loss
            if bone_loss_fn is not None and bone_weight > 0:
                loss = loss + bone_weight * bone_loss_fn(pose_pred, pose_gt_target)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  WARNING: NaN/Inf loss, skipping batch")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            if env_norm and global_mean is not None:
                pose_pred_eval = pose_pred + global_mean.view(1, 1, 3)
            else:
                pose_pred_eval = pose_pred
            mpjpe = compute_mpjpe(pose_pred_eval, pose_gt, align=False)
            pampjpe = compute_pa_mpjpe(pose_pred_eval.cpu(), pose_gt.cpu())

        loss_meter.update(loss.item(), B)
        mpjpe_meter.update(mpjpe.item(), B)
        pampjpe_meter.update(pampjpe.item(), B)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", mpjpe=f"{mpjpe_meter.avg:.1f}", pa=f"{pampjpe_meter.avg:.1f}")

    return {'loss': loss_meter.avg, 'mpjpe': mpjpe_meter.avg, 'pa_mpjpe': pampjpe_meter.avg}

@torch.no_grad()
def evaluate(model, dataloader, device, verbose=False, env_norm=False, global_mean=None):
    """Evaluate model."""
    model.eval()

    total_mpjpe = 0
    total_pampjpe = 0
    total_pck_50 = 0
    total_samples = 0

    has_phase2 = False
    total_phase1_mpjpe = 0
    total_phase2_mpjpe = 0
    total_root_error = 0

    for batch in dataloader:
        if len(batch) == 3:
            csi, pose_gt, _ = batch
        else:
            csi, pose_gt = batch[:2]

        csi = csi.to(device)
        pose_gt = pose_gt.to(device)
        B = csi.size(0)

        output = model(csi)
        pose_pred = output['pose']

        if env_norm and global_mean is not None:
            pose_pred_denorm = pose_pred + global_mean.view(1, 1, 3)
        else:
            pose_pred_denorm = pose_pred

        mpjpe = compute_mpjpe(pose_pred_denorm, pose_gt, align=False)
        pampjpe = compute_pa_mpjpe(pose_pred_denorm.cpu(), pose_gt.cpu())
        pck_50 = compute_pck(pose_pred_denorm, pose_gt, thresholds=[0.5])['PCK@50']

        if 'phase1_pose' in output and 'phase2_pose' in output:
            has_phase2 = True
            phase1_pose = output['phase1_pose']
            phase2_pose = output['phase2_pose']
            root_position = output['root_position']

            if env_norm and global_mean is not None:
                phase2_pose_denorm = phase2_pose + global_mean.view(1, 1, 3)
                root_position_denorm = root_position + global_mean
            else:
                phase2_pose_denorm = phase2_pose
                root_position_denorm = root_position

            if verbose and total_samples == 0:
                pose_diff = (pose_pred_denorm - phase2_pose_denorm).abs().mean().item() * 1000
                print(f"  [Debug] |pose - phase2_pose| = {pose_diff:.2f}mm (should be > 0 if fusion works)")
                if hasattr(model, 'training_mode'):
                    print(f"  [Debug] model.training_mode = {model.training_mode} (should be False in eval)")
                print(f"  [Debug] phase1_pose[0,0] = {phase1_pose[0,0].tolist()} (should be ~[0,0,0])")
                print(f"  [Debug] phase2_pose[0,0] = {phase2_pose_denorm[0,0].tolist()} (predicted root)")
                print(f"  [Debug] root_position[0] = {root_position_denorm[0].tolist()}")
                print(f"  [Debug] pose_pred[0,0] = {pose_pred_denorm[0,0].tolist()} (should = phase1[0,0] + root)")
                print(f"  [Debug] GT pose[0,0] = {pose_gt[0,0].tolist()} (ground truth root)")

            gt_root = pose_gt[:, 0, :]
            gt_relative = pose_gt - gt_root.unsqueeze(1)

            phase1_mpjpe = compute_mpjpe(phase1_pose, gt_relative, align=False)
            total_phase1_mpjpe += phase1_mpjpe.item() * B

            phase2_mpjpe = compute_mpjpe(phase2_pose_denorm, pose_gt, align=False)
            total_phase2_mpjpe += phase2_mpjpe.item() * B

            root_error = torch.norm(root_position_denorm - gt_root, dim=-1).mean() * 1000
            total_root_error += root_error.item() * B

        total_mpjpe += mpjpe.item() * B
        total_pampjpe += pampjpe.item() * B
        total_pck_50 += pck_50.item() * B
        total_samples += B

    result = {
        'mpjpe': total_mpjpe / total_samples,
        'pa_mpjpe': total_pampjpe / total_samples,
        'pck@50': total_pck_50 / total_samples,
    }

    if has_phase2:
        result['phase1_mpjpe'] = total_phase1_mpjpe / total_samples
        result['phase2_mpjpe'] = total_phase2_mpjpe / total_samples
        result['root_error'] = total_root_error / total_samples

        if verbose:
            print(f"  [Debug] Phase1 relative MPJPE: {result['phase1_mpjpe']:.1f}mm")
            print(f"  [Debug] Phase2 absolute MPJPE: {result['phase2_mpjpe']:.1f}mm")
            print(f"  [Debug] Root position error: {result['root_error']:.1f}mm")
            print(f"  [Debug] Fused MPJPE (output['pose']): {result['mpjpe']:.1f}mm")
            if hasattr(model, 'training_mode'):
                print(f"  [Debug] model.training_mode = {model.training_mode} (should be False)")
            if 'pose_equals_phase2' not in result:
                pass

    return result

def train_phase1(model, train_loader, test_loader, cfg, device, save_dir,
                 env_norm=False, global_mean=None):
    """Phase 1 training loop (env_norm enables absolute-pose prediction)."""
    n_epochs = cfg['epochs']
    lr = cfg['lr']
    weight_decay = cfg.get('weight_decay', 0.02)
    limb_weight = cfg.get('limb_weight', 0.1)
    bone_weight = cfg.get('bone_weight', 0.0)
    num_joints = cfg.get('num_joints', 17)
    warmup_epochs = cfg.get('warmup_epochs', 5)
    simple_loss = cfg.get('simple_loss', False)

    limb_loss_fn = LimbLengthLoss(num_joints=num_joints).to(device)
    bone_loss_fn = BoneDirectionLoss(num_joints=num_joints).to(device)

    loss_str = ("SIMPLE (MPJPE only)" if simple_loss else
                f"MULTI-TERM (L1+MSE+Limb(w={limb_weight})"
                f"{f'+Bone(w={bone_weight})' if bone_weight > 0 else ''})")

    print("=" * 70)
    print("CSI-StructPose V2 Training - PHASE 1 (Relative Pose)")
    print("=" * 70)
    print(f"  Epochs: {n_epochs}, LR: {lr}, WD: {weight_decay}, Warmup: {warmup_epochs}")
    print(f"  Loss: {loss_str}")
    print(f"  Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    print("=" * 70)

    freeze_encoder = cfg.get('freeze_encoder', False)
    if freeze_encoder:
        for p in model.csi_encoder.parameters():
            p.requires_grad_(False)
        n_frozen = sum(p.numel() for p in model.csi_encoder.parameters())
        print(f"  Encoder FROZEN ({n_frozen/1e6:.2f}M params); training SGA+decoder head only "
              f"(DT-Pose regime, preserves SSL features).")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, n_epochs)

    best_mpjpe = float('inf')
    best_pampjpe = float('inf')
    best_mpjpe_epoch = 0
    best_pampjpe_epoch = 0

    for epoch in range(1, n_epochs + 1):
        train_metrics = train_phase1_epoch(
            model, train_loader, optimizer, device, epoch, n_epochs,
            limb_loss_fn=limb_loss_fn, limb_weight=limb_weight,
            simple_loss=simple_loss,
            env_norm=env_norm, global_mean=global_mean,
            bone_loss_fn=bone_loss_fn, bone_weight=bone_weight,
            freeze_encoder=freeze_encoder
        )
        scheduler.step()

        test_metrics = evaluate(model, test_loader, device,
                                env_norm=env_norm, global_mean=global_mean)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{n_epochs} (lr={current_lr:.2e})")
        print(f"  Train: loss={train_metrics['loss']:.4f}, mpjpe={train_metrics['mpjpe']:.1f}mm, pa={train_metrics['pa_mpjpe']:.1f}mm")
        print(f"  Test:  mpjpe={test_metrics['mpjpe']:.1f}mm, pa={test_metrics['pa_mpjpe']:.1f}mm, pck@50={test_metrics['pck@50']:.1f}%")

        save_every = cfg.get('save_every', 0)
        if save_every and (epoch % save_every == 0 or epoch == n_epochs):
            ep_path = os.path.join(save_dir, f'v2_epoch{epoch}.pth')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'test_metrics': test_metrics, 'cfg': cfg}, ep_path)
            print(f"  -> periodic ckpt saved: {ep_path}")

        if test_metrics['mpjpe'] < best_mpjpe:
            best_mpjpe = test_metrics['mpjpe']
            best_mpjpe_epoch = epoch

            save_path = os.path.join(save_dir, 'v2_best_mpjpe.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_mpjpe': best_mpjpe,
                'test_metrics': test_metrics,
                'cfg': cfg
            }, save_path)
            print(f"  -> New best MPJPE! Saved to {save_path}")

        if test_metrics['pa_mpjpe'] < best_pampjpe:
            best_pampjpe = test_metrics['pa_mpjpe']
            best_pampjpe_epoch = epoch

            save_path = os.path.join(save_dir, 'v2_best_pampjpe.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_pa_mpjpe': best_pampjpe,
                'test_metrics': test_metrics,
                'cfg': cfg
            }, save_path)
            print(f"  -> New best PA-MPJPE! Saved to {save_path}")

        if cfg.get('early_stop', 20) > 0 and (epoch - best_pampjpe_epoch) >= cfg['early_stop']:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\nPhase 1 completed!")
    print(f"  Best MPJPE: {best_mpjpe:.1f}mm at epoch {best_mpjpe_epoch}")
    print(f"  Best PA-MPJPE: {best_pampjpe:.1f}mm at epoch {best_pampjpe_epoch}")
    return best_pampjpe

def train_phase2_dualpath_epoch(model, dataloader, optimizer, device, epoch, n_epochs,
                                 limb_loss_fn, root_weight=1.0, limb_weight=0.1,
                                 simple_loss=False, env_norm=False, global_mean=None):
    """Train Phase 2 dual-path architecture.

    Loss = L_final + root_weight * L_root + limb_weight * L_limb

    Note: Phase 1 is frozen, so we don't supervise it during training.
          Phase 1 output is only used at inference to provide relative pose.

    Args:
        simple_loss: If True, use only MPJPE loss like DT-Pose/GraphPose-Fi.
                     This can help avoid overfitting on random_split.
        env_norm: If True, apply global XY normalization.
        global_mean: [3] tensor of global position mean from training data.
    """
    model.train()

    loss_meter = AverageMeter()
    mpjpe_meter = AverageMeter()
    pampjpe_meter = AverageMeter()
    root_error_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f"Phase2 Epoch {epoch}/{n_epochs}")

    for batch in pbar:
        if len(batch) == 3:
            csi, pose_gt, _ = batch
        else:
            csi, pose_gt = batch[:2]

        csi = csi.to(device)
        pose_gt = pose_gt.to(device)
        B = csi.size(0)

        num_joints = pose_gt.shape[1]
        gt_root_orig = pose_gt[:, 0, :].clone() if num_joints == 17 else (pose_gt[:, 5, :] + pose_gt[:, 7, :]) / 2

        if env_norm and global_mean is not None:
            pose_gt_norm = pose_gt - global_mean.view(1, 1, 3)
            gt_root = gt_root_orig - global_mean
        else:
            pose_gt_norm = pose_gt
            gt_root = gt_root_orig

        optimizer.zero_grad()

        output = model(csi)
        pose_final = output['pose']
        root_position = output['root_position']
        aux_poses = output.get('aux_poses')

        if simple_loss:
            loss = torch.norm(pose_final - pose_gt_norm, dim=-1).mean()
        else:
            L_final = F.l1_loss(pose_final, pose_gt_norm) + F.mse_loss(pose_final, pose_gt_norm)
            L_root = F.l1_loss(root_position, gt_root) + F.mse_loss(root_position, gt_root)
            L_limb = limb_loss_fn(pose_final, pose_gt_norm)
            loss = L_final + root_weight * L_root + limb_weight * L_limb

        if aux_poses is not None and len(aux_poses) > 1:
            aux_loss = pose_final.new_zeros(())
            for ap in aux_poses[:-1]:
                if simple_loss:
                    aux_loss = aux_loss + torch.norm(ap - pose_gt_norm, dim=-1).mean()
                else:
                    aux_loss = aux_loss + F.l1_loss(ap, pose_gt_norm) + F.mse_loss(ap, pose_gt_norm)
            loss = loss + aux_loss / (len(aux_poses) - 1)

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            if env_norm and global_mean is not None:
                pose_final_denorm = pose_final + global_mean.view(1, 1, 3)
                root_position_denorm = root_position + global_mean
                mpjpe = compute_mpjpe(pose_final_denorm, pose_gt, align=False)
                pampjpe = compute_pa_mpjpe(pose_final_denorm.cpu(), pose_gt.cpu())
                root_error = torch.norm(root_position_denorm - gt_root_orig, dim=-1).mean() * 1000
            else:
                mpjpe = compute_mpjpe(pose_final, pose_gt, align=False)
                pampjpe = compute_pa_mpjpe(pose_final.cpu(), pose_gt.cpu())
                root_error = torch.norm(root_position - gt_root_orig, dim=-1).mean() * 1000

        loss_meter.update(loss.item(), B)
        mpjpe_meter.update(mpjpe.item(), B)
        pampjpe_meter.update(pampjpe.item(), B)
        root_error_meter.update(root_error.item(), B)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", mpjpe=f"{mpjpe_meter.avg:.1f}",
                        pa=f"{pampjpe_meter.avg:.1f}", root=f"{root_error_meter.avg:.1f}")

    return {
        'loss': loss_meter.avg,
        'mpjpe': mpjpe_meter.avg,
        'pa_mpjpe': pampjpe_meter.avg,
        'root_error': root_error_meter.avg,
    }

def train_phase2_dualpath(model, train_loader, test_loader, cfg, device, save_dir,
                          env_norm=False, global_mean=None):
    """Phase 2 dual-path training loop."""
    n_epochs = cfg['epochs']
    lr = cfg['lr']
    num_joints = cfg.get('num_joints', 17)
    finetune = cfg.get('finetune', False)
    backbone_lr_ratio = cfg.get('backbone_lr_ratio', 0.1)
    warmup_epochs = cfg.get('warmup_epochs', 5)

    root_weight = cfg.get('phase2_root_weight', 1.0)
    limb_weight = cfg.get('limb_weight', 0.1)
    simple_loss = cfg.get('simple_loss', False)

    limb_loss_fn = LimbLengthLoss(num_joints=num_joints).to(device)

    mode_str = "FINETUNE" if finetune else "FROZEN"
    loss_str = "SIMPLE (MPJPE only)" if simple_loss else "MULTI-TERM"

    print("=" * 70)
    print("CSI-StructPose V2 Training - PHASE 2 (Dual-Path)")
    print("=" * 70)
    print(f"  Phase 1: {mode_str}")
    print(f"  Loss: {loss_str}")
    print(f"  Epochs: {n_epochs}, LR: {lr}, Warmup: {warmup_epochs}")
    print(f"  Weights: root={root_weight}, limb={limb_weight}")
    print(f"  Env normalization: {env_norm}")
    print(f"  Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    print("=" * 70)

    if finetune:
        phase1_modules = ['csi_encoder', 'body_part_generator', 'skeleton_transformer',
                          'part_aggregator', 'joint_decoder']
        phase1_params = []
        phase2_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(m in name for m in phase1_modules):
                    phase1_params.append(param)
                else:
                    phase2_params.append(param)

        optimizer = optim.AdamW([
            {'params': phase1_params, 'lr': lr * backbone_lr_ratio},
            {'params': phase2_params, 'lr': lr},
        ], weight_decay=0.01)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs, n_epochs)

    best_mpjpe = float('inf')
    best_pampjpe = float('inf')
    best_mpjpe_epoch = 0
    best_pampjpe_epoch = 0

    for epoch in range(1, n_epochs + 1):
        train_metrics = train_phase2_dualpath_epoch(
            model, train_loader, optimizer, device, epoch, n_epochs,
            limb_loss_fn=limb_loss_fn,
            root_weight=root_weight,
            limb_weight=limb_weight,
            simple_loss=simple_loss,
            env_norm=env_norm,
            global_mean=global_mean,
        )
        scheduler.step()

        test_metrics = evaluate(
            model, test_loader, device,
            verbose=(epoch == 1 or epoch % 10 == 0),
            env_norm=env_norm,
            global_mean=global_mean,
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{n_epochs} (lr={current_lr:.2e})")
        print(f"  Train: loss={train_metrics['loss']:.4f}, mpjpe={train_metrics['mpjpe']:.1f}mm, "
              f"pa={train_metrics['pa_mpjpe']:.1f}mm, root={train_metrics['root_error']:.1f}mm")

        if 'phase2_mpjpe' in test_metrics:
            print(f"  Test:  fused={test_metrics['mpjpe']:.1f}mm, phase2={test_metrics['phase2_mpjpe']:.1f}mm, "
                  f"root_err={test_metrics['root_error']:.1f}mm, pa={test_metrics['pa_mpjpe']:.1f}mm")
        else:
            print(f"  Test:  mpjpe={test_metrics['mpjpe']:.1f}mm, pa={test_metrics['pa_mpjpe']:.1f}mm, pck@50={test_metrics['pck@50']:.1f}%")

        if test_metrics['mpjpe'] < best_mpjpe:
            best_mpjpe = test_metrics['mpjpe']
            best_mpjpe_epoch = epoch

            save_path = os.path.join(save_dir, 'v2_phase2_best_mpjpe.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_mpjpe': best_mpjpe,
                'test_metrics': test_metrics,
                'cfg': cfg
            }, save_path)
            print(f"  -> New best MPJPE! Saved to {save_path}")

        if test_metrics['pa_mpjpe'] < best_pampjpe:
            best_pampjpe = test_metrics['pa_mpjpe']
            best_pampjpe_epoch = epoch

            save_path = os.path.join(save_dir, 'v2_phase2_best_pampjpe.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_pa_mpjpe': best_pampjpe,
                'test_metrics': test_metrics,
                'cfg': cfg
            }, save_path)
            print(f"  -> New best PA-MPJPE! Saved to {save_path}")

    print(f"\nPhase 2 completed!")
    print(f"  Best MPJPE: {best_mpjpe:.1f}mm at epoch {best_mpjpe_epoch}")
    print(f"  Best PA-MPJPE: {best_pampjpe:.1f}mm at epoch {best_pampjpe_epoch}")
    return best_mpjpe

def main(args):
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    if getattr(args, 'part_points', None):
        import models.csi_structpose_v2 as _m
        h, t, limb = (int(x) for x in args.part_points.split(','))
        _m.PART_POINTS = {'Head': h, 'Torso': t,
                          'L_Arm': limb, 'R_Arm': limb,
                          'L_Leg': limb, 'R_Leg': limb}
        print(f"[part_points] override -> {_m.PART_POINTS} "
              f"(total {h + t + 4 * limb})")

    if args.phase == 1:
        root_relative = not args.absolute
    elif args.phase == 2:
        root_relative = False
        if not args.phase1_ckpt:
            raise ValueError("Phase 2 requires --phase1_ckpt")
    else:
        raise ValueError("Must specify --phase 1 or --phase 2")

    if args.lr is None:
        args.lr = 1e-3 if args.phase == 2 else 1e-4
        print(f"Using default LR for Phase {args.phase}: {args.lr}")

    cfg = {
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'early_stop': args.early_stop,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'limb_weight': args.limb_weight,
        'bone_weight': args.bone_weight,
        'freeze_encoder': args.freeze_encoder,
        'save_every': args.save_every,
        'num_joints': 17 if args.dataset == 'mmfi' else 14,
        'embed_dim': args.embed_dim,
        'num_encoder_layers': args.encoder_layers,
        'num_graph_layers': args.graph_layers,
        'num_gcn_layers': args.gcn_layers,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'use_improved_gcn': args.improved_gcn,
        'cheby_k': args.cheby_k,
        'encoder_type': args.encoder,
        'pretrained_encoder': not args.no_pretrained,
        'phase2_root_weight': args.phase2_root_weight,
        'finetune': args.finetune,
        'backbone_lr_ratio': args.backbone_lr_ratio,
        'simple_loss': args.simple_loss,
        'data_root': args.data_root,
        'ablation_mode': args.ablation_mode,
        'sga_mode': args.sga_mode,
        'aspn_mode': args.aspn_mode,
        'use_refiner': args.use_refiner,
        'use_iter_refiner': args.use_iter_refiner,
        'use_dual_refiner': args.use_dual_refiner,
        'use_rigid': args.use_rigid,
        'rigid_residual': args.rigid_residual,
        'refiner_layers': args.refiner_layers,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_name = os.path.basename(args.data_root.rstrip('/'))
    data_name = data_name.replace('_absolute', '')
    save_dir = os.path.join("pretrained", f"v2_{args.dataset}_{data_name}_phase{args.phase}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("pretrained", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Phase: {args.phase}")
    print(f"Coordinates: {'root-relative' if root_relative else 'absolute'}")
    print(f"Env normalization: {args.env_norm}")

    global_mean = None
    if args.env_norm:
        if args.dataset == 'mmfi' and args.loeo_test_env is not None:
            global_mean_np, _tr_envs = compute_global_position_mean_loeo(args.data_root, args.loeo_test_env)
            print(f"\n[LOEO env_norm] global mean from TRAIN envs {_tr_envs} (test={args.loeo_test_env}):")
        else:
            global_mean_np = compute_global_position_mean(args.data_root)
            print(f"\nGlobal position mean (from all training data):")
        global_mean = torch.tensor(global_mean_np, dtype=torch.float32, device=device)
        print(f"  x={global_mean_np[0]:.3f}, y={global_mean_np[1]:.3f}, z={global_mean_np[2]:.3f}")
        print(f"  (whole pose centered by this mean, all 3 axes)")

    if args.dataset == 'mmfi':
        if args.loeo_test_env is not None:
            from datasets import make_loeo_split
            print(f"\n[LOEO] test env = {args.loeo_test_env}; "
                  f"per-env data root = {args.data_root}")
            train_dataset, test_dataset = make_loeo_split(
                per_env_root=args.data_root,
                test_env=args.loeo_test_env,
                root_relative=root_relative,
            )
        else:
            from datasets import MMFiPreprocessedCSIOnly
            train_dataset = MMFiPreprocessedCSIOnly(args.data_root, 'train', root_relative=root_relative)
            test_dataset = MMFiPreprocessedCSIOnly(args.data_root, 'test', root_relative=root_relative)
    else:
        from datasets import WiFiPose3DPreprocessed
        train_dataset = WiFiPose3DPreprocessed(args.data_root, 'train', root_relative=root_relative)
        test_dataset = WiFiPose3DPreprocessed(args.data_root, 'test', root_relative=root_relative)

    rng_generator = torch.Generator()
    rng_generator.manual_seed(0)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              generator=rng_generator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    sample_csi = train_dataset[0][0]
    csi_shape = tuple(sample_csi.shape)
    cfg['num_frames'] = int(csi_shape[0])
    cfg['num_subcarriers'] = int(csi_shape[3])
    cfg['in_chan_mult'] = int(csi_shape[4]) if len(csi_shape) == 5 else 1
    print(f"CSI input per-sample shape: {csi_shape} -> num_frames={cfg['num_frames']}, "
          f"num_subcarriers={cfg['num_subcarriers']}, in_chan_mult={cfg['in_chan_mult']}")

    if args.phase == 1:
        model = build_csi_structpose_v2(cfg)
        model = model.to(device)
        if getattr(args, 'encoder_ckpt', None):
            ck = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
            enc_sd = ck.get('encoder_state_dict', ck)
            missing, unexpected = model.csi_encoder.load_state_dict(enc_sd, strict=False)
            print(f"Loaded SSL-pretrained encoder from {args.encoder_ckpt} "
                  f"(val_recon={ck.get('val_recon')}, missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        model = build_csi_structpose_v2_phase2(cfg)
        model = model.to(device)
        model = load_phase1_into_phase2(model, args.phase1_ckpt, device, freeze_phase1=not args.finetune)
        if getattr(args, 'encoder_ckpt', None):
            ck = torch.load(args.encoder_ckpt, map_location=device, weights_only=False)
            enc_sd = ck.get('encoder_state_dict', ck)
            missing, unexpected = model.csi_encoder.load_state_dict(enc_sd, strict=False)
            print(f"Loaded SSL-pretrained encoder into Phase-2 from {args.encoder_ckpt} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} (trainable: {trainable_params:,})")

    if args.phase == 1:
        best_metric = train_phase1(model, train_loader, test_loader, cfg, device, save_dir,
                                   env_norm=args.env_norm, global_mean=global_mean)
    else:
        best_metric = train_phase2_dualpath(
            model, train_loader, test_loader, cfg, device, save_dir,
            env_norm=args.env_norm,
            global_mean=global_mean,
        )

    import shutil
    if args.phase == 1:
        src = os.path.join(save_dir, 'v2_best.pth')
        dst = os.path.join("pretrained", 'v2_best.pth')
    else:
        src = os.path.join(save_dir, 'v2_phase2_best.pth')
        dst = os.path.join("pretrained", 'v2_phase2_best.pth')

    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied to {dst}")

    return best_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSI-StructPose V2 Training')

    parser.add_argument('--phase', type=int, required=True, choices=[1, 2],
                        help='Training phase: 1=relative pose, 2=absolute pose')
    parser.add_argument('--phase1_ckpt', type=str, default=None,
                        help='Phase 1 checkpoint (required for Phase 2)')

    parser.add_argument('--phase2_root_weight', type=float, default=1.0,
                        help='Root position loss weight (default: 1.0)')
    parser.add_argument('--finetune', action='store_true',
                        help='Joint training (unfreeze Phase 1)')
    parser.add_argument('--backbone_lr_ratio', type=float, default=0.1,
                        help='Backbone LR ratio when finetuning (default: 0.1)')
    parser.add_argument('--simple_loss', action='store_true',
                        help='Use simple MPJPE loss only (like DT-Pose/GraphPose-Fi)')

    parser.add_argument('--encoder_ckpt', type=str, default=None,
                        help='Optional pretrained CSIEncoderV2 weights to warm-start the encoder')

    parser.add_argument('--data_root', type=str, required=True,
                        help='Preprocessed data directory')
    parser.add_argument('--dataset', type=str, default='mmfi', choices=['mmfi', 'wifipose'],
                        help='Dataset type (default: mmfi)')
    parser.add_argument('--absolute', action='store_true',
                        help='Use absolute coordinates (default: root-relative)')
    parser.add_argument('--env_norm', action='store_true',
                        help='Enable room-wise XY position normalization (Phase 2 only)')
    parser.add_argument('--loeo_test_env', type=str, default=None,
                        choices=[None, 'E01', 'E02', 'E03', 'E04'],
                        help='Leave-one-environment-out: env to hold out for test. '
                             'When set, --data_root must point to a per-env preprocessed '
                             'directory produced by scripts/preprocess_per_env.py.')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Learning rate warmup epochs (default: 5)')
    parser.add_argument('--early_stop', type=int, default=20,
                        help='Early stopping patience (0=disabled)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-4 for Phase1, 1e-3 for Phase2)')
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='Weight decay for AdamW (default: 0.02, same as GraphPose-Fi)')
    parser.add_argument('--limb_weight', type=float, default=0.1)
    parser.add_argument('--bone_weight', type=float, default=0.0,
                        help='Weight for the bone-direction (articulation) loss; '
                             '0 disables. Try ~0.5-2.0 to fight structure collapse.')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze the (SSL-pretrained) CSI encoder; train only the '
                             'SGA+decoder head (DT-Pose regime, preserves SSL features).')
    parser.add_argument('--save_every', type=int, default=0,
                        help='Also save a v2_epoch{N}.pth every N epochs (0=off); lets us pick a '
                             'variance/PA trade-off checkpoint, not just the best-PA mean-hedge one.')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--encoder_layers', type=int, default=4)
    parser.add_argument('--graph_layers', type=int, default=4)
    parser.add_argument('--gcn_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--improved_gcn', action='store_true',
                        help='Use improved GCN (Chebyshev + attention)')
    parser.add_argument('--cheby_k', type=int, default=3)
    parser.add_argument('--encoder', type=str, default='transformer',
                        choices=['transformer', 'resnet34', 'resnet34_v1'],
                        help='Encoder type: transformer (CNN+Trans) or resnet34 (ImageNet pretrained)')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use ImageNet pretrained weights for ResNet34 encoder')
    parser.add_argument('--ablation_mode', type=str, default=None,
                        choices=['no_bplq'],
                        help='Ablation: "no_bplq" removes BP-LQ pipeline (CSI→MLP→pose)')
    parser.add_argument('--sga_mode', type=str, default='skeleton',
                        choices=['skeleton', 'full_attn', 'none'],
                        help='SGA variant: skeleton (default), full_attn (no skeleton mask), none (skip SGA)')
    parser.add_argument('--aspn_mode', type=str, default='full',
                        choices=['full', 'no_decomp', 'no_phase'],
                        help='ASPN variant: full (default), no_decomp (skip spatial decomp), no_phase (amplitude only)')
    parser.add_argument('--use_refiner', action='store_true',
                        help='Phase 2: add single residual pose-refinement decoder (legacy)')
    parser.add_argument('--use_iter_refiner', action='store_true',
                        help='Phase 2: iterative residual keypoint-decoder (PETR-style) + deep supervision; '
                             'replaces the pooled root head, predicts absolute pose directly')
    parser.add_argument('--use_dual_refiner', action='store_true',
                        help='Phase 2: keep ASPN root anchor + refiner over BOTH frozen Stage-1 tokens and a '
                             'trainable localization token-encoder (multimodal ASPN+CSI)')
    parser.add_argument('--use_rigid', action='store_true',
                        help='Phase 2: abandon ASPN; predict a rigid world-pose (R,t) from CSI tokens and apply '
                             'to the frozen canonical Stage-1 pose (absolute = R@stage1 + t)')
    parser.add_argument('--rigid_residual', action='store_true',
                        help='Add a small non-rigid residual refiner after the rigid transform')
    parser.add_argument('--refiner_layers', type=int, default=3,
                        help='Number of transformer-decoder layers in the Phase-2 refiner(s)')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--part_points', type=str, default=None,
                        help='Override BP-LQ token allocation as '
                             '"head,torso,limb" (limbs equal), total must '
                             'stay 150 for capacity-matched comparison, '
                             'e.g. 20,30,25 (default) or 25,25,25')

    args = parser.parse_args()
    main(args)
