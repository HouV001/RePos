"""
CSI-StructPose V2: Body-Part Latent Queries with Skeleton Graph Attention

Key improvements over V1:
1. Single phase architecture (no Phase 2)
2. Body-part latent query generation (6 parts)
3. Skeleton-graph attention instead of kNN
4. Part-to-Joint decoder with GCN refinement

Architecture:
    CSI → Encoder → Body-Part PC Generator → Skeleton-Graph Transformer → Part-to-Joint Decoder → Pose
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from .skeleton_gcn import build_adjacency_matrix, SKELETON_14, SKELETON_17

BODY_PARTS = ['Head', 'Torso', 'L_Arm', 'R_Arm', 'L_Leg', 'R_Leg']
NUM_PARTS = 6

PART_POINTS = {
    'Head': 20,
    'Torso': 30,
    'L_Arm': 25,
    'R_Arm': 25,
    'L_Leg': 25,
    'R_Leg': 25,
}

PART_ADJACENCY = torch.tensor([
    [1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1],
], dtype=torch.float32)

PART_TO_JOINTS_17 = {
    0: [9, 10],
    1: [0, 7, 8],
    2: [11, 12, 13],
    3: [14, 15, 16],
    4: [1, 2, 3],
    5: [4, 5, 6],
}

PART_TO_JOINTS_14 = {
    0: [0, 1],
    1: [],
    2: [2, 4, 8],
    3: [3, 6, 10],
    4: [5, 9, 12],
    5: [7, 11, 13],
}

PART_TO_JOINTS_18 = {
    0: [0, 14, 15, 16, 17],
    1: [1],
    2: [2, 3, 4],
    3: [5, 6, 7],
    4: [8, 9, 10],
    5: [11, 12, 13],
}

class CSIEncoderV2(nn.Module):
    """
    Improved CSI Encoder with image-like view.

    Treats CSI as image: [B, T, rx, tx, sub] → [B, T*rx*tx, sub] → CNN + Transformer

    This preserves spatial structure better than per-pair processing.
    """

    def __init__(self,
                 num_frames: int = 5,
                 num_rx: int = 3,
                 num_tx: int = 3,
                 num_subcarriers: int = 114,
                 embed_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 in_chan_mult: int = 1):
        super().__init__()

        self.num_frames = num_frames
        self.num_antennas = num_rx * num_tx
        self.num_subcarriers = num_subcarriers
        self.embed_dim = embed_dim
        self.in_chan_mult = in_chan_mult

        self.cnn = nn.Sequential(
            nn.Conv2d(num_rx * num_tx * in_chan_mult, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, num_subcarriers * num_frames, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            csi: [B, T, rx, tx, sub] amplitude-only, OR
                 [B, T, rx, tx, sub, C] with C extra channels per antenna
                 (e.g. C=2 real+imag for complex CSI).

        Returns:
            features: [B, L, embed_dim] encoded features
        """
        csi = torch.clamp(csi, -10.0, 10.0)

        csi = torch.where(torch.isfinite(csi), csi, torch.zeros_like(csi))

        if csi.dim() == 6:
            B, T, rx, tx, sub, C = csi.shape
            csi = csi.permute(0, 2, 3, 5, 4, 1).reshape(B, rx * tx * C, sub, T)
        else:
            B, T, rx, tx, sub = csi.shape
            csi = csi.permute(0, 2, 3, 4, 1).reshape(B, rx * tx, sub, T)

        features = self.cnn(csi)

        features = features.flatten(2).permute(0, 2, 1)

        features = features + self.pos_embed[:, :features.size(1), :]

        features = self.transformer(features)
        features = self.norm(features)

        return features

class CSIEncoderResNet34(nn.Module):
    """
    ResNet34-based CSI Encoder (similar to GraphPose-Fi).

    Processes each antenna pair through ResNet34, then aggregates.
    Uses ImageNet pretrained weights for better feature extraction.

    Input: [B, T, rx, tx, sub] → process per antenna → aggregate → [B, L, D]
    """

    def __init__(self,
                 num_frames: int = 5,
                 num_rx: int = 3,
                 num_tx: int = 3,
                 num_subcarriers: int = 114,
                 embed_dim: int = 256,
                 pretrained: bool = True,
                 agg_mode: str = 'attn'):
        super().__init__()

        import torchvision

        self.num_frames = num_frames
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_antennas = num_rx * num_tx
        self.num_subcarriers = num_subcarriers
        self.embed_dim = embed_dim
        self.agg_mode = agg_mode

        resnet = torchvision.models.resnet34(
            weights='IMAGENET1K_V1' if pretrained else None
        )

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.bn_final = nn.BatchNorm2d(512)

        if agg_mode == 'attn':
            self.ant_attn = nn.Sequential(
                nn.Conv1d(512, 128, 1),
                nn.ReLU(),
                nn.Conv1d(128, 1, 1),
            )
        elif agg_mode == 'mean':
            self.ant_attn = None
        else:
            raise ValueError(f"Unknown agg_mode: {agg_mode}")

        self.proj = nn.Linear(512, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def _encode_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode single antenna CSI through ResNet34.

        Args:
            x: [B, 1, H, W] single channel CSI image

        Returns:
            features: [B, 512, h, w] ResNet features
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            csi: [B, T, rx, tx, sub] CSI amplitude

        Returns:
            features: [B, L, embed_dim] encoded features
        """
        B, T, rx, tx, sub = csi.shape

        csi = torch.clamp(csi, -10.0, 10.0)
        csi = torch.where(torch.isfinite(csi), csi, torch.zeros_like(csi))

        csi = csi.permute(0, 2, 3, 4, 1).reshape(B, self.num_antennas, sub, T)

        ant_features = []
        for i in range(self.num_antennas):
            x = csi[:, i:i+1, :, :]
            feat = self._encode_single(x)
            feat = feat.mean(dim=[2, 3])
            ant_features.append(feat)

        ant_features = torch.stack(ant_features, dim=1)

        if self.agg_mode == 'attn':
            x = ant_features.permute(0, 2, 1)
            attn = self.ant_attn(x)
            attn = F.softmax(attn, dim=-1)
            features = (x * attn).sum(dim=-1)
        else:
            features = ant_features.mean(dim=1)

        features = self.proj(features)
        features = self.norm(features)

        features = features.unsqueeze(1).expand(-1, 64, -1)

        return features

class CSIEncoderResNet34V2(nn.Module):
    """
    Improved ResNet34 Encoder that preserves spatial structure better.

    Instead of processing each antenna separately, treats CSI as multi-channel image.
    More similar to original GraphPose-Fi approach.

    Input: [B, T, rx, tx, sub] → [B, rx*tx, sub, T] → ResNet → [B, L, D]
    """

    def __init__(self,
                 num_frames: int = 5,
                 num_rx: int = 3,
                 num_tx: int = 3,
                 num_subcarriers: int = 114,
                 embed_dim: int = 256,
                 pretrained: bool = True):
        super().__init__()

        import torchvision

        self.num_frames = num_frames
        self.num_antennas = num_rx * num_tx
        self.num_subcarriers = num_subcarriers
        self.embed_dim = embed_dim

        resnet = torchvision.models.resnet34(
            weights='IMAGENET1K_V1' if pretrained else None
        )

        self.conv1 = nn.Conv2d(self.num_antennas, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            pretrained_weight = resnet.conv1.weight.data
            new_weight = pretrained_weight.repeat(1, 3, 1, 1)[:, :self.num_antennas, :, :]
            self.conv1.weight.data = new_weight / 3

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.bn_final = nn.BatchNorm2d(512)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 1))

        self.proj = nn.Linear(512, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            csi: [B, T, rx, tx, sub] CSI amplitude

        Returns:
            features: [B, L, embed_dim] encoded features (L=64)
        """
        B, T, rx, tx, sub = csi.shape

        csi = torch.clamp(csi, -10.0, 10.0)
        csi = torch.where(torch.isfinite(csi), csi, torch.zeros_like(csi))

        csi = csi.permute(0, 2, 3, 4, 1).reshape(B, self.num_antennas, sub, T)

        x = self.conv1(csi)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn_final(x)

        x = self.adaptive_pool(x)

        x = x.squeeze(-1).permute(0, 2, 1)

        x = self.proj(x)
        x = self.norm(x)

        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=64, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)

        x = x + self.pos_embed

        return x

class BodyPartLatentQueryGenerator(nn.Module):
    """
    Body-Part Latent Query generator with optional two-stage architecture.

    Two modes:
    - Legacy (use_two_stage=False): Direct high-dim generation (~20M params)
    - Efficient (use_two_stage=True): Two-stage low→high dim (~3M params)

    Parameter comparison (for 150 points, point_dim=256):
    - Legacy: ~20M params (6 independent high-dim generators)
    - Two-stage: ~3M params (6 low-dim generators + shared projection)
    """

    def __init__(self,
                 embed_dim: int = 256,
                 num_heads: int = 8,
                 point_dim: int = 256,
                 low_dim: int = 64,
                 dropout: float = 0.1,
                 use_two_stage: bool = True):
        super().__init__()

        self.embed_dim = embed_dim
        self.point_dim = point_dim
        self.low_dim = low_dim
        self.num_parts = NUM_PARTS
        self.use_two_stage = use_two_stage

        self.part_names = BODY_PARTS
        self.part_points = [PART_POINTS[name] for name in self.part_names]
        self.total_points = sum(self.part_points)

        self.part_offsets = [0]
        for n in self.part_points[:-1]:
            self.part_offsets.append(self.part_offsets[-1] + n)

        self.part_queries = nn.Parameter(torch.randn(NUM_PARTS, embed_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.point_generators = nn.ModuleList()
        if use_two_stage:
            for i, num_points in enumerate(self.part_points):
                self.point_generators.append(
                    nn.Sequential(
                        nn.Linear(embed_dim, embed_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(embed_dim, num_points * low_dim),
                    )
                )
            self.shared_projection = nn.Sequential(
                nn.Linear(low_dim, point_dim),
                nn.LayerNorm(point_dim),
            )
        else:
            for i, num_points in enumerate(self.part_points):
                self.point_generators.append(
                    nn.Sequential(
                        nn.Linear(embed_dim, embed_dim * 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(embed_dim * 2, num_points * point_dim),
                    )
                )
            self.shared_projection = None

        self.point_pos_embeds = nn.ParameterList()
        for num_points in self.part_points:
            self.point_pos_embeds.append(
                nn.Parameter(torch.randn(num_points, point_dim) * 0.02)
            )

    def forward(self, csi_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            csi_features: [B, L, embed_dim] encoded CSI features

        Returns:
            points: [B, total_points, point_dim] point features
            part_ids: [B, total_points] part assignment (0-5)
        """
        B = csi_features.shape[0]
        device = csi_features.device

        queries = self.part_queries.unsqueeze(0).expand(B, -1, -1)

        part_features, _ = self.cross_attn(queries, csi_features, csi_features)
        part_features = self.cross_norm(queries + part_features)

        all_points = []
        all_part_ids = []

        for i, (generator, pos_embed, num_points) in enumerate(
            zip(self.point_generators, self.point_pos_embeds, self.part_points)
        ):
            part_feat = part_features[:, i, :]

            if self.use_two_stage:
                points_low = generator(part_feat)
                points_low = points_low.view(B, num_points, self.low_dim)
                points = self.shared_projection(points_low)
            else:
                points = generator(part_feat)
                points = points.view(B, num_points, self.point_dim)

            points = points + pos_embed.unsqueeze(0)

            all_points.append(points)
            all_part_ids.append(torch.full((B, num_points), i, device=device, dtype=torch.long))

        points = torch.cat(all_points, dim=1)
        part_ids = torch.cat(all_part_ids, dim=1)

        return points, part_ids

class SkeletonGraphAttention(nn.Module):
    """
    Skeleton-based attention instead of kNN.

    Two types of attention:
    1. Intra-Part: Points within the same body part attend to each other
    2. Inter-Part: Points from connected body parts attend to each other

    Optimization: Pre-compute attention masks in __init__ since skeleton structure is fixed.
    """

    def __init__(self,
                 dim: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.num_parts = NUM_PARTS

        self.part_points = [PART_POINTS[name] for name in BODY_PARTS]
        self.total_points = sum(self.part_points)

        intra_mask, inter_mask = self._precompute_masks()
        self.register_buffer('intra_mask', intra_mask)
        self.register_buffer('inter_mask', inter_mask)

        self.intra_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.intra_norm = nn.LayerNorm(dim)

        self.inter_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.inter_norm = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def _precompute_masks(self) -> tuple:
        """
        Pre-compute attention masks based on fixed skeleton structure.

        Returns:
            intra_mask: [N, N] float mask for intra-part attention (-inf for blocked)
            inter_mask: [N, N] float mask for inter-part attention (-inf for blocked)
        """
        N = self.total_points

        part_ids = []
        for part_idx, num_points in enumerate(self.part_points):
            part_ids.extend([part_idx] * num_points)
        part_ids = torch.tensor(part_ids)

        part_i = part_ids.unsqueeze(1)
        part_j = part_ids.unsqueeze(0)

        same_part = (part_i == part_j)
        intra_mask = torch.zeros(N, N)
        intra_mask.masked_fill_(~same_part, float('-inf'))

        adj = PART_ADJACENCY
        connected = adj[part_i.expand(N, N), part_j.expand(N, N)]
        inter_mask = torch.zeros(N, N)
        inter_mask.masked_fill_(connected == 0, float('-inf'))

        return intra_mask, inter_mask

    def forward(self, points: torch.Tensor, part_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [B, N, dim] point features
            part_ids: [B, N] part assignment (unused, masks are pre-computed)

        Returns:
            points: [B, N, dim] updated point features
        """
        intra_out, _ = self.intra_attn(points, points, points, attn_mask=self.intra_mask)
        points = self.intra_norm(points + intra_out)

        inter_out, _ = self.inter_attn(points, points, points, attn_mask=self.inter_mask)
        points = self.inter_norm(points + inter_out)

        points = self.ffn_norm(points + self.ffn(points))

        return points

class SkeletonGraphTransformer(nn.Module):
    """
    Multi-layer Skeleton Graph Transformer.

    Replaces PointTransformer's kNN-based attention with skeleton-aware attention.
    """

    def __init__(self,
                 dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            SkeletonGraphAttention(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, points: torch.Tensor, part_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [B, N, dim] point features
            part_ids: [B, N] part assignment

        Returns:
            points: [B, N, dim] transformed point features
        """
        for layer in self.layers:
            points = layer(points, part_ids)

        return self.norm(points)

class DirectPoseRegressor(nn.Module):
    """
    Ablation: CSI features → Cross-attn → MLP → pose, bypassing the BP-LQ pipeline.

    Used when --ablation_mode no_bplq to test the effectiveness of body-part
    latent queries. Uses a simple MLP decoder (no GCN) to isolate the
    contribution of the BP-LQ pipeline.

    Architecture:
        CSI features → Cross-attention with joint queries → MLP → Pose
    """

    def __init__(self,
                 embed_dim: int = 256,
                 num_joints: int = 17,
                 gcn_layers: int = 4,
                 dropout: float = 0.1,
                 use_improved_gcn: bool = True,
                 cheby_k: int = 3,
                 num_heads: int = 8):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim

        self.joint_queries = nn.Parameter(torch.randn(num_joints, embed_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.pose_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 3)
        )

    def forward(self, csi_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            csi_features: [B, L, embed_dim] encoded CSI features
        Returns:
            pose: [B, num_joints, 3]
        """
        B = csi_features.shape[0]

        joint_queries = self.joint_queries.unsqueeze(0).expand(B, -1, -1)

        joint_features, _ = self.cross_attn(joint_queries, csi_features, csi_features)
        joint_features = self.cross_norm(joint_queries + joint_features)

        pose = self.pose_head(joint_features)

        return pose

class FullSelfAttentionTransformer(nn.Module):
    """
    Ablation: standard Transformer encoder without skeleton-graph attention mask.

    Used when --sga_mode full_attn to test the effectiveness of skeleton-aware
    attention vs. unconstrained self-attention.
    """

    def __init__(self,
                 dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, points: torch.Tensor, part_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [B, N, dim] point features
            part_ids: [B, N] part assignment (ignored — no skeleton mask)
        Returns:
            points: [B, N, dim] transformed point features
        """
        return self.norm(self.encoder(points))

class SimplePointCloudDecoder(nn.Module):
    """
    Ablation: BP-LQs mean pool → MLP → pose (when sga_mode='none').

    Used when --sga_mode none to test the effectiveness of skeleton-graph
    attention. Skips both SGA and GCN, using only mean pooling + MLP.

    Architecture:
        BP-LQs [B, 150, point_dim] → mean pool → MLP → Pose [B, 17, 3]
    """

    def __init__(self,
                 point_dim: int = 256,
                 num_joints: int = 17,
                 dropout: float = 0.1):
        super().__init__()
        self.num_joints = num_joints
        self.point_dim = point_dim

        self.pose_head = nn.Sequential(
            nn.Linear(point_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_joints * 3)
        )

    def forward(self, points: torch.Tensor, part_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [B, N, point_dim] point features
            part_ids: [B, N] part assignment (unused)
        Returns:
            pose: [B, num_joints, 3]
        """
        B = points.shape[0]

        x = points.mean(dim=1)

        x = self.pose_head(x)

        return x.view(B, self.num_joints, 3)

class PartFeatureAggregator(nn.Module):
    """
    Aggregate point features into part-level features.

    Uses attention pooling within each body part.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()

        self.dim = dim
        self.num_parts = NUM_PARTS
        self.part_points = [PART_POINTS[name] for name in BODY_PARTS]

        self.part_queries = nn.Parameter(torch.randn(NUM_PARTS, dim) * 0.02)

        self.attn_pool = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, points: torch.Tensor, part_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            points: [B, N, dim] point features
            part_ids: [B, N] part assignment

        Returns:
            part_features: [B, 6, dim] aggregated part features
            part_centroids: [B, 6, dim] mean of each part's points (for constraints)
        """
        B = points.shape[0]

        part_features = []
        part_centroids = []

        offset = 0
        for i, num_points in enumerate(self.part_points):
            part_points = points[:, offset:offset + num_points, :]

            centroid = part_points.mean(dim=1, keepdim=True)
            part_centroids.append(centroid)

            query = self.part_queries[i:i+1].unsqueeze(0).expand(B, -1, -1)
            pooled, _ = self.attn_pool(query, part_points, part_points)

            part_features.append(pooled)
            offset += num_points

        part_features = torch.cat(part_features, dim=1)
        part_features = self.norm(part_features)

        part_centroids = torch.cat(part_centroids, dim=1)

        return part_features, part_centroids

class PartToJointDecoder(nn.Module):
    """
    Decode part features to joint positions using GCN-Transformer fusion.

    Architecture (inspired by DT-Pose + GraphPose-Fi):
    1. Cross-attention: Joint queries attend to part features with part-guided bias
    2. GCN-Transformer fusion: Local (Chebyshev GCN) + Global (Self-Attention)
    3. Final pose prediction

    Key improvements:
    - Default to GCN-Transformer fusion (not basic GCN)
    - Chebyshev K-hop aggregation for local skeleton topology
    - Multi-head self-attention for global joint relationships
    - Learnable joint embeddings for positional encoding
    """

    def __init__(self,
                 dim: int = 256,
                 num_joints: int = 17,
                 gcn_layers: int = 4,
                 dropout: float = 0.1,
                 use_improved_gcn: bool = True,
                 cheby_k: int = 3,
                 num_heads: int = 4):
        super().__init__()

        self.dim = dim
        self.num_joints = num_joints
        self.num_parts = NUM_PARTS
        self.use_improved_gcn = use_improved_gcn

        self.part_to_joints = {14: PART_TO_JOINTS_14, 17: PART_TO_JOINTS_17,
                               18: PART_TO_JOINTS_18}[num_joints]

        self.joint_queries = nn.Parameter(torch.randn(num_joints, dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(dim, 8, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)

        self.register_buffer('part_bias', self._build_part_bias())

        if use_improved_gcn:
            from .skeleton_gcn import ImprovedSkeletonGCN
            self.gcn_transformer = ImprovedSkeletonGCN(
                in_features=dim,
                hidden_dim=dim,
                out_features=3,
                num_joints=num_joints,
                num_layers=gcn_layers,
                K=cheby_k,
                num_heads=num_heads,
                dropout=dropout,
                use_graph_attention=True
            )
            self.pose_head = None
        else:
            self.gcn_transformer = None
            self.gcn_layers_basic = nn.ModuleList()
            self.gcn_norms = nn.ModuleList()

            A = build_adjacency_matrix(num_joints, self_loop=True)
            self.register_buffer('adj_matrix', A)

            for _ in range(gcn_layers):
                self.gcn_layers_basic.append(nn.Linear(dim, dim))
                self.gcn_norms.append(nn.LayerNorm(dim))

            self.gcn_dropout = nn.Dropout(dropout)

            self.pose_head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim // 2, 3)
            )

    def _build_part_bias(self) -> torch.Tensor:
        """
        Build attention bias to encourage joints to attend to their parent parts.

        Returns:
            bias: [num_joints, num_parts] attention bias
        """
        bias = torch.zeros(self.num_joints, self.num_parts)

        for part_idx, joint_list in self.part_to_joints.items():
            for joint_idx in joint_list:
                bias[joint_idx, part_idx] = 1.0

        row_sum = bias.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1
        bias = bias / row_sum

        return bias

    def forward(self, part_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            part_features: [B, 6, dim] part-level features

        Returns:
            pose: [B, num_joints, 3] predicted joint positions
        """
        B = part_features.shape[0]

        joint_queries = self.joint_queries.unsqueeze(0).expand(B, -1, -1)

        attn_bias = self.part_bias.unsqueeze(0).expand(B, -1, -1)
        attn_mask = (1.0 - attn_bias) * -5.0

        joint_features, attn_weights = self.cross_attn(
            joint_queries, part_features, part_features
        )

        bias_weighted_features = torch.bmm(attn_bias, part_features)
        joint_features = joint_features + 0.1 * bias_weighted_features

        joint_features = self.cross_norm(joint_queries + joint_features)

        if self.use_improved_gcn:
            pose = self.gcn_transformer(joint_features)
        else:
            h = joint_features
            for gcn, norm in zip(self.gcn_layers_basic, self.gcn_norms):
                h_new = torch.matmul(self.adj_matrix, gcn(h))
                h_new = F.gelu(h_new)
                h_new = self.gcn_dropout(h_new)
                h = norm(h + h_new)

            pose = self.pose_head(h)

        return pose

class CSIStructPoseV2(nn.Module):
    """
    CSI-StructPose V2 Phase 1: Skeleton-aware latent queries for relative pose estimation.

    Architecture:
        CSI → CSI Encoder → Body-Part PC Generator → Skeleton-Graph Transformer
            → Part Aggregator → Part-to-Joint Decoder → relative_pose [B, J, 3]

    For absolute pose estimation (Phase 2), use CSIStructPoseV2Phase2 instead.
    """

    def __init__(self,
                 num_frames: int = 5,
                 num_rx: int = 3,
                 num_tx: int = 3,
                 num_subcarriers: int = 114,
                 embed_dim: int = 256,
                 point_dim: int = 256,
                 num_encoder_layers: int = 4,
                 num_graph_layers: int = 4,
                 num_gcn_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_joints: int = 17,
                 use_improved_gcn: bool = True,
                 use_two_stage_generator: bool = True,
                 legacy_mode: bool = False,
                 cheby_k: int = 3,
                 encoder_type: str = 'transformer',
                 pretrained_encoder: bool = True,
                 ablation_mode: Optional[str] = None,
                 sga_mode: str = 'skeleton',
                 in_chan_mult: int = 1):
        """
        Args:
            encoder_type: 'transformer' (default CNN+Transformer) or 'resnet34' (ResNet34 like GraphPose-Fi)
            in_chan_mult: per-antenna input channels (1=amplitude, 2=complex real+imag)
            pretrained_encoder: Whether to use ImageNet pretrained weights for ResNet34
            ablation_mode: None (full model) or 'no_bplq' (CSI encoder → MLP, bypass BP-LQ pipeline)
            sga_mode: 'skeleton' (default SGA), 'full_attn' (unconstrained self-attention), 'none' (skip SGA)
        """
        super().__init__()

        if legacy_mode:
            use_improved_gcn = False
            use_two_stage_generator = False

        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_subcarriers = num_subcarriers
        self.use_improved_gcn = use_improved_gcn
        self.use_two_stage_generator = use_two_stage_generator
        self.legacy_mode = legacy_mode
        self.encoder_type = encoder_type
        self.ablation_mode = ablation_mode
        self.sga_mode = sga_mode

        if encoder_type == 'resnet34':
            self.csi_encoder = CSIEncoderResNet34V2(
                num_frames=num_frames,
                num_rx=num_rx,
                num_tx=num_tx,
                num_subcarriers=num_subcarriers,
                embed_dim=embed_dim,
                pretrained=pretrained_encoder
            )
        elif encoder_type == 'resnet34_v1':
            self.csi_encoder = CSIEncoderResNet34(
                num_frames=num_frames,
                num_rx=num_rx,
                num_tx=num_tx,
                num_subcarriers=num_subcarriers,
                embed_dim=embed_dim,
                pretrained=pretrained_encoder,
                agg_mode='attn'
            )
        else:
            self.csi_encoder = CSIEncoderV2(
                num_frames=num_frames,
                num_rx=num_rx,
                num_tx=num_tx,
                num_subcarriers=num_subcarriers,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                num_heads=num_heads,
                dropout=dropout,
                in_chan_mult=in_chan_mult
            )

        if ablation_mode == 'no_bplq':
            self.direct_regressor = DirectPoseRegressor(
                embed_dim=embed_dim,
                num_joints=num_joints,
                gcn_layers=num_gcn_layers,
                dropout=dropout,
                use_improved_gcn=use_improved_gcn,
                cheby_k=cheby_k,
                num_heads=num_heads
            )
            self.body_part_generator = None
            self.skeleton_transformer = None
            self.part_aggregator = None
            self.joint_decoder = None
            return

        self.body_part_generator = BodyPartLatentQueryGenerator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            point_dim=point_dim,
            dropout=dropout,
            use_two_stage=use_two_stage_generator
        )

        if sga_mode == 'skeleton':
            self.skeleton_transformer = SkeletonGraphTransformer(
                dim=point_dim,
                num_layers=num_graph_layers,
                num_heads=num_heads,
                dropout=dropout
            )
            self.simple_decoder = None
        elif sga_mode == 'full_attn':
            self.skeleton_transformer = FullSelfAttentionTransformer(
                dim=point_dim,
                num_layers=num_graph_layers,
                num_heads=num_heads,
                dropout=dropout
            )
            self.simple_decoder = None
        elif sga_mode == 'none':
            self.skeleton_transformer = None
            self.simple_decoder = SimplePointCloudDecoder(
                point_dim=point_dim,
                num_joints=num_joints,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown sga_mode: {sga_mode}")

        if sga_mode != 'none':
            self.part_aggregator = PartFeatureAggregator(
                dim=point_dim,
                num_heads=num_heads // 2
            )
        else:
            self.part_aggregator = None

        if sga_mode != 'none':
            self.joint_decoder = PartToJointDecoder(
                dim=point_dim,
                num_joints=num_joints,
                gcn_layers=num_gcn_layers,
                dropout=dropout,
                use_improved_gcn=use_improved_gcn,
                cheby_k=cheby_k,
                num_heads=num_heads
            )
        else:
            self.joint_decoder = None

    def forward(self, csi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            csi: [B, T, rx, tx, sub] WiFi CSI input

        Returns:
            dict with pose, csi_features, part_features, part_centroids, points, part_ids
        """
        csi_features = self.csi_encoder(csi)

        if self.ablation_mode == 'no_bplq':
            pose = self.direct_regressor(csi_features)
            return {
                'pose': pose,
                'csi_features': csi_features,
                'part_features': None,
                'part_centroids': None,
                'points': None,
                'part_ids': None,
            }

        points, part_ids = self.body_part_generator(csi_features)

        if self.sga_mode == 'none':
            pose = self.simple_decoder(points, part_ids)
            return {
                'pose': pose,
                'csi_features': csi_features,
                'part_features': None,
                'part_centroids': None,
                'points': points,
                'part_ids': part_ids,
            }

        if self.skeleton_transformer is not None:
            points = self.skeleton_transformer(points, part_ids)

        part_features, part_centroids = self.part_aggregator(points, part_ids)
        pose = self.joint_decoder(part_features)

        return {
            'pose': pose,
            'csi_features': csi_features,
            'part_features': part_features,
            'part_centroids': part_centroids,
            'points': points,
            'part_ids': part_ids
        }

def build_csi_structpose_v2(cfg: dict) -> CSIStructPoseV2:
    """Build CSI-StructPose V2 Phase 1 model from config.

    Args:
        cfg: Configuration dict with optional keys:
            - encoder_type: 'transformer' (default) or 'resnet34'
            - pretrained_encoder: bool, use ImageNet weights for ResNet34
            - use_improved_gcn: bool
            - ablation_mode: None or 'no_bplq'
            - sga_mode: 'skeleton', 'full_attn', or 'none'
            - ... (see CSIStructPoseV2.__init__)
    """
    return CSIStructPoseV2(
        num_frames=cfg.get('num_frames', 5),
        num_rx=cfg.get('num_rx', 3),
        num_tx=cfg.get('num_tx', 3),
        num_subcarriers=cfg.get('num_subcarriers', 114),
        embed_dim=cfg.get('embed_dim', 256),
        point_dim=cfg.get('point_dim', 256),
        num_encoder_layers=cfg.get('num_encoder_layers', 4),
        num_graph_layers=cfg.get('num_graph_layers', 4),
        num_gcn_layers=cfg.get('num_gcn_layers', 4),
        num_heads=cfg.get('num_heads', 8),
        dropout=cfg.get('dropout', 0.1),
        num_joints=cfg.get('num_joints', 17),
        use_improved_gcn=cfg.get('use_improved_gcn', True),
        use_two_stage_generator=cfg.get('use_two_stage_generator', True),
        legacy_mode=cfg.get('legacy_mode', False),
        cheby_k=cfg.get('cheby_k', 3),
        encoder_type=cfg.get('encoder_type', 'transformer'),
        pretrained_encoder=cfg.get('pretrained_encoder', True),
        ablation_mode=cfg.get('ablation_mode', None),
        sga_mode=cfg.get('sga_mode', 'skeleton'),
        in_chan_mult=cfg.get('in_chan_mult', 1),
    )

class Stage2PoseRefiner(nn.Module):
    """
    Residual pose-refinement decoder for Phase 2.

    Stage 1 produces a frozen, canonical root-relative pose. This decoder seeds
    per-joint queries from that pose, cross-attends to the (frozen) CSI encoder
    features, and predicts a per-joint residual Delta. The final pose is
    stage1_pose + root + Delta, so Stage 2 can correct *shape* (not just
    translate), lifting the absolute error below the frozen Stage-1 relative floor.

    The output projection is zero-initialized so the refiner starts as identity
    (Delta = 0) -> training begins exactly at the non-refined baseline and improves.
    """

    def __init__(self,
                 num_joints: int,
                 embed_dim: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.num_joints = num_joints
        self.joint_proj = nn.Linear(3, embed_dim)
        self.joint_id_embed = nn.Parameter(torch.randn(1, num_joints, embed_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, stage1_pose: torch.Tensor, csi_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stage1_pose: [B, J, 3] frozen relative pose (structural prior)
            csi_features: [B, L, D] frozen CSI encoder features (memory)
        Returns:
            delta: [B, J, 3] per-joint residual correction
        """
        q = self.joint_proj(stage1_pose) + self.joint_id_embed
        q = self.decoder(q, csi_features)
        q = self.norm(q)
        return self.out(q)

class Stage2IterativeRefiner(nn.Module):
    """
    Iterative residual keypoint-decoder for Phase 2 (PETR-style, adapted).

    Per-keypoint queries are seeded from the frozen Stage-1 canonical pose and
    cross-attend to the UN-POOLED CSI token sequence. Each layer predicts a small
    residual to the keypoint coordinates; the coordinate reference is detached
    between layers (iterative refinement). Every layer's pose is returned so the
    training loop can apply deep supervision.

    This replaces the pooled-MLP root head (one-shot regression from a pooled
    feature vector -> overfits) with attention-over-tokens + iterative residual
    refinement + deep supervision -- the structure that regularizes PETR.
    """

    def __init__(self,
                 num_joints: int,
                 embed_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.num_joints = num_joints
        self.num_layers = num_layers
        self.joint_proj = nn.Linear(3, embed_dim)
        self.joint_id_embed = nn.Parameter(torch.randn(1, num_joints, embed_dim) * 0.02)
        self.ref_proj = nn.Linear(3, embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
                dropout=dropout, activation='gelu', batch_first=True)
            for _ in range(num_layers)
        ])
        self.coord_heads = nn.ModuleList([nn.Linear(embed_dim, 3) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, stage1_pose: torch.Tensor, csi_features: torch.Tensor):
        """
        Args:
            stage1_pose: [B, J, 3] frozen canonical pose -> initial reference points
            csi_features: [B, L, D] frozen, UN-POOLED CSI encoder tokens (memory)
        Returns:
            list of [B, J, 3] poses, one per layer (last = final); for deep supervision
        """
        h = self.joint_proj(stage1_pose) + self.joint_id_embed
        ref = stage1_pose
        outs = []
        for layer, head in zip(self.layers, self.coord_heads):
            q = h + self.ref_proj(ref.detach())
            h = layer(q, csi_features)
            ref = ref + head(self.norm(h))
            outs.append(ref)
            ref = ref.detach()
        return outs

def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """6D rotation representation (Zhou et al. 2019) -> rotation matrix [.,3,3]."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / (a2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)

class Stage2RigidPose(nn.Module):
    """
    Rigid world-pose head for Phase 2 (replaces the ASPN translation anchor).

    A global query cross-attends to the (un-pooled) CSI tokens and predicts a
    6-DoF rigid transform (R in SO(3), t in R^3) that places the frozen canonical
    Stage-1 pose into the camera frame:  absolute = R @ stage1_pose + t.

    Rationale: Stage-1 shape is already excellent (PA-MPJPE 60.8); the entire
    remaining error is rigid placement (mostly orientation), which a single
    consistent rotation fixes far more efficiently than free per-joint residuals.
    The rigid ceiling is exactly PA-MPJPE. An optional small non-rigid residual
    cleans up what a pure rigid transform cannot.
    """

    def __init__(self,
                 num_joints: int,
                 embed_dim: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_residual: bool = False):
        super().__init__()
        self.num_joints = num_joints
        self.use_residual = use_residual
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.rot_head = nn.Linear(embed_dim, 6)
        self.trans_head = nn.Linear(embed_dim, 3)
        nn.init.zeros_(self.rot_head.weight)
        with torch.no_grad():
            self.rot_head.bias.copy_(torch.tensor([1., 0., 0., 0., 1., 0.]))
        nn.init.zeros_(self.trans_head.weight)
        nn.init.zeros_(self.trans_head.bias)
        if use_residual:
            self.refiner = Stage2PoseRefiner(
                num_joints=num_joints, embed_dim=embed_dim,
                num_layers=2, num_heads=num_heads, dropout=dropout)

    def forward(self, stage1_pose: torch.Tensor, memory: torch.Tensor):
        """
        Args:
            stage1_pose: [B, J, 3] frozen canonical pose
            memory: [B, L, D] CSI tokens (frozen shape + trainable localization)
        Returns:
            pose [B,J,3], R [B,3,3], t [B,3]
        """
        B = stage1_pose.shape[0]
        ctx = self.norm(self.decoder(self.query.expand(B, -1, -1), memory))[:, 0, :]
        R = rot6d_to_matrix(self.rot_head(ctx))
        t = self.trans_head(ctx)
        pose = torch.einsum('bij,bnj->bni', R, stage1_pose) + t.unsqueeze(1)
        if self.use_residual:
            pose = pose + self.refiner(pose, memory)
        return pose, R, t

class CSIStructPoseV2Phase2(nn.Module):
    """
    CSI-StructPose V2 Phase 2 with Learned Phase Position Predictor.

    Architecture:
        Path 1 (Part-wise, frozen):
            CSI → CSIEncoderV2 → BP-LQ → SkeletonGraphTrans
            → PartAggregator → PartToJointDecoder → relative_pose [B, J, 3]

        Path 2 (Position from raw CSI only):
            Raw CSI → LearnedPhaseAoAToFExtractor → AoA-ToF heatmap [B, 1, 90, 64]
            → HeatmapCNN → features [B, 256]
            → MLP → XY/Z Heads → root_position [B, 3]

        Output: absolute_pose = relative_pose + root_position
    """

    def __init__(self,
                 num_frames: int = 5,
                 num_rx: int = 3,
                 num_tx: int = 3,
                 num_subcarriers: int = 114,
                 embed_dim: int = 256,
                 point_dim: int = 256,
                 num_encoder_layers: int = 4,
                 num_graph_layers: int = 4,
                 num_gcn_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_joints: int = 17,
                 position_hidden_dim: int = 512,
                 use_improved_gcn: bool = True,
                 use_two_stage_generator: bool = True,
                 legacy_mode: bool = False,
                 cheby_k: int = 3,
                 aspn_mode: str = 'full',
                 in_chan_mult: int = 1,
                 use_refiner: bool = False,
                 refiner_layers: int = 3,
                 use_iter_refiner: bool = False,
                 use_dual_refiner: bool = False,
                 use_rigid: bool = False,
                 rigid_residual: bool = False):
        """
        Args:
            aspn_mode: 'full' (default), 'no_decomp', or 'no_phase' — controls ASPN variant
            in_chan_mult: per-antenna input channels (1=amplitude, 2=complex real+imag)
            use_refiner: add a single residual pose-refinement decoder (legacy)
            refiner_layers: number of transformer-decoder layers in the refiner(s)
            use_iter_refiner: replace the pooled root head with an iterative residual
                keypoint-decoder (PETR-style) + deep supervision; predicts absolute pose
                directly. When True the ASPN/pooled root head is not built.
            use_dual_refiner: keep the ASPN root anchor and add a refiner that cross-attends
                to BOTH the frozen Stage-1 shape tokens AND a trainable localization
                token-encoder (multimodal). Breaks the frozen-feature bottleneck while
                keeping the stable position anchor.
        """
        super().__init__()

        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.freeze_phase1 = True

        self.csi_encoder = CSIEncoderV2(
            num_frames=num_frames,
            num_rx=num_rx,
            num_tx=num_tx,
            num_subcarriers=num_subcarriers,
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            in_chan_mult=in_chan_mult
        )

        self.body_part_generator = BodyPartLatentQueryGenerator(
            embed_dim=embed_dim,
            num_heads=num_heads,
            point_dim=point_dim,
            dropout=dropout,
            use_two_stage=use_two_stage_generator
        )

        self.skeleton_transformer = SkeletonGraphTransformer(
            dim=point_dim,
            num_layers=num_graph_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        self.part_aggregator = PartFeatureAggregator(
            dim=point_dim,
            num_heads=num_heads // 2
        )

        self.joint_decoder = PartToJointDecoder(
            dim=point_dim,
            num_joints=num_joints,
            gcn_layers=num_gcn_layers,
            dropout=dropout,
            use_improved_gcn=use_improved_gcn,
            cheby_k=cheby_k,
            num_heads=num_heads
        )

        self.use_iter_refiner = use_iter_refiner
        self.use_rigid = use_rigid and not use_iter_refiner
        self.use_dual_refiner = use_dual_refiner and not use_iter_refiner and not self.use_rigid
        self.use_refiner = use_refiner and not use_iter_refiner and not self.use_rigid and not self.use_dual_refiner

        if use_iter_refiner:
            self.position_predictor = None
            self.iter_refiner = Stage2IterativeRefiner(
                num_joints=num_joints,
                embed_dim=embed_dim,
                num_layers=refiner_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        elif self.use_rigid:
            self.position_predictor = None
            self.loc_encoder = CSIEncoderV2(
                num_frames=num_frames,
                num_rx=num_rx,
                num_tx=num_tx,
                num_subcarriers=num_subcarriers,
                embed_dim=embed_dim,
                num_layers=num_encoder_layers,
                num_heads=num_heads,
                dropout=dropout,
                in_chan_mult=in_chan_mult,
            )
            self.rigid_pose = Stage2RigidPose(
                num_joints=num_joints,
                embed_dim=embed_dim,
                num_layers=refiner_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_residual=rigid_residual,
            )
        else:
            from .physics_position import LearnedPhasePositionPredictor
            self.position_predictor = LearnedPhasePositionPredictor(
                csi_dim=embed_dim,
                num_rx=num_rx,
                num_tx=num_tx,
                num_subcarriers=num_subcarriers,
                num_frames=num_frames,
                hidden_dim=position_hidden_dim,
                aspn_mode=aspn_mode,
                in_chan_mult=in_chan_mult,
            )
            if self.use_dual_refiner:
                self.loc_encoder = CSIEncoderV2(
                    num_frames=num_frames,
                    num_rx=num_rx,
                    num_tx=num_tx,
                    num_subcarriers=num_subcarriers,
                    embed_dim=embed_dim,
                    num_layers=num_encoder_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                    in_chan_mult=in_chan_mult,
                )
                self.pose_refiner = Stage2PoseRefiner(
                    num_joints=num_joints,
                    embed_dim=embed_dim,
                    num_layers=refiner_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            elif self.use_refiner:
                self.pose_refiner = Stage2PoseRefiner(
                    num_joints=num_joints,
                    embed_dim=embed_dim,
                    num_layers=refiner_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                )

    def forward(self, csi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            csi: [B, T, rx, tx, sub] WiFi CSI input (amplitude)

        Returns:
            dict with pose, phase1_pose, root_position, aoa_tof_heatmap
        """
        if getattr(self, 'freeze_phase1', True):
            self.csi_encoder.eval()
            self.body_part_generator.eval()
            self.skeleton_transformer.eval()
            self.part_aggregator.eval()
            self.joint_decoder.eval()
            with torch.no_grad():
                csi_features = self.csi_encoder(csi)
                points, part_ids = self.body_part_generator(csi_features)
                points = self.skeleton_transformer(points, part_ids)
                part_features, part_centroids = self.part_aggregator(points, part_ids)
                phase1_pose = self.joint_decoder(part_features)
        else:
            csi_features = self.csi_encoder(csi)
            points, part_ids = self.body_part_generator(csi_features)
            points = self.skeleton_transformer(points, part_ids)
            part_features, part_centroids = self.part_aggregator(points, part_ids)
            phase1_pose = self.joint_decoder(part_features)

        phase1_pose = phase1_pose - phase1_pose[:, 0:1, :]

        if self.use_iter_refiner:
            aux_poses = self.iter_refiner(phase1_pose, csi_features)
            pose = aux_poses[-1]
            if self.num_joints == 14:
                root_position = (pose[:, 5, :] + pose[:, 7, :]) / 2
            else:
                root_position = pose[:, 0, :]
            return {
                'pose': pose,
                'phase1_pose': phase1_pose,
                'root_position': root_position,
                'aux_poses': aux_poses,
                'refine_delta': None,
                'aoa_tof_heatmap': None,
            }

        if self.use_rigid:
            loc_tokens = self.loc_encoder(csi)
            memory = torch.cat([csi_features, loc_tokens], dim=1)
            pose, R, t = self.rigid_pose(phase1_pose, memory)
            return {
                'pose': pose,
                'phase1_pose': phase1_pose,
                'root_position': t,
                'rigid_R': R,
                'refine_delta': None,
                'aoa_tof_heatmap': None,
            }

        position_output = self.position_predictor(csi)
        root_position = position_output['position']

        pose = phase1_pose + root_position.unsqueeze(1)

        delta = None
        if self.use_dual_refiner:
            loc_tokens = self.loc_encoder(csi)
            memory = torch.cat([csi_features, loc_tokens], dim=1)
            delta = self.pose_refiner(phase1_pose, memory)
            pose = pose + delta
        elif self.use_refiner:
            delta = self.pose_refiner(phase1_pose, csi_features)
            pose = pose + delta

        return {
            'pose': pose,
            'phase1_pose': phase1_pose,
            'root_position': root_position,
            'refine_delta': delta,
            'aoa_tof_heatmap': position_output.get('aoa_tof_heatmap'),
        }

def build_csi_structpose_v2_phase2(cfg: dict) -> CSIStructPoseV2Phase2:
    """Build CSI-StructPose V2 Phase 2 with learned phase position predictor."""
    return CSIStructPoseV2Phase2(
        num_frames=cfg.get('num_frames', 5),
        num_rx=cfg.get('num_rx', 3),
        num_tx=cfg.get('num_tx', 3),
        num_subcarriers=cfg.get('num_subcarriers', 114),
        embed_dim=cfg.get('embed_dim', 256),
        point_dim=cfg.get('point_dim', 256),
        num_encoder_layers=cfg.get('num_encoder_layers', 4),
        num_graph_layers=cfg.get('num_graph_layers', 4),
        num_gcn_layers=cfg.get('num_gcn_layers', 4),
        num_heads=cfg.get('num_heads', 8),
        dropout=cfg.get('dropout', 0.1),
        num_joints=cfg.get('num_joints', 17),
        position_hidden_dim=cfg.get('position_hidden_dim', 512),
        use_improved_gcn=cfg.get('use_improved_gcn', True),
        use_two_stage_generator=cfg.get('use_two_stage_generator', True),
        legacy_mode=cfg.get('legacy_mode', False),
        cheby_k=cfg.get('cheby_k', 3),
        aspn_mode=cfg.get('aspn_mode', 'full'),
        in_chan_mult=cfg.get('in_chan_mult', 1),
        use_refiner=cfg.get('use_refiner', False),
        refiner_layers=cfg.get('refiner_layers', 3),
        use_iter_refiner=cfg.get('use_iter_refiner', False),
        use_dual_refiner=cfg.get('use_dual_refiner', False),
        use_rigid=cfg.get('use_rigid', False),
        rigid_residual=cfg.get('rigid_residual', False),
    )

def load_phase1_into_phase2(phase2_model: CSIStructPoseV2Phase2,
                                     phase1_checkpoint_path: str,
                                     device: torch.device,
                                     freeze_phase1: bool = True) -> CSIStructPoseV2Phase2:
    """Load Phase 1 weights into Phase 2 Physics model and optionally freeze them."""
    print(f"Loading Phase 1 checkpoint: {phase1_checkpoint_path}")
    checkpoint = torch.load(phase1_checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    phase1_modules = [
        'csi_encoder', 'body_part_generator', 'skeleton_transformer',
        'part_aggregator', 'joint_decoder'
    ]

    phase1_state_dict = {}
    for key, value in state_dict.items():
        for module in phase1_modules:
            if key.startswith(module):
                phase1_state_dict[key] = value
                break

    if hasattr(phase2_model, 'loc_encoder'):
        loc_state = {k.replace('csi_encoder.', 'loc_encoder.', 1): v
                     for k, v in phase1_state_dict.items() if k.startswith('csi_encoder.')}
        phase1_state_dict = {**phase1_state_dict, **loc_state}
        print(f"  Warm-started loc_encoder from Stage-1 encoder ({len(loc_state)} tensors)")

    missing, unexpected = phase2_model.load_state_dict(phase1_state_dict, strict=False)

    _trainable_prefixes = ('position_predictor', 'pose_refiner', 'iter_refiner', 'loc_encoder', 'rigid_pose')
    expected_missing = [k for k in missing if k.startswith(_trainable_prefixes)]
    actual_missing = [k for k in missing if not k.startswith(_trainable_prefixes)]

    if actual_missing:
        print(f"  WARNING: Missing Phase 1 keys: {actual_missing[:5]}...")
    print(f"  Loaded Phase 1 modules: {len(phase1_state_dict)} parameters")
    print(f"  Position predictor to train: {len(expected_missing)} parameters")

    phase2_model.freeze_phase1 = freeze_phase1

    if freeze_phase1:
        frozen_count = 0
        trainable_count = 0

        for name, param in phase2_model.named_parameters():
            is_phase1 = any(name.startswith(m) for m in phase1_modules)
            if is_phase1:
                param.requires_grad = False
                frozen_count += param.numel()
            else:
                trainable_count += param.numel()

        print(f"  Frozen (Phase 1): {frozen_count:,} parameters")
        print(f"  Trainable (Position Predictor): {trainable_count:,} parameters")
    else:
        print(f"  Joint fine-tune: Stage 1 UNFROZEN (gradients flow into Path 1)")

    return phase2_model
