"""
CSI-StructPose Model Components

Phase 1: Root-relative pose estimation with body-part latent queries
Phase 2: Absolute pose estimation with amplitude-based position prediction
"""

from .csi_structpose_v2 import (
    CSIStructPoseV2,
    build_csi_structpose_v2,
    CSIEncoderV2,
    BodyPartLatentQueryGenerator,
    SkeletonGraphAttention,
    SkeletonGraphTransformer,
    PartFeatureAggregator,
    PartToJointDecoder,
    CSIStructPoseV2Phase2,
    build_csi_structpose_v2_phase2,
    load_phase1_into_phase2,
)

from .skeleton_gcn import (
    build_adjacency_matrix,
    SKELETON_14,
    SKELETON_17,
)

from .physics_position import (
    LearnedPhaseAoAToFExtractor,
    LearnedPhasePositionPredictor,
    build_learned_phase_position_predictor,
    AoAToFExtractor,
    PhysicsGuidedPositionPredictor,
    HeatmapCNN,
)

from .losses import (
    chamfer_loss,
    compute_mpjpe,
    compute_pampjpe,
    LimbLengthLoss,
)

__all__ = [
    'CSIStructPoseV2',
    'build_csi_structpose_v2',
    'CSIEncoderV2',
    'BodyPartLatentQueryGenerator',
    'SkeletonGraphAttention',
    'SkeletonGraphTransformer',
    'PartFeatureAggregator',
    'PartToJointDecoder',
    'CSIStructPoseV2Phase2',
    'build_csi_structpose_v2_phase2',
    'load_phase1_into_phase2',
    'build_adjacency_matrix',
    'SKELETON_14',
    'SKELETON_17',
    'LearnedPhaseAoAToFExtractor',
    'LearnedPhasePositionPredictor',
    'build_learned_phase_position_predictor',
    'AoAToFExtractor',
    'PhysicsGuidedPositionPredictor',
    'HeatmapCNN',
    'chamfer_loss',
    'compute_mpjpe',
    'compute_pampjpe',
    'LimbLengthLoss',
]
