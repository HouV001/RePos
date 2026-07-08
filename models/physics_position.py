"""
Position Predictor for WiFi CSI-based Pose Estimation

Uses learned phase estimation from amplitude CSI to compute AoA-ToF features,
then predicts root position through CNN + fusion architecture.

Reference:
- DLoc (MobiCom 2020): AoA-ToF heatmap + CNN for localization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class HeatmapCNN(nn.Module):
    """CNN to process AoA-ToF heatmap for position estimation."""

    def __init__(self, num_aoa_bins: int = 90, num_tof_bins: int = 64, embed_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        x = self.encoder(heatmap)
        x = self.pool(x)
        x = self.proj(x)
        return x

class LearnedPhaseAoAToFExtractor(nn.Module):
    """
    AoA-ToF extractor with learnable phase estimation from amplitude CSI.

    Key insight: Learn to estimate phase-like features from amplitude,
    then apply correct AoA-ToF processing (per-subcarrier beamforming + IFFT).

    Architecture:
        CSI Amp [B, T, RX, TX, Sub]
            ↓
        Phase Estimator → phase_est [B, T, RX*TX, Sub]
            ↓
        Construct complex CSI: amp * exp(j * phase_est * π)
            ↓
        Per-subcarrier beamforming (keep Sub dimension)
            ↓
        aoa_sub [B, T, num_aoa, Sub]
            ↓
        IFFT along Sub dimension
            ↓
        aoa_tof [B, T, num_aoa, num_tof] (true 2D heatmap)
            ↓
        Time average → [B, num_aoa, num_tof]
    """

    def __init__(self,
                 num_rx: int = 3,
                 num_tx: int = 3,
                 num_subcarriers: int = 114,
                 num_frames: int = 5,
                 antenna_spacing: float = 0.025,
                 subcarrier_spacing: float = 312.5e3,
                 center_freq: float = 5.8e9,
                 num_aoa_bins: int = 90,
                 num_tof_bins: int = 64):
        super().__init__()

        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_subcarriers = num_subcarriers
        self.num_frames = num_frames
        self.antenna_spacing = antenna_spacing
        self.subcarrier_spacing = subcarrier_spacing
        self.center_freq = center_freq
        self.wavelength = 3e8 / center_freq

        self.num_aoa_bins = num_aoa_bins
        self.num_tof_bins = num_tof_bins

        self.phase_estimator = nn.Sequential(
            nn.Conv2d(num_rx * num_tx, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_rx * num_tx, kernel_size=3, padding=1),
            nn.Tanh()
        )

        angles = torch.linspace(-np.pi/2, np.pi/2, num_aoa_bins)
        phase_shifts = 2 * np.pi * antenna_spacing * torch.sin(angles) / self.wavelength
        rx_indices = torch.arange(num_rx).float()
        steering = torch.exp(1j * torch.outer(rx_indices, phase_shifts))
        self.register_buffer('steering_vectors', torch.view_as_real(steering))
        self.register_buffer('aoa_angles', angles)

        max_delay = num_subcarriers / (subcarrier_spacing * num_subcarriers)
        tof_delays = torch.linspace(0, max_delay, num_tof_bins)
        self.register_buffer('tof_delays', tof_delays)

    def forward(self, csi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract AoA-ToF features with correct physics.

        Args:
            csi: [B, T, RX, TX, Sub] amplitude CSI

        Returns:
            aoa_tof_heatmap: [B, 1, num_aoa_bins, num_tof_bins] - true 2D heatmap
            aoa_profile: [B, num_aoa_bins] - marginalized AoA
            tof_profile: [B, num_tof_bins] - marginalized ToF
        """
        B, T, RX, TX, Sub = csi.shape

        csi_2d = csi.permute(0, 2, 3, 4, 1).reshape(B, RX * TX, Sub, T)
        phase_est = self.phase_estimator(csi_2d)
        phase_est = phase_est.permute(0, 3, 1, 2)
        phase_est = phase_est.view(B, T, RX, TX, Sub)

        csi_complex = csi * torch.exp(1j * phase_est * np.pi)

        csi_rx = csi_complex.mean(dim=3)

        steering = torch.view_as_complex(self.steering_vectors)
        csi_rx_flat = csi_rx.permute(0, 1, 3, 2).reshape(B * T, Sub, RX)
        aoa_sub = torch.matmul(csi_rx_flat, steering.conj())
        aoa_sub = aoa_sub.view(B, T, Sub, self.num_aoa_bins)
        aoa_sub = aoa_sub.permute(0, 1, 3, 2)

        pad_size = self.num_tof_bins - Sub
        if pad_size > 0:
            aoa_sub_padded = F.pad(aoa_sub, (0, pad_size))
        else:
            aoa_sub_padded = aoa_sub[..., :self.num_tof_bins]

        aoa_tof = torch.fft.ifft(aoa_sub_padded, dim=-1)
        aoa_tof = torch.abs(aoa_tof) ** 2

        aoa_tof_heatmap = aoa_tof.mean(dim=1)

        heatmap_sum = aoa_tof_heatmap.sum(dim=(-2, -1), keepdim=True) + 1e-8
        aoa_tof_heatmap = aoa_tof_heatmap / heatmap_sum

        aoa_profile = aoa_tof_heatmap.sum(dim=-1)
        tof_profile = aoa_tof_heatmap.sum(dim=-2)

        aoa_tof_heatmap = aoa_tof_heatmap.unsqueeze(1)

        return aoa_tof_heatmap, aoa_profile, tof_profile

class DirectAmplitudeCNN(nn.Module):
    """
    Ablation: amplitude CSI → CNN → features, no phase estimation or spatial decomposition.

    Used when --aspn_mode no_phase to test the contribution of learned phase
    and spatial decomposition.
    """

    def __init__(self,
                 num_antenna_pairs: int = 9,
                 num_subcarriers: int = 114,
                 num_frames: int = 5,
                 embed_dim: int = 256):
        super().__init__()

        self.num_antenna_pairs = num_antenna_pairs
        self.num_subcarriers = num_subcarriers
        self.num_frames = num_frames

        self.encoder = nn.Sequential(
            nn.Conv2d(num_antenna_pairs, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            csi: [B, T, RX, TX, Sub] amplitude CSI
        Returns:
            features: [B, embed_dim]
        """
        B, T, RX, TX, Sub = csi.shape
        x = csi.permute(0, 2, 3, 4, 1).reshape(B, RX * TX, Sub, T)
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

class DirectComplexCNN(nn.Module):
    """
    Ablation: learned phase + complex CSI → CNN → features, skip spatial decomposition.

    Used when --aspn_mode no_decomp to test the contribution of the differentiable
    AoA-ToF spatial decomposition layer.
    """

    def __init__(self,
                 num_rx: int = 3,
                 num_tx: int = 3,
                 num_subcarriers: int = 114,
                 num_frames: int = 5,
                 embed_dim: int = 256,
                 in_chan_mult: int = 1):
        super().__init__()

        self.num_rx = num_rx
        self.num_tx = num_tx
        self.in_chan_mult = in_chan_mult
        num_antenna_pairs = num_rx * num_tx
        in_ch = num_antenna_pairs * in_chan_mult

        if in_chan_mult == 1:
            self.phase_estimator = nn.Sequential(
                nn.Conv2d(num_antenna_pairs, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, num_antenna_pairs, kernel_size=3, padding=1),
                nn.Tanh(),
            )
        else:
            self.phase_estimator = None

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            csi: [B, T, RX, TX, Sub] amplitude CSI, or
                 [B, T, RX, TX, Sub, C] multi-channel (e.g. real+imag complex).
        Returns:
            features: [B, embed_dim]
        """
        if csi.dim() == 6:
            B, T, RX, TX, Sub, C = csi.shape
            x = csi.permute(0, 2, 3, 5, 4, 1).reshape(B, RX * TX * C, Sub, T)
        else:
            B, T, RX, TX, Sub = csi.shape
            csi_2d = csi.permute(0, 2, 3, 4, 1).reshape(B, RX * TX, Sub, T)
            phase_est = self.phase_estimator(csi_2d)
            csi_complex = csi_2d * torch.exp(1j * phase_est * np.pi)
            x = torch.abs(csi_complex).float()

        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)

class LearnedPhasePositionPredictor(nn.Module):
    """
    Position Predictor using Learned Phase AoA-ToF extraction.

    Architecture:
        Raw CSI → LearnedPhaseAoAToFExtractor → AoA-ToF heatmap [B, 1, 90, 64]
            ↓
        HeatmapCNN → features [B, 256]
            ↓
        MLP → XY Head [B, 2] + Z Head [B, 1]
            ↓
        position = [xy, z]
    """

    def __init__(self,
                 csi_dim: int = 256,
                 num_rx: int = 3,
                 num_tx: int = 3,
                 num_subcarriers: int = 114,
                 num_frames: int = 5,
                 num_aoa_bins: int = 90,
                 num_tof_bins: int = 64,
                 hidden_dim: int = 512,
                 aspn_mode: str = 'full',
                 in_chan_mult: int = 1):
        """
        Args:
            aspn_mode: 'full' (default: learned phase + spatial decomposition + heatmap CNN),
                      'no_decomp' (learned phase + direct CNN, skip spatial decomposition),
                      'no_phase' (amplitude-only CNN, no phase or decomposition)
            in_chan_mult: per-antenna input channels (1=amplitude, 2=complex real+imag)
        """
        super().__init__()

        self.csi_dim = csi_dim
        self.aspn_mode = aspn_mode
        self.in_chan_mult = in_chan_mult

        if aspn_mode == 'full':
            self.aoa_tof_extractor = LearnedPhaseAoAToFExtractor(
                num_rx=num_rx,
                num_tx=num_tx,
                num_subcarriers=num_subcarriers,
                num_frames=num_frames,
                num_aoa_bins=num_aoa_bins,
                num_tof_bins=num_tof_bins,
            )
            self.heatmap_cnn = HeatmapCNN(
                num_aoa_bins=num_aoa_bins,
                num_tof_bins=num_tof_bins,
                embed_dim=csi_dim
            )
        elif aspn_mode == 'no_decomp':
            self.direct_complex_cnn = DirectComplexCNN(
                num_rx=num_rx,
                num_tx=num_tx,
                num_subcarriers=num_subcarriers,
                num_frames=num_frames,
                embed_dim=csi_dim,
                in_chan_mult=in_chan_mult,
            )
        elif aspn_mode == 'no_phase':
            self.direct_amplitude_cnn = DirectAmplitudeCNN(
                num_antenna_pairs=num_rx * num_tx,
                num_subcarriers=num_subcarriers,
                num_frames=num_frames,
                embed_dim=csi_dim,
            )
        else:
            raise ValueError(f"Unknown aspn_mode: {aspn_mode}")

        self.mlp = nn.Sequential(
            nn.Linear(csi_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.xy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

        self.z_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, csi: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            csi: [B, T, RX, TX, Sub] amplitude CSI

        Returns:
            dict with position, aoa_tof_heatmap (None for ablation modes)
        """
        aoa_tof_heatmap = None

        if self.aspn_mode == 'full':
            aoa_tof_heatmap, aoa_profile, tof_profile = self.aoa_tof_extractor(csi)
            features = self.heatmap_cnn(aoa_tof_heatmap)
        elif self.aspn_mode == 'no_decomp':
            features = self.direct_complex_cnn(csi)
        elif self.aspn_mode == 'no_phase':
            features = self.direct_amplitude_cnn(csi)

        features = self.mlp(features)
        xy = self.xy_head(features)
        z = self.z_head(features)
        position = torch.cat([xy, z], dim=-1)

        return {
            'position': position,
            'aoa_tof_heatmap': aoa_tof_heatmap,
        }

def build_learned_phase_position_predictor(cfg: dict) -> LearnedPhasePositionPredictor:
    """Build learned phase position predictor."""
    return LearnedPhasePositionPredictor(
        csi_dim=cfg.get('embed_dim', 256),
        num_rx=cfg.get('num_rx', 3),
        num_tx=cfg.get('num_tx', 3),
        num_subcarriers=cfg.get('num_subcarriers', 114),
        num_frames=cfg.get('num_frames', 5),
        num_aoa_bins=cfg.get('num_aoa_bins', 90),
        num_tof_bins=cfg.get('num_tof_bins', 64),
        hidden_dim=cfg.get('position_hidden_dim', 512),
        aspn_mode=cfg.get('aspn_mode', 'full'),
    )

class AoAToFExtractor(nn.Module):
    """
    AoA-ToF feature extractor for complex CSI (with real phase information).

    Used by PhysicsGuidedPositionPredictor for WiFiPose3D dataset.
    """

    def __init__(self,
                 num_rx: int = 3,
                 num_tx: int = 3,
                 num_subcarriers: int = 114,
                 num_frames: int = 5,
                 antenna_spacing: float = 0.025,
                 subcarrier_spacing: float = 312.5e3,
                 center_freq: float = 5.8e9,
                 num_aoa_bins: int = 90,
                 num_tof_bins: int = 64):
        super().__init__()

        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_subcarriers = num_subcarriers
        self.num_frames = num_frames
        self.antenna_spacing = antenna_spacing
        self.subcarrier_spacing = subcarrier_spacing
        self.center_freq = center_freq
        self.wavelength = 3e8 / center_freq

        self.num_aoa_bins = num_aoa_bins
        self.num_tof_bins = num_tof_bins

        angles = torch.linspace(-np.pi/2, np.pi/2, num_aoa_bins)
        phase_shifts = 2 * np.pi * antenna_spacing * torch.sin(angles) / self.wavelength
        rx_indices = torch.arange(num_rx).float()
        steering = torch.exp(1j * torch.outer(rx_indices, phase_shifts))
        self.register_buffer('steering_vectors', torch.view_as_real(steering))
        self.register_buffer('aoa_angles', angles)

        max_delay = num_subcarriers / (subcarrier_spacing * num_subcarriers)
        tof_delays = torch.linspace(0, max_delay, num_tof_bins)
        self.register_buffer('tof_delays', tof_delays)

    def compute_cir(self, csi_complex: torch.Tensor) -> torch.Tensor:
        """Compute Channel Impulse Response via IFFT."""
        B, T, RX, TX, Sub = csi_complex.shape
        pad_size = self.num_tof_bins
        csi_padded = F.pad(csi_complex, (0, pad_size - Sub % pad_size))
        cir = torch.fft.ifft(csi_padded, dim=-1).abs()
        cir = cir[..., :self.num_tof_bins]
        return cir

    def compute_aoa_profile(self, csi_complex: torch.Tensor) -> torch.Tensor:
        """Compute AoA profile using beamforming."""
        B, T, RX, TX, Sub = csi_complex.shape
        csi_rx = csi_complex.mean(dim=(1, 3, 4))
        steering = torch.view_as_complex(self.steering_vectors)
        aoa_profile = torch.abs(csi_rx @ steering.conj()) ** 2
        aoa_profile = aoa_profile / (aoa_profile.sum(dim=-1, keepdim=True) + 1e-8)
        return aoa_profile

    def forward(self, csi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract AoA-ToF features from complex CSI.

        Args:
            csi: [B, T, RX, TX, Sub] complex CSI (or [B, T, RX, TX, Sub, 2] for real/imag)

        Returns:
            aoa_tof_heatmap: [B, 1, num_aoa_bins, num_tof_bins]
            aoa_profile: [B, num_aoa_bins]
            tof_profile: [B, num_tof_bins]
        """
        if csi.dim() == 6:
            csi = torch.view_as_complex(csi)

        B = csi.shape[0]

        cir = self.compute_cir(csi)
        tof_profile = cir.mean(dim=(1, 2, 3))
        tof_profile = tof_profile / (tof_profile.sum(dim=-1, keepdim=True) + 1e-8)

        aoa_profile = self.compute_aoa_profile(csi)

        aoa_tof_heatmap = torch.einsum('ba,bt->bat', aoa_profile, tof_profile)
        aoa_tof_heatmap = aoa_tof_heatmap.unsqueeze(1)

        return aoa_tof_heatmap, aoa_profile, tof_profile

class PhysicsGuidedPositionPredictor(nn.Module):
    """
    Position Predictor for complex CSI datasets (WiFiPose3D).

    Uses real phase information for accurate AoA-ToF extraction.
    Used by V1 (CSIStructPose) and V3 (CSIStructPoseV3) models.
    """

    def __init__(self,
                 csi_dim: int = 256,
                 num_rx: int = 3,
                 num_tx: int = 3,
                 num_subcarriers: int = 114,
                 num_frames: int = 5,
                 num_aoa_bins: int = 90,
                 num_tof_bins: int = 64,
                 hidden_dim: int = 512):
        super().__init__()

        self.csi_dim = csi_dim

        self.aoa_tof_extractor = AoAToFExtractor(
            num_rx=num_rx,
            num_tx=num_tx,
            num_subcarriers=num_subcarriers,
            num_frames=num_frames,
            num_aoa_bins=num_aoa_bins,
            num_tof_bins=num_tof_bins,
        )

        self.heatmap_cnn = HeatmapCNN(
            num_aoa_bins=num_aoa_bins,
            num_tof_bins=num_tof_bins,
            embed_dim=csi_dim
        )

        self.csi_pool = nn.Sequential(
            nn.Linear(csi_dim, csi_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(csi_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.xy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

        self.z_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.distance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )

    def forward(self, csi: torch.Tensor, csi_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            csi: [B, T, RX, TX, Sub] or [B, T, RX, TX, Sub, 2] complex CSI
            csi_features: [B, L, csi_dim] encoded CSI features

        Returns:
            dict with position, aoa_tof_heatmap, distance
        """
        aoa_tof_heatmap, aoa_profile, tof_profile = self.aoa_tof_extractor(csi)

        physics_features = self.heatmap_cnn(aoa_tof_heatmap)

        csi_pooled = csi_features.mean(dim=1)
        csi_pooled = self.csi_pool(csi_pooled)

        combined = torch.cat([physics_features, csi_pooled], dim=-1)
        fused = self.fusion(combined)

        xy = self.xy_head(fused)
        z = self.z_head(fused)
        position = torch.cat([xy, z], dim=-1)
        distance = self.distance_head(fused)

        return {
            'position': position,
            'aoa_tof_heatmap': aoa_tof_heatmap,
            'aoa_profile': aoa_profile,
            'tof_profile': tof_profile,
            'distance': distance,
        }
