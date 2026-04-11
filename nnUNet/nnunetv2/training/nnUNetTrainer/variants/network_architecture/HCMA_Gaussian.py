from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA import HCMA


def _gaussian_kernel1d(kernel_size: int, sigma: float) -> torch.Tensor:
    center = (kernel_size - 1) / 2.0
    xs = torch.arange(kernel_size, dtype=torch.float32)
    kernel = torch.exp(-0.5 * ((xs - center) / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_kernel3d(kernel_size: int, sigma: float) -> torch.Tensor:
    k1 = _gaussian_kernel1d(kernel_size, sigma)
    k3 = torch.einsum("i,j,k->ijk", k1, k1, k1)
    k3 = k3 / k3.sum()
    return k3


class _GaussianBranch3D(nn.Module):
    """Depthwise fixed 3D Gaussian filtering branch."""

    def __init__(self, channels: int, kernel_size: int, sigma: float) -> None:
        super().__init__()
        self.filter = nn.Conv3d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        kernel = _gaussian_kernel3d(kernel_size, sigma)
        with torch.no_grad():
            self.filter.weight.copy_(kernel.view(1, 1, kernel_size, kernel_size, kernel_size).repeat(channels, 1, 1, 1, 1))
        self.filter.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.filter(x)


class GaussianSupplement3D(nn.Module):
    """Multi-scale 3D Gaussian feature supplement for small-target enhancement."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 32,
        sigmas: Sequence[float] = (0.8, 1.4, 2.0),
        kernel_sizes: Sequence[int] = (3, 5, 7),
        alpha: float = 0.25,
    ) -> None:
        super().__init__()
        assert len(sigmas) == len(kernel_sizes), "sigmas and kernel_sizes must have same length"

        self.pre = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.branches = nn.ModuleList(
            [_GaussianBranch3D(hidden_channels, int(k), float(s)) for k, s in zip(kernel_sizes, sigmas)]
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(hidden_channels * (len(self.branches) + 1), hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.delta_head = nn.Conv3d(hidden_channels, num_classes, kernel_size=1, bias=True)
        self.logit_gate = nn.Sequential(
            nn.Conv3d(num_classes, num_classes, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))

    def forward(self, feat: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        z = self.pre(feat)
        pyramids: List[torch.Tensor] = [z]
        for branch in self.branches:
            pyramids.append(branch(z))
        g = self.fuse(torch.cat(pyramids, dim=1))

        delta = self.delta_head(g)
        gate = self.logit_gate(logits)
        return logits + self.alpha * gate * delta


class HCMA_Gaussian(nn.Module):
    """HCMA + GaussianSupplement3D for small-target segmentation."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        input_size: Tuple[int, int, int] = (64, 128, 128),
        predict_mode: bool = False,
    ) -> None:
        super().__init__()
        self.predict_mode = predict_mode
        self.enable_foreground_rescue = True
        self.min_foreground_ratio = 0.001
        self.target_foreground_ratio = 0.004
        self.max_foreground_boost = 3.0

        self.backbone = HCMA(
            in_channels=in_channels,
            n_classes=n_classes,
            patch_ini=list(input_size),
            predict_mode=False,
            use_small_lesion_refine=True,
        )
        self.gaussian = GaussianSupplement3D(
            in_channels=32,
            num_classes=n_classes,
            hidden_channels=32,
            sigmas=(0.8, 1.4, 2.0),
            kernel_sizes=(3, 5, 7),
            alpha=0.25,
        )

    def _foreground_rescue_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if (not self.enable_foreground_rescue) or logits.ndim != 5 or logits.shape[1] != 2:
            return logits

        adjusted = logits.clone()
        eps = 1e-6
        for b in range(adjusted.shape[0]):
            margin = adjusted[b, 1] - adjusted[b, 0]
            pred_fg_ratio = float((margin > 0).float().mean().item())
            if pred_fg_ratio >= self.min_foreground_ratio:
                continue

            flat_margin = margin.reshape(-1).float().detach()
            q = torch.quantile(flat_margin, 1.0 - self.target_foreground_ratio)
            boost = float((-q + eps).item())
            if boost > 0:
                boost = min(boost, self.max_foreground_boost)
                adjusted[b, 1] = adjusted[b, 1] + boost
        return adjusted

    def forward(self, x: torch.Tensor):
        feat, logits = self.backbone(x)
        logits = self.gaussian(feat, logits)
        if self.predict_mode:
            logits = self._foreground_rescue_logits(logits)
            return logits
        return feat, logits
