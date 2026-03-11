"""HCMA_SvANet_v2.py — Optimised HCMA + Scale-Variant Attention network.

Improvements over HCMA_SvANet.py
----------------------------------
[1] ScaleVariantAttention3DV2
    • Dual-pool SE: parallel AvgPool + MaxPool statistics summed before
      excitation — captures both mean-channel and peak-response information.
    • InstanceNorm inserted after the first 1×1 conv → training stability.
    • Inference: averages responses over ALL pool resolutions to close the
      train/test gap. V1 degraded to a single k=1 global average pool at
      test time, discarding all multi-scale capacity learned during training.

[2] Bottleneck double-attention eliminated
    • Bug: sv_attn_enc[depth] AND sv_attn_dec[0] both acted on the same
      bottleneck feature back-to-back, wasting parameters and distorting
      gradient flow.
    • Fix: one dedicated `sv_attn_bottleneck` is applied once in the encoder;
      `sv_attn_dec` (now depth modules, not depth+1) fires only after each
      decoder Up block.

[3] UpV2 — checkerboard-free decoder block
    • nn.ConvTranspose3d → trilinear Upsample + Conv3d 3×3×3 + IN + LReLU.
    • Eliminates the aliasing / checkerboard artefacts that arise from strided
      transposed convolutions, which degrades boundary segmentation quality.
    • [NEW] Dynamic spatial alignment to handle stride=1 or rounding mismatches.

[4] Parameterised input_size
    • `patch_ini` is no longer hardcoded as [64, 192, 128].
    • Pass `input_size=(D, H, W)` to the constructor (default kept for
      backward compatibility with the original HCMA patch schedule).

[5] Optional lightweight decoder Mamba  (use_decoder_mamba=True)
    • Inserts a MambaLayer after each Up block for long-range contextual
      modelling that mirrors the Mamba augmentation already present in the
      encoder Down blocks.
"""

import random
from typing import Iterable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA import (
    DenseConv,
    Down,
    MambaLayer,
    Out,
)


# ─────────────────────────────────────────────────────────────────────────────
# [1] Improved Scale-Variant Channel Attention
# ─────────────────────────────────────────────────────────────────────────────

class ScaleVariantAttention3DV2(nn.Module):
    """Improved 3-D scale-variant channel attention (V2).

    Changes vs V1
    -------------
    * Dual-branch statistics: AvgPool + MaxPool are summed before excitation,
      providing both mean-channel and peak-response information.
    * InstanceNorm inserted after the first 1×1 conv to stabilise training.
    * At inference time all pool resolutions are used and their context vectors
      are averaged, fully exploiting the multi-scale capacity learned during
      training (V1 degraded to k=1 global average pool at test time).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        squeeze_factor: int = 4,
        pool_res: Iterable[int] = (1, 2, 3),
        act: Type[nn.Module] = nn.ReLU,
        scale_act: Type[nn.Module] = nn.Sigmoid,
        moc_order: bool = True,
    ) -> None:
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(in_channels // squeeze_factor, 8)

        pool_res = list(pool_res)
        if 1 not in pool_res:
            pool_res.append(1)
        self.pool_res: List[int] = pool_res
        self.moc_order = moc_order

        # Dual-pool branches: AvgPool + MaxPool
        self.pools_avg = nn.ModuleDict(
            {str(k): nn.AdaptiveAvgPool3d((k, k, k)) for k in self.pool_res}
        )
        self.pools_max = nn.ModuleDict(
            {str(k): nn.AdaptiveMaxPool3d((k, k, k)) for k in self.pool_res}
        )

        # SE bottleneck with BatchNorm for training stability
        self.se = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_channels, affine=True),
            act(),
            nn.Conv3d(hidden_channels, in_channels, kernel_size=1, bias=True),
            scale_act(),
        )

    def _aggregate_to_scalar(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Reduce spatial dims to (1,1,1) using avg+max dual-pool at scale k."""
        feat = self.pools_avg[str(k)](x) + self.pools_max[str(k)](x)
        if feat.shape[-3:] != torch.Size([1, 1, 1]):
            b, c, d, h, w = feat.shape
            feat = feat.view(b, c, -1).mean(dim=-1, keepdim=True).view(b, c, 1, 1, 1)
        return feat

    def _sample_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            k = random.choice(self.pool_res)
            x1 = x
            if self.moc_order:
                idx = torch.randperm(x.shape[1], device=x.device)
                x1 = x[:, idx, ...]
            return self._aggregate_to_scalar(x1, k)
        else:
            # Average over ALL scales to close the train/test gap
            vectors = [self._aggregate_to_scalar(x, k) for k in self.pool_res]
            return torch.stack(vectors, dim=0).mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(self._sample_pool(x))


# ─────────────────────────────────────────────────────────────────────────────
# [3] Checkerboard-free Decoder Up Block
# ─────────────────────────────────────────────────────────────────────────────

class UpV2(nn.Module):
    """Decoder Up block without checkerboard artefacts.

    Replaces ``nn.ConvTranspose3d(stride, stride)`` with::

        trilinear Upsample → Conv3d 3×3×3 → InstanceNorm → LeakyReLU

    The deep (low-resolution) feature is projected to *high_channels*,
    fused with the skip connection via add/cat, then spatially upsampled.
    
    [NEW] Dynamic spatial alignment handles stride=1 or rounding mismatches
    without additional computational overhead in the common case.
    """

    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        out_channels: int,
        num_conv: int = 1,
        stride: Tuple[int, ...] = (2, 2, 2),
        fusion_mode: str = "add",
        **kwargs,   # absorbs unused kwargs (expand_rate, dropout_rate, …)
    ) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode
        self.stride = stride

        # Project deep feature: low_channels → high_channels
        proj_layers: List[nn.Module] = [
            nn.Sequential(
                nn.Conv3d(low_channels, high_channels, 1, bias=False),
                nn.InstanceNorm3d(high_channels, affine=True),
                nn.LeakyReLU(inplace=True),
            )
        ]
        for _ in range(1, num_conv):
            proj_layers.append(
                nn.Sequential(
                    nn.Conv3d(high_channels, high_channels, 3, 1, 1, bias=False),
                    nn.InstanceNorm3d(high_channels, affine=True),
                    nn.LeakyReLU(inplace=True),
                )
            )
        self.extractor = nn.ModuleList(proj_layers)

        # 根据 fusion_mode 确定上采样卷积的输入通道数
        up_in_channels = high_channels * 2 if fusion_mode == "cat" else high_channels

        # Checkerboard-free upsample
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=False),
            nn.Conv3d(up_in_channels, high_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(high_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def _align_spatial(self, x_low: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
        """对齐 x_low 和 x_high 的空间尺寸（仅在必要时执行插值）。
        
        这种方式几乎不影响性能：
        - 当尺寸匹配时（大多数情况），直接返回 x_low，零开销
        - 当尺寸不匹配时，trilinear 插值非常快（相比卷积计算量很小）
        """
        if x_low.shape[2:] != x_high.shape[2:]:
            x_low = F.interpolate(
                x_low,
                size=x_high.shape[2:],
                mode="trilinear",
                align_corners=False
            )
        return x_low

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor) -> torch.Tensor:
        # 特征提取/投影
        for ext in self.extractor:
            x_low = ext(x_low)
        
        # 动态对齐空间尺寸（处理 stride=1 或尺寸舍入导致的不匹配）
        x_low = self._align_spatial(x_low, x_high)
        
        # 特征融合
        if self.fusion_mode == "cat":
            x = torch.cat([x_low, x_high], dim=1)
        else:
            x = x_low + x_high
        
        return self.up(x)


# ─────────────────────────────────────────────────────────────────────────────
# [2, 4, 5] Main optimised network
# ─────────────────────────────────────────────────────────────────────────────

class HCMA_SvANet_v2(nn.Module):
    """Optimised HCMA backbone with improved scale-variant attention (V2).

    See module docstring for the full change list.

    Parameters
    ----------
    in_channels : int
    n_classes : int
    depth : int
        Number of encoder/decoder levels (excluding stem), default 4.
    channels : list of int
        Feature-map widths at each level; length must be depth+1.
    input_size : (D, H, W)
        [NEW] Spatial size of the input patch used to compute positional
        embeddings inside Down blocks. Replaces the hardcoded [64,192,128].
    use_decoder_mamba : bool
        [NEW] If True, inserts a MambaLayer after every decoder Up block to
        add long-range context symmetrically with the encoder.
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        depth: int = 4,
        conv=DenseConv,
        channels: Optional[List[int]] = None,
        encoder_num_conv: Optional[List[int]] = None,
        decoder_num_conv: Optional[List[int]] = None,
        encoder_expand_rate: Optional[List[int]] = None,
        decoder_expand_rate: Optional[List[int]] = None,
        strides: Optional[List[tuple]] = None,
        dropout_rate_list: Optional[List[float]] = None,
        drop_path_rate_list: Optional[List[float]] = None,
        deep_supervision: bool = False,
        predict_mode: bool = False,
        is_skip: bool = False,
        is_split: bool = True,
        is_slice_attention: bool = True,
        # [4] Parameterised input volume size (D, H, W)
        input_size: Tuple[int, int, int] = (64, 192, 128),
        # [5] Insert lightweight MambaLayer after each decoder Up block
        use_decoder_mamba: bool = False,
    ) -> None:
        super().__init__()

        # ── defaults ──────────────────────────────────────────────────────
        if channels is None:
            channels = [2**i for i in range(5, 10)]
        if encoder_num_conv is None:
            encoder_num_conv = [1] * depth
        if decoder_num_conv is None:
            decoder_num_conv = [1] * depth
        if encoder_expand_rate is None:
            encoder_expand_rate = [4] * depth
        if decoder_expand_rate is None:
            decoder_expand_rate = [4] * depth
        if strides is None:
            strides = [(2, 2, 2)] * 3 + [(1, 1, 1)]
        if dropout_rate_list is None:
            dropout_rate_list = [0.025, 0.05, 0.1, 0.1]
        if drop_path_rate_list is None:
            drop_path_rate_list = [0.025, 0.05, 0.1, 0.1]

        assert len(channels) == depth + 1, "len(channels) must equal depth + 1"
        assert len(strides) == depth, "len(strides) must equal depth"

        self.depth = depth
        self.deep_supervision = deep_supervision
        self.predict_mode = predict_mode
        self.is_skip = is_skip
        self.use_decoder_mamba = use_decoder_mamba

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skips = nn.ModuleList()

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoders.append(DenseConv(in_channels, channels[0]))

        # [4] use parameterised input_size instead of hardcoded [64, 192, 128]
        patch_ini = list(input_size)
        for i in range(depth):
            for j in range(3):
                patch_ini[j] = int(patch_ini[j] / strides[i][0])
            self.encoders.append(
                Down(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    conv=conv,
                    num_conv=encoder_num_conv[i],
                    stride=strides[i],
                    patch_size=list(patch_ini),
                    is_split=is_split,
                    expand_rate=encoder_expand_rate[i],
                    dropout_rate=dropout_rate_list[i],
                    drop_path_rate=drop_path_rate_list[i],
                    is_slice_attention=is_slice_attention,
                )
            )

        # ── Decoder (UpV2) ────────────────────────────────────────────────
        # [3] Use checkerboard-free UpV2 blocks
        for i in range(depth):
            stride_val = strides[depth - i - 1][0]
            patch_ini = [p * stride_val for p in patch_ini]
            self.decoders.append(
                UpV2(
                    low_channels=channels[depth - i],
                    high_channels=channels[depth - i - 1],
                    out_channels=channels[depth - i - 1],
                    num_conv=decoder_num_conv[depth - i - 1],
                    stride=strides[depth - i - 1],
                    fusion_mode="add",
                    expand_rate=decoder_expand_rate[depth - i - 1],
                    dropout_rate=dropout_rate_list[depth - i - 1],
                    drop_path_rate=drop_path_rate_list[depth - i - 1],
                )
            )

        # ── Skip connections ──────────────────────────────────────────────
        for i in range(depth):
            if is_skip:
                from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA import (
                    TripleLine3DFusion,
                )
                self.skips.append(
                    TripleLine3DFusion(
                        in_channels=channels[depth - i], kernel_size=7
                    )
                )
            else:
                self.skips.append(nn.Identity())

        # ── Output heads ──────────────────────────────────────────────────
        self.out = nn.ModuleList(
            [Out(channels[depth - i - 1], n_classes) for i in range(depth)]
        )

        # ── [2] Scale-variant attention (bottleneck double-attention fixed) ─
        #
        # V1 layout  (depth+1 enc + depth+1 dec):
        #   sv_attn_enc[depth] acts on bottleneck feature
        #   sv_attn_dec[0]     ALSO acts on the SAME bottleneck feature  ← BUG
        #
        # V2 layout:
        #   sv_attn_enc[0..depth-1]  → encoder levels (stem + non-bottleneck)
        #   sv_attn_bottleneck        → bottleneck, exactly ONCE
        #   sv_attn_dec[0..depth-1]  → fired after each decoder Up block

        # Stem + intermediate encoder levels  (depth modules for channels[0..depth-1])
        self.sv_attn_enc = nn.ModuleList(
            [ScaleVariantAttention3DV2(ch) for ch in channels[:-1]]
        )
        # Bottleneck: one dedicated module  — fixes the double-attention bug
        self.sv_attn_bottleneck = ScaleVariantAttention3DV2(channels[depth])
        # Decoder: one module per Up block  (depth modules for channels[depth-1..0])
        self.sv_attn_dec = nn.ModuleList(
            [ScaleVariantAttention3DV2(ch) for ch in channels[-2::-1]]
        )

        # ── [5] Optional decoder Mamba ────────────────────────────────────
        if use_decoder_mamba:
            self.dec_mamba = nn.ModuleList(
                [MambaLayer(dim=channels[depth - i - 1]) for i in range(depth)]
            )

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        encoder_features: List = []   # (x_down_pre_extractor | None, x_post_extractor)
        decoder_features: List = []

        # ── Encoder pass ──────────────────────────────────────────────────
        for i, encoder in enumerate(self.encoders):
            if i == 0:
                # Stem: DenseConv, no spatial downsampling
                x = encoder(x)
                x = self.sv_attn_enc[0](x)
                encoder_features.append((None, x))
            elif i < self.depth:
                # Intermediate Down blocks
                x_down, x = encoder(x)
                x = self.sv_attn_enc[i](x)
                encoder_features.append((x_down, x))
            else:
                # Bottleneck Down block: apply ONE dedicated attention only
                x_down, x = encoder(x)
                x = self.sv_attn_bottleneck(x)
                encoder_features.append((x_down, x))

        # ── Decoder pass ──────────────────────────────────────────────────
        #
        # encoder_features[depth] = (x_down_at_depth, bottleneck_output)
        #   x_down_at_depth  : pre-extractor feature, channels[depth-1]
        #   bottleneck_output: post-extractor feature, channels[depth]
        #                      already attended by sv_attn_bottleneck above
        #
        # The decoder loop runs i = 1..depth (depth iterations):
        #   decoders[i-1] = UpV2(low=channels[depth-i+1], high=channels[depth-i])
        #   Skip connection for step i uses the pre-extractor feature from
        #   encoder level (depth - i + 1), which carries channels[depth-i].

        x_down, x_dec = encoder_features[self.depth]   # unpack bottleneck
        x_dec = self.skips[0](x_dec)                   # optional TripleLine on bottleneck

        for i in range(1, self.depth + 1):
            if i == self.depth:
                # Last Up block: use x_down that was updated in the previous
                # iteration (= pre-extractor from encoder level 1, channels[0])
                x_dec = self.decoders[i - 1](x_dec, x_down)
            else:
                x_dec = self.decoders[i - 1](x_dec, self.skips[i](x_down))
                # Advance the skip-connection pointer to the next shallower level
                x_down = encoder_features[self.depth - i][0]

            # sv_attn_dec fires only after Up blocks, not at bottleneck
            x_dec = self.sv_attn_dec[i - 1](x_dec)

            # [5] Optional decoder Mamba for long-range context
            if self.use_decoder_mamba:
                x_dec = self.dec_mamba[i - 1](x_dec)

            decoder_features.append(x_dec)

        # ── Output ────────────────────────────────────────────────────────
        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, decoder_features)][::-1]
        if self.predict_mode:
            return self.out[-1](decoder_features[-1])
        return x_dec, self.out[-1](decoder_features[-1])


# ─────────────────────────────────────────────────────────────────────────────
# 测试代码
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 简单测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = HCMA_SvANet_v2(
        in_channels=1,
        n_classes=3,
        depth=4,
        channels=[32, 64, 128, 256, 512],
        input_size=(64, 192, 128),
        use_decoder_mamba=False,
    ).to(device)
    
    # 测试输入
    x = torch.randn(2, 1, 64, 192, 128).to(device)
    
    model.eval()
    with torch.no_grad():
        output, seg = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output feature shape: {output.shape}")
        print(f"Segmentation shape: {seg.shape}")
    
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")