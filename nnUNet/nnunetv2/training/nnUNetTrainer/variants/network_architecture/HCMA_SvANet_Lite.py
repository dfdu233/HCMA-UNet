from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA import (
    DenseConv,
    Down,
    Out,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA_SvANet_v2 import (
    ScaleVariantAttention3DV2,
    UpV2,
)


class HCMA_SvANet_Lite(nn.Module):
    """A lighter HCMA+SvA variant focused on speed/stability balance.

    Design choices:
    - Keep HCMA encoder/decoder macro-structure and skip interfaces unchanged.
    - Use checkerboard-free UpV2 in decoder.
    - Apply scale-variant attention only on selected stages instead of all stages.
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
        input_size: Tuple[int, int, int] = (64, 192, 128),
        encoder_attn_levels: Sequence[int] = (0, 2),
        decoder_attn_levels: Sequence[int] = (0,),
        pool_res: Iterable[int] = (1, 2),
        use_small_lesion_refine: bool = False,
    ) -> None:
        super().__init__()

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
        self.use_small_lesion_refine = use_small_lesion_refine
        self.enable_foreground_rescue = True
        self.min_foreground_ratio = 0.001
        self.target_foreground_ratio = 0.004
        self.max_foreground_boost = 3.0

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skips = nn.ModuleList()

        self.encoders.append(DenseConv(in_channels, channels[0]))

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

        for i in range(depth):
            if is_skip:
                from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA import (
                    TripleLine3DFusion,
                )

                self.skips.append(TripleLine3DFusion(in_channels=channels[depth - i], kernel_size=7))
            else:
                self.skips.append(nn.Identity())

        self.out = nn.ModuleList([Out(channels[depth - i - 1], n_classes) for i in range(depth)])

        encoder_attn_levels = sorted(set(int(i) for i in encoder_attn_levels if 0 <= int(i) < depth))
        decoder_attn_levels = sorted(set(int(i) for i in decoder_attn_levels if 0 <= int(i) < depth))

        self.sv_attn_enc = nn.ModuleDict(
            {
                str(i): ScaleVariantAttention3DV2(channels[i], pool_res=pool_res)
                for i in encoder_attn_levels
            }
        )
        self.sv_attn_bottleneck = ScaleVariantAttention3DV2(channels[depth], pool_res=pool_res)
        self.sv_attn_dec = nn.ModuleDict(
            {
                str(i): ScaleVariantAttention3DV2(channels[depth - i - 1], pool_res=pool_res)
                for i in decoder_attn_levels
            }
        )

        if self.use_small_lesion_refine:
            self.small_lesion_refine = nn.Sequential(
                nn.Conv3d(channels[0] * 2, channels[0], kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(channels[0], affine=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(channels[0], n_classes, kernel_size=1, bias=True),
            )
            self.small_lesion_alpha = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def _enc_attend(self, level: int, x: torch.Tensor) -> torch.Tensor:
        key = str(level)
        if key in self.sv_attn_enc:
            return self.sv_attn_enc[key](x)
        return x

    def _dec_attend(self, level: int, x: torch.Tensor) -> torch.Tensor:
        key = str(level)
        if key in self.sv_attn_dec:
            return self.sv_attn_dec[key](x)
        return x

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
        encoder_features = []
        decoder_features = []

        for i, encoder in enumerate(self.encoders):
            if i == 0:
                x = encoder(x)
                x = self._enc_attend(0, x)
                encoder_features.append((None, x))
            elif i < self.depth:
                x_down, x = encoder(x)
                x = self._enc_attend(i, x)
                encoder_features.append((x_down, x))
            else:
                x_down, x = encoder(x)
                x = self.sv_attn_bottleneck(x)
                encoder_features.append((x_down, x))

        x_down, x_dec = encoder_features[self.depth]
        x_dec = self.skips[0](x_dec)

        for i in range(1, self.depth + 1):
            if i == self.depth:
                x_dec = self.decoders[i - 1](x_dec, x_down)
            else:
                x_dec = self.decoders[i - 1](x_dec, self.skips[i](x_down))
                x_down = encoder_features[self.depth - i][0]

            x_dec = self._dec_attend(i - 1, x_dec)
            decoder_features.append(x_dec)

        base_logits = self.out[-1](decoder_features[-1])
        if self.use_small_lesion_refine:
            stem_feat = encoder_features[0][1]
            refine_logits = self.small_lesion_refine(torch.cat([decoder_features[-1], stem_feat], dim=1))
            base_logits = base_logits + self.small_lesion_alpha * refine_logits

        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, decoder_features)][::-1]
        if self.predict_mode:
            base_logits = self._foreground_rescue_logits(base_logits)
            return base_logits
        return x_dec, base_logits
