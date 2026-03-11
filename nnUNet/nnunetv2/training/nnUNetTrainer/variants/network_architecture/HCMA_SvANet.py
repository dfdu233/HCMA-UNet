import random
from typing import Iterable, List, Optional, Type

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA import (
    DenseConv,
    Down,
    Up,
    Out,
)


class ScaleVariantAttention3D(nn.Module):
    """3D version of SvANet MoCAttention (scale-variant attention)."""

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

        self.pools = nn.ModuleDict(
            {str(k): nn.AdaptiveAvgPool3d((k, k, k)) for k in self.pool_res}
        )

        self.se = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1, bias=True),
            act(),
            nn.Conv3d(hidden_channels, in_channels, kernel_size=1, bias=True),
            scale_act(),
        )

    def _sample_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            k = random.choice(self.pool_res)
            x1 = x
            if self.moc_order:
                idx = torch.randperm(x.shape[1], device=x.device)
                x1 = x[:, idx, ...]
            attn = self.pools[str(k)](x1)
            if attn.shape[-1] > 1 or attn.shape[-2] > 1 or attn.shape[-3] > 1:
                b, c, d, h, w = attn.shape
                attn = attn.view(b, c, d * h * w)
                rand_idx = torch.randint(0, attn.shape[-1], (1,), device=x.device)
                attn = attn[:, :, rand_idx].view(b, c, 1, 1, 1)
        else:
            attn = self.pools["1"](x)
        return attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self._sample_pool(x)
        return x * self.se(attn)


class HCMA_SvANet(nn.Module):
    """HCMA backbone with SvANet scale-variant attention fused into encoder/decoder."""

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
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [2**i for i in range(5, 10)]
        if encoder_num_conv is None:
            encoder_num_conv = [1, 1, 1, 1]
        if decoder_num_conv is None:
            decoder_num_conv = [1, 1, 1, 1]
        if encoder_expand_rate is None:
            encoder_expand_rate = [4] * 4
        if decoder_expand_rate is None:
            decoder_expand_rate = [4] * 4
        if strides is None:
            strides = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)]
        if dropout_rate_list is None:
            dropout_rate_list = [0.025, 0.05, 0.1, 0.1]
        if drop_path_rate_list is None:
            drop_path_rate_list = [0.025, 0.05, 0.1, 0.1]

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.deep_supervision = deep_supervision
        self.predict_mode = predict_mode
        self.is_skip = is_skip

        assert len(channels) == depth + 1, "len(channels) != depth + 1"
        assert len(strides) == depth, "len(strides) != depth"

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.skips = nn.ModuleList()

        self.encoders.append(DenseConv(in_channels, channels[0]))

        patch_ini = [64, 192, 128]
        for i in range(self.depth):
            for j in range(3):
                patch_ini[j] = int(patch_ini[j] / strides[i][0])
            self.encoders.append(
                Down(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    conv=conv,
                    num_conv=encoder_num_conv[i],
                    stride=strides[i],
                    patch_size=patch_ini,
                    is_split=is_split,
                    expand_rate=encoder_expand_rate[i],
                    dropout_rate=dropout_rate_list[i],
                    drop_path_rate=drop_path_rate_list[i],
                    is_slice_attention=is_slice_attention,
                )
            )

        for i in range(self.depth):
            stride_val = strides[self.depth - i - 1][0]
            patch_ini = [p * stride_val for p in patch_ini]
            self.decoders.append(
                Up(
                    low_channels=channels[self.depth - i],
                    high_channels=channels[self.depth - i - 1],
                    out_channels=channels[self.depth - i - 1],
                    patch_size=patch_ini,
                    is_split=is_split,
                    num_conv=decoder_num_conv[self.depth - i - 1],
                    stride=strides[self.depth - i - 1],
                    fusion_mode="add",
                    expand_rate=decoder_expand_rate[self.depth - i - 1],
                    dropout_rate=dropout_rate_list[self.depth - i - 1],
                    drop_path_rate=drop_path_rate_list[self.depth - i - 1],
                )
            )

        for i in range(self.depth):
            if self.is_skip:
                from nnunetv2.training.nnUNetTrainer.variants.network_architecture.HCMA import (
                    TripleLine3DFusion,
                )

                self.skips.append(
                    TripleLine3DFusion(in_channels=channels[self.depth - i], kernel_size=7)
                )
            else:
                self.skips.append(nn.Identity())

        self.out = nn.ModuleList(
            [Out(channels[depth - i - 1], n_classes) for i in range(depth)]
        )

        self.sv_attn_enc = nn.ModuleList(
            [ScaleVariantAttention3D(ch) for ch in channels]
        )
        self.sv_attn_dec = nn.ModuleList(
            [ScaleVariantAttention3D(ch) for ch in channels[::-1]]
        )

    def forward(self, x: torch.Tensor):
        encoder_features = []
        decoder_features = []

        for i, encoder in enumerate(self.encoders):
            if i == 0:
                x = encoder(x)
                x = self.sv_attn_enc[i](x)
                encoder_features.append([x])
            else:
                x_down, x = encoder(x)
                x = self.sv_attn_enc[i](x)
                encoder_features.append([x_down, x])

        for i in range(self.depth + 1):
            if i == 0:
                x_down, x_dec = (
                    encoder_features[self.depth - i][0],
                    encoder_features[self.depth - i][1],
                )
                x_dec = self.skips[i](x_dec)
                x_dec = self.sv_attn_dec[i](x_dec)
            elif i == self.depth:
                x_dec = self.decoders[i - 1](x_dec, x_down)
                x_dec = self.sv_attn_dec[i](x_dec)
                decoder_features.append(x_dec)
            else:
                x_dec = self.decoders[i - 1](x_dec, self.skips[i](x_down))
                x_dec = self.sv_attn_dec[i](x_dec)
                x_down = encoder_features[self.depth - i][0]
                decoder_features.append(x_dec)

        if self.deep_supervision:
            return [m(mask) for m, mask in zip(self.out, decoder_features)][::-1]
        if self.predict_mode:
            return self.out[-1](decoder_features[-1])
        return x_dec, self.out[-1](decoder_features[-1])
