import argparse
import traceback

import torch
import torch.nn.functional as F

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.SwinHR import SwinHR
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.SwinUNETRv2 import SwinUNETR
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.MedNeXt import MedNeXt
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.Mamba3d import Mamba3d
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.segFormer3d import SegFormer3D
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.SingleBaselinev5 import SingleBaselinev5


def build_model(name: str, in_channels: int = 1, num_classes: int = 2):
    if name == "SwinHR":
        return SwinHR(
            img_size=(128, 128, 128),
            in_channels=in_channels,
            out_channels=num_classes,
            depths=(1, 1, 1, 1),
            num_heads=(1, 2, 4, 8),
            use_checkpoint=True,
        )

    if name == "SwinUNETRv2":
        return SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=in_channels,
            out_channels=num_classes,
            use_v2=False,
            predict_mode=True,
            use_checkpoint=True,
        )

    if name == "MedNeXt":
        return MedNeXt(
            in_channels=in_channels,
            n_channels=32,
            n_classes=num_classes,
            exp_r=[2, 3, 4, 4, 4, 4, 4, 3, 2],
            kernel_size=3,
            deep_supervision=False,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            predict_mode=True,
        )

    if name == "Mamba3d":
        return Mamba3d(in_channels=in_channels, n_classes=num_classes, predict_mode=True)

    if name == "segFormer3d":
        return SegFormer3D(in_channels=in_channels, num_classes=num_classes)

    if name == "SingleBaselinev5":
        return SingleBaselinev5(in_channels=in_channels, n_classes=num_classes, predict_mode=True)

    if name == "nn":
        return get_network_from_plans(
            arch_class_name="dynamic_network_architectures.architectures.unet.PlainConvUNet",
            arch_kwargs={
                "n_stages": 6,
                "features_per_stage": [32, 64, 128, 256, 320, 320],
                "conv_op": "torch.nn.modules.conv.Conv3d",
                "kernel_sizes": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
                "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
                "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
                "conv_bias": True,
                "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": "torch.nn.LeakyReLU",
                "nonlin_kwargs": {"inplace": True},
            },
            arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
            input_channels=in_channels,
            output_channels=num_classes,
            allow_init=True,
            deep_supervision=False,
        )

    raise ValueError(f"Unsupported model: {name}")


def to_logits(output):
    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise RuntimeError("Model returned empty output")
        output = output[-1]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(f"Unexpected output type: {type(output)}")
    return output


def run(name: str, epochs: int, iters: int, in_channels: int, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name, in_channels=in_channels, num_classes=num_classes).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    print(f"[{name}] device={device}, epochs={epochs}, iters={iters}, patch=(128,128,128)")

    for ep in range(epochs):
        for it in range(iters):
            x = torch.randn(1, in_channels, 128, 128, 128, device=device)
            y = torch.randint(0, num_classes, (1, 128, 128, 128), device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                out = to_logits(model(x))
                if out.shape[2:] != (128, 128, 128):
                    raise RuntimeError(f"Unexpected output shape {tuple(out.shape)}")
                loss = F.cross_entropy(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"[{name}] epoch={ep} iter={it} loss={float(loss.detach().cpu()):.6f}")

    print(f"[{name}] OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2)
    args = parser.parse_args()

    try:
        run(args.model, args.epochs, args.iters, args.in_channels, args.num_classes)
    except Exception as exc:
        print(f"[{args.model}] FAILED: {exc}")
        traceback.print_exc()
        raise
