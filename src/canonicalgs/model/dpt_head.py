from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_scratch(in_shape: list[int], out_shape: int) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, padding=1, bias=False)
    return scratch


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.residual_1 = ResidualConvUnit(features)
        self.residual_2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
        size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            x = x + self.residual_1(skip)
        x = self.residual_2(x)
        if size is None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        else:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(x)


class DptFeatureHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        features: int,
        out_channels: list[int],
        lowres_out_channels: int,
        fullres_out_channels: int,
        output_stride: int = 4,
        use_clstoken: bool = True,
    ) -> None:
        super().__init__()
        self.use_clstoken = use_clstoken
        self.output_stride = max(1, output_stride)

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channel, kernel_size=1)
                for out_channel in out_channels
            ]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )
        if use_clstoken:
            self.readout_projects = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU(),
                    )
                    for _ in out_channels
                ]
            )

        self.scratch = _make_scratch(out_channels, features)
        self.refinenet1 = FeatureFusionBlock(features)
        self.refinenet2 = FeatureFusionBlock(features)
        self.refinenet3 = FeatureFusionBlock(features)
        self.refinenet4 = FeatureFusionBlock(features)

        self.lowres_head = nn.Conv2d(features, lowres_out_channels, kernel_size=3, padding=1)
        self.fullres_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features // 2, fullres_out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        dino_features: list[tuple[torch.Tensor, torch.Tensor] | torch.Tensor],
        patch_h: int,
        patch_w: int,
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        for index, feature in enumerate(dino_features):
            if self.use_clstoken:
                tokens, cls_token = feature
                readout = cls_token.unsqueeze(1).expand_as(tokens)
                tokens = self.readout_projects[index](torch.cat((tokens, readout), dim=-1))
            else:
                tokens = feature  # type: ignore[assignment]

            tokens = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)
            tokens = self.projects[index](tokens)
            tokens = self.resize_layers[index](tokens)
            outputs.append(tokens)

        layer1, layer2, layer3, layer4 = outputs
        layer1 = self.scratch.layer1_rn(layer1)
        layer2 = self.scratch.layer2_rn(layer2)
        layer3 = self.scratch.layer3_rn(layer3)
        layer4 = self.scratch.layer4_rn(layer4)

        path4 = self.refinenet4(layer4, size=layer3.shape[-2:])
        path3 = self.refinenet3(path4, layer3, size=layer2.shape[-2:])
        path2 = self.refinenet2(path3, layer2, size=layer1.shape[-2:])
        path1 = self.refinenet1(path2, layer1, size=layer1.shape[-2:])

        lowres_features = self.lowres_head(path1)
        fullres_features = self.fullres_head(path1)
        target_height = max(1, output_size[0] // self.output_stride)
        target_width = max(1, output_size[1] // self.output_stride)
        fullres_features = F.interpolate(
            fullres_features,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=True,
        )
        return lowres_features, fullres_features
