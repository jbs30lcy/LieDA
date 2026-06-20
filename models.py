from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TrackerOutput:
    heatmap: Tensor
    offset: Tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DecodeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        return self.conv(x)


class TinyNet(nn.Module):
    """Small heatmap + offset tracker for temporal crops."""

    def __init__(
        self,
        in_channels: int = 7,
        channels: tuple[int, ...] = (32, 64, 128, 192, 256),
        heatmap_channels: int = 1,
        offset_channels: int = 2,
    ) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("channels must contain 5 stages.")

        blocks: list[nn.Module] = []
        prev_channels = in_channels
        for next_channels in channels:
            blocks.append(ConvBlock(prev_channels, next_channels))
            prev_channels = next_channels

        self.encoder = nn.Sequential(*blocks)
        self.heatmap_head = nn.Conv2d(prev_channels, heatmap_channels, kernel_size=1)
        self.offset_head = nn.Conv2d(prev_channels, offset_channels, kernel_size=1)

    def forward(self, x: Tensor) -> TrackerOutput:
        features = self.encoder(x)
        return TrackerOutput(
            heatmap=self.heatmap_head(features),
            offset=self.offset_head(features),
        )


class HeatUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        channels: tuple[int, ...] = (32, 64, 128, 192, 256),
        heatmap_channels: int = 1,
    ) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("channels must contain 5 stages.")

        encoder_blocks: list[nn.Module] = []
        prev_channels = in_channels
        for next_channels in channels:
            encoder_blocks.append(ConvBlock(prev_channels, next_channels))
            prev_channels = next_channels
        self.encoder = nn.ModuleList(encoder_blocks)

        c1, c2, c3, c4, c5 = channels
        self.decoder = nn.ModuleList(
            [
                DecodeBlock(c5, c4, skip_channels=c4),
                DecodeBlock(c4, c3, skip_channels=c3),
                DecodeBlock(c3, c2, skip_channels=c2),
                DecodeBlock(c2, c1, skip_channels=c1),
                DecodeBlock(c1, c1),
            ]
        )
        self.heatmap_head = nn.Conv2d(c1, heatmap_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skips: list[Tensor] = []
        features = x
        for block in self.encoder:
            features = block(features)
            skips.append(features)

        features = self.decoder[0](features, skips[3])
        features = self.decoder[1](features, skips[2])
        features = self.decoder[2](features, skips[1])
        features = self.decoder[3](features, skips[0])
        features = self.decoder[4](features)
        return self.heatmap_head(features)


class PartialHeatUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        channels: tuple[int, ...] = (32, 64, 128, 192, 256),
        heatmap_channels: int = 1,
        offset_channels: int = 2,
    ) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("channels must contain 5 stages.")

        encoder_blocks: list[nn.Module] = []
        prev_channels = in_channels
        for next_channels in channels:
            encoder_blocks.append(ConvBlock(prev_channels, next_channels))
            prev_channels = next_channels
        self.encoder = nn.ModuleList(encoder_blocks)

        _c1, _c2, c3, c4, c5 = channels
        self.decoder = nn.ModuleList(
            [
                DecodeBlock(c5, c4, skip_channels=c4),
                DecodeBlock(c4, c3, skip_channels=c3),
            ]
        )
        self.heatmap_head = nn.Conv2d(c3, heatmap_channels, kernel_size=1)
        self.offset_head = nn.Conv2d(c3, offset_channels, kernel_size=1)

    def forward(self, x: Tensor) -> TrackerOutput:
        skips: list[Tensor] = []
        features = x
        for block in self.encoder:
            features = block(features)
            skips.append(features)

        features = self.decoder[0](features, skips[3])
        features = self.decoder[1](features, skips[2])
        return TrackerOutput(
            heatmap=self.heatmap_head(features),
            offset=self.offset_head(features),
        )


def make_distance_squared_map(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    center: tuple[float, float] | None = None,
    normalize: bool = True,
) -> Tensor:
    if center is None:
        center_x = (width - 1) * 0.5
        center_y = (height - 1) * 0.5
    else:
        center_x, center_y = center

    ys = torch.arange(height, device=device, dtype=dtype).view(height, 1)
    xs = torch.arange(width, device=device, dtype=dtype).view(1, width)
    distance2 = (xs - center_x).square() + (ys - center_y).square()

    if normalize:
        max_distance2 = distance2.max().clamp_min(torch.finfo(dtype).eps)
        distance2 = distance2 / max_distance2

    return distance2.view(1, 1, height, width)


def apply_distance_bias(
    heatmap: Tensor,
    *,
    gamma: float,
    enabled: bool = False,
    center: tuple[float, float] | None = None,
    normalize: bool = True,
) -> Tensor:
    if not enabled:
        return heatmap

    _, _, height, width = heatmap.shape
    distance2 = make_distance_squared_map(
        height,
        width,
        device=heatmap.device,
        dtype=heatmap.dtype,
        center=center,
        normalize=normalize,
    )
    return heatmap + gamma * distance2


@torch.no_grad()
def decode_centers(
    output: TrackerOutput | Tensor,
    *,
    input_size: int = 416,
    use_distance_bias: bool = False,
    gamma: float = 0.0,
    distance_center: tuple[float, float] | None = None,
    normalize_distance: bool = True,
) -> Tensor:
    heatmap = output.heatmap if isinstance(output, TrackerOutput) else output
    offset = output.offset if isinstance(output, TrackerOutput) else None
    batch_size, _, height, width = heatmap.shape
    stride = input_size / float(width)

    scores = apply_distance_bias(
        heatmap,
        gamma=gamma,
        enabled=use_distance_bias,
        center=distance_center,
        normalize=normalize_distance,
    )
    flat_indices = scores.flatten(start_dim=2).argmax(dim=2).squeeze(1)
    ys = torch.div(flat_indices, width, rounding_mode="floor")
    xs = flat_indices.remainder(width)

    if offset is None:
        dx = torch.zeros(batch_size, device=heatmap.device, dtype=heatmap.dtype)
        dy = torch.zeros(batch_size, device=heatmap.device, dtype=heatmap.dtype)
        center_dtype = heatmap.dtype
    else:
        batch_indices = torch.arange(batch_size, device=heatmap.device)
        dx = offset[batch_indices, 0, ys, xs]
        dy = offset[batch_indices, 1, ys, xs]
        center_dtype = offset.dtype

    centers_x = (xs.to(center_dtype) + 0.5 + dx) * stride
    centers_y = (ys.to(center_dtype) + 0.5 + dy) * stride
    return torch.stack((centers_x, centers_y), dim=1)
