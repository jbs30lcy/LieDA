from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HeatmapElement:
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

    def forward(self, x: Tensor) -> HeatmapElement:
        features = self.encoder(x)
        return HeatmapElement(
            heatmap=self.heatmap_head(features),
            offset=self.offset_head(features),
        )


class ShortNet(nn.Module):
    """Small skip-free heatmap + offset tracker with three stacked conv stages."""

    def __init__(
        self,
        in_channels: int = 7,
        stage_channels: tuple[int, int, int] = (16, 32, 64),
        heatmap_channels: int = 1,
        offset_channels: int = 2,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 3:
            raise ValueError("stage_channels must contain three stages.")

        layers: list[nn.Module] = []
        prev_channels = in_channels
        for stage_channel in stage_channels:
            for conv_idx in range(3):
                stride = 2 if conv_idx == 2 else 1
                layers.extend(
                    [
                        nn.Conv2d(prev_channels, stage_channel, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(stage_channel),
                        nn.ReLU(inplace=True),
                    ]
                )
                prev_channels = stage_channel
        self.net = nn.Sequential(*layers)
        self.heatmap_head = nn.Conv2d(prev_channels, heatmap_channels, kernel_size=1)
        self.offset_head = nn.Conv2d(prev_channels, offset_channels, kernel_size=1)

    def forward(self, x: Tensor) -> HeatmapElement:
        features = self.net(x)
        return HeatmapElement(
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

    def forward(self, x: Tensor) -> HeatmapElement:
        skips: list[Tensor] = []
        features = x
        for block in self.encoder:
            features = block(features)
            skips.append(features)

        features = self.decoder[0](features, skips[3])
        features = self.decoder[1](features, skips[2])
        return HeatmapElement(
            heatmap=self.heatmap_head(features),
            offset=self.offset_head(features),
        )


class MergeNet(nn.Module):
    """Merge current all-object heatmap with previous target heatmap."""

    def __init__(
        self,
        heatmap_model: nn.Module,
        *,
        merge_channels: tuple[int, ...] = (16, 16, 16),
        heatmap_freeze: bool = False,
        merge_freeze: bool = False,
        teacher_forcing: float = 1.0,
        initial_target_center: tuple[float, float] = (300.0, 300.0),
        initial_target_input_size: int = 600,
        initial_target_sigma: float = 4.0,
    ) -> None:
        super().__init__()
        if not merge_channels:
            raise ValueError("merge_channels must contain at least one stage.")
        self.heatmap_model = heatmap_model
        self.initial_target_center = initial_target_center
        self.initial_target_input_size = int(initial_target_input_size)
        self.initial_target_sigma = float(initial_target_sigma)
        self.set_teacher_forcing(teacher_forcing)

        layers: list[nn.Module] = []
        prev_channels = 2
        for next_channels in merge_channels:
            layers.extend(
                [
                    nn.Conv2d(prev_channels, next_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            prev_channels = next_channels
        layers.append(nn.Conv2d(prev_channels, 1, kernel_size=1))
        self.merge_model = nn.Sequential(*layers)

        self.set_heatmap_freeze(heatmap_freeze)
        self.set_merge_freeze(merge_freeze)

    def set_teacher_forcing(self, teacher_forcing: float) -> None:
        teacher_forcing = float(teacher_forcing)
        if not 0.0 <= teacher_forcing <= 1.0:
            raise ValueError(f"teacher_forcing must be in [0, 1], got {teacher_forcing}.")
        self.teacher_forcing = teacher_forcing

    def set_heatmap_freeze(self, freeze: bool = True) -> None:
        for parameter in self.heatmap_model.parameters():
            parameter.requires_grad_(not freeze)

    def set_merge_freeze(self, freeze: bool = True) -> None:
        for parameter in self.merge_model.parameters():
            parameter.requires_grad_(not freeze)

    def heatmap_parameters(self):
        return self.heatmap_model.parameters()

    def merge_parameters(self):
        return self.merge_model.parameters()

    def load_heatmap_state_dict(self, state_dict: dict[str, Tensor], *, strict: bool = True):
        heatmap_prefix = "heatmap_model."
        if any(key.startswith(heatmap_prefix) for key in state_dict):
            state_dict = {
                key[len(heatmap_prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(heatmap_prefix)
            }
        return self.heatmap_model.load_state_dict(state_dict, strict=strict)

    def _heatmap_forward(self, x: Tensor) -> HeatmapElement:
        heatmap_trainable = any(parameter.requires_grad for parameter in self.heatmap_model.parameters())
        if heatmap_trainable:
            all_output = self.heatmap_model(x)
        else:
            with torch.no_grad():
                all_output = self.heatmap_model(x)
        if not isinstance(all_output, HeatmapElement):
            raise TypeError("MergeNet heatmap_model must return HeatmapElement.")
        return all_output

    def _initial_target_like(self, heatmap: Tensor) -> Tensor:
        batch_size, _channels, height, width = heatmap.shape
        center_x, center_y = self.initial_target_center
        stride_x = self.initial_target_input_size / float(width)
        stride_y = self.initial_target_input_size / float(height)
        grid_x = heatmap.new_tensor(center_x / stride_x)
        grid_y = heatmap.new_tensor(center_y / stride_y)

        ys = torch.arange(height, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, height, 1)
        xs = torch.arange(width, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, 1, width)
        distance2 = (xs - grid_x).square() + (ys - grid_y).square()
        initial = torch.exp(distance2 / (-2.0 * self.initial_target_sigma * self.initial_target_sigma))
        return initial.expand(batch_size, 1, height, width).contiguous()

    def choose_next_target(
        self,
        output_heatmap: Tensor,
        teacher_target: Tensor | None = None,
        teacher_forcing: float | None = None,
    ) -> Tensor:
        teacher_forcing = self.teacher_forcing if teacher_forcing is None else float(teacher_forcing)
        if not 0.0 <= teacher_forcing <= 1.0:
            raise ValueError(f"teacher_forcing must be in [0, 1], got {teacher_forcing}.")
        if self.training and teacher_target is not None and teacher_forcing > 0.0:
            semistep_sample = torch.rand((), device=output_heatmap.device).item()
            if teacher_forcing >= 1.0 or semistep_sample < teacher_forcing:
                return teacher_target
        return output_heatmap

    def forward_step(
        self,
        x: Tensor,
        prev_target_heatmap: Tensor | None = None,
        teacher_target: Tensor | None = None,
        teacher_forcing: float | None = None,
    ) -> tuple[HeatmapElement, Tensor, HeatmapElement]:
        all_output = self._heatmap_forward(x)
        if prev_target_heatmap is None:
            prev_target_heatmap = self._initial_target_like(all_output.heatmap)
        merge_input = torch.cat((prev_target_heatmap, all_output.heatmap), dim=1)
        merged_heatmap = self.merge_model(merge_input)
        next_target = self.choose_next_target(
            merged_heatmap,
            teacher_target=teacher_target,
            teacher_forcing=teacher_forcing,
        )
        return HeatmapElement(heatmap=merged_heatmap, offset=all_output.offset), next_target, all_output

    def forward(
        self,
        x: Tensor,
        prev_target_heatmap: Tensor | None = None,
        teacher_targets: Tensor | None = None,
        teacher_forcing: float | None = None,
        return_sequence: bool = True,
    ):
        if x.ndim == 4:
            output, next_target, _all_output = self.forward_step(
                x,
                prev_target_heatmap,
                teacher_target=teacher_targets,
                teacher_forcing=teacher_forcing,
            )
            return output if prev_target_heatmap is None else (output, next_target)
        if x.ndim != 5:
            raise ValueError("x must have shape (B, C, H, W) or (B, T, C, H, W).")
        if teacher_targets is not None and (
            teacher_targets.ndim != 5 or teacher_targets.shape[:2] != x.shape[:2]
        ):
            raise ValueError("teacher_targets must have shape (B, T, 1, H, W).")

        outputs: list[HeatmapElement] = []
        last_output: HeatmapElement | None = None
        next_target = prev_target_heatmap
        for frame_idx in range(x.shape[1]):
            teacher_target = None if teacher_targets is None else teacher_targets[:, frame_idx]
            output, next_target, _all_output = self.forward_step(
                x[:, frame_idx],
                next_target,
                teacher_target=teacher_target,
                teacher_forcing=teacher_forcing,
            )
            last_output = output
            if return_sequence:
                outputs.append(output)
        if not return_sequence:
            if last_output is None:
                raise RuntimeError("No output was produced.")
            return last_output, next_target
        return HeatmapElement(
            heatmap=torch.stack([output.heatmap for output in outputs], dim=1),
            offset=torch.stack([output.offset for output in outputs], dim=1),
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
    output: HeatmapElement | Tensor,
    *,
    input_size: int = 416,
    use_distance_bias: bool = False,
    gamma: float = 0.0,
    distance_center: tuple[float, float] | None = None,
    normalize_distance: bool = True,
) -> Tensor:
    heatmap = output.heatmap if isinstance(output, HeatmapElement) else output
    offset = output.offset if isinstance(output, HeatmapElement) else None
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
