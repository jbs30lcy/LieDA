from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

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


class LieDA(nn.Module):
    """Small heatmap + offset tracker for 416x416 temporal crops."""

    def __init__(
        self,
        in_channels: int = 9,
        channels: tuple[int, ...] = (32, 64, 128, 192, 256),
        heatmap_channels: int = 1,
        offset_channels: int = 2,
    ) -> None:
        super().__init__()
        if len(channels) != 5:
            raise ValueError("channels must contain 5 stages so 416x416 becomes 13x13.")

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


def make_distance_squared_map(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    center: tuple[float, float] | None = None,
    normalize: bool = True,
) -> Tensor:
    """Returns a (1, 1, H, W) squared-distance map in grid coordinates."""
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
    """Optionally scores cells with heatmap + gamma * distance^2 before argmax."""
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
    output: TrackerOutput,
    *,
    input_size: int = 416,
    use_distance_bias: bool = False,
    gamma: float = 0.0,
    distance_center: tuple[float, float] | None = None,
    normalize_distance: bool = True,
) -> Tensor:
    """Decodes coarse argmax + local offset into crop-local pixel centers.

    Offset is interpreted in output-grid cell units. The returned tensor is
    shaped (B, 2) and ordered as (x, y) in the input crop coordinate system.
    """
    heatmap = output.heatmap
    offset = output.offset
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

    batch_indices = torch.arange(batch_size, device=heatmap.device)
    dx = offset[batch_indices, 0, ys, xs]
    dy = offset[batch_indices, 1, ys, xs]

    centers_x = (xs.to(offset.dtype) + 0.5 + dx) * stride
    centers_y = (ys.to(offset.dtype) + 0.5 + dy) * stride
    return torch.stack((centers_x, centers_y), dim=1)


def _batch_inputs(batch: Any) -> Tensor:
    if isinstance(batch, dict):
        for key in ("inputs", "x", "image", "images"):
            if key in batch:
                return batch[key]
        raise KeyError("Batch dict must contain one of: inputs, x, image, images.")

    if isinstance(batch, (tuple, list)):
        return batch[0]

    return batch


def train(
    model: Any,
    batches: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str | None = None,
    epochs: int = 1,
    use_distance_bias: bool = False,
    gamma: float = 0.0,
) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)
    model.train()

    for _epoch in range(epochs):
        for batch in batches:
            inputs = _batch_inputs(batch).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)

            biased_heatmap = apply_distance_bias(
                output.heatmap,
                gamma=gamma,
                enabled=use_distance_bias,
            )

            # TODO: define the loss once target heatmap/offset conventions are fixed.
            # Example placeholders available here:
            # - output.heatmap: raw heatmap logits, shape (B, 1, 13, 13)
            # - biased_heatmap: output.heatmap + gamma * distance^2 when enabled
            # - output.offset: local offset field, shape (B, 2, 13, 13)
            # - batch: original prepared batch with labels/targets
            loss: Tensor | None = None

            if loss is None:
                raise NotImplementedError("TODO: implement heatmap/offset loss.")

            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    model = LieDA()
    # train(model, )