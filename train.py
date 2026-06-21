from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import time
import uuid
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from models import TinyNet, PartialHeatUNet, LieDA, HeatmapElement, apply_distance_bias, decode_centers
from my_dataloader import my_dataloader


@dataclass(frozen=True)
class LossOutput:
    loss: Tensor
    heatmap_loss: Tensor
    offset_loss: Tensor
    center_loss: Tensor


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _batch_inputs(batch: Any) -> Tensor:
    if isinstance(batch, dict):
        for key in ("frames", "inputs", "x", "image", "images"):
            if key in batch:
                return batch[key]
        raise KeyError("Batch dict must contain one of: frames, inputs, x, image, images.")

    if isinstance(batch, (tuple, list)):
        return batch[0]

    return batch


def _batch_positions(batch: Any) -> Tensor:
    if isinstance(batch, dict):
        for key in ("positions", "centers", "center", "target_centers", "target_center"):
            if key in batch:
                return batch[key]
        raise KeyError(
            "Batch dict must contain positions/centers for semistep crop centers."
        )

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[1]

    raise KeyError("Batch must include positions for semistep crop centers.")


def _batch_dummy_positions(batch: Any) -> Tensor:
    if isinstance(batch, dict):
        for key in ("dummy_positions", "object_positions", "objects", "object_centers"):
            if key in batch:
                return batch[key]
        raise KeyError(
            "Batch dict must contain dummy_positions/object_positions for dummy object heatmap training."
        )

    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        return batch[2]

    raise KeyError("Batch must include dummy object positions.")


def _batch_dummy_mask(batch: Any) -> Tensor:
    if isinstance(batch, dict):
        for key in ("dummy_mask", "object_mask", "objects_mask", "object_valid"):
            if key in batch:
                return batch[key]
        raise KeyError(
            "Batch dict must contain dummy_mask/object_mask for dummy object heatmap training."
        )

    if isinstance(batch, (tuple, list)) and len(batch) >= 4:
        return batch[3]

    raise KeyError("Batch must include a dummy object mask.")


def _image_to_temporal_tensor(image: Any, *, frames: int = 3) -> Tensor:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
    return image_tensor.repeat(frames, 1, 1)


def crop_around_centers(images: Tensor, centers: Tensor, crop_size: int = 416) -> Tensor:
    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W).")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape (B, 2).")

    batch_size, _channels, height, width = images.shape
    half_size = crop_size // 2
    pad_left = pad_right = pad_top = pad_bottom = half_size
    padded = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    crops: list[Tensor] = []
    for batch_idx in range(batch_size):
        center_x = int(torch.round(centers[batch_idx, 0]).item()) + pad_left
        center_y = int(torch.round(centers[batch_idx, 1]).item()) + pad_top
        left = center_x - half_size
        top = center_y - half_size
        crops.append(padded[batch_idx, :, top : top + crop_size, left : left + crop_size])

    return torch.stack(crops)


def crop_center(images: Tensor, crop_size: int = 576) -> Tensor:
    if images.ndim != 4:
        raise ValueError("images must have shape (B, C, H, W).")

    _batch_size, _channels, height, width = images.shape
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"crop_size must fit within images, got crop_size={crop_size}, image={(height, width)}."
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return images[:, :, top : top + crop_size, left : left + crop_size]


def make_semistep_inputs(
    frames: Tensor,
    positions: Tensor,
    semistep: int,
    *,
    crop_size: int = 576,
) -> Tensor:
    if frames.ndim != 5:
        raise ValueError("frames must have shape (B, T, C, H, W).")
    if frames.shape[2] != 3:
        raise ValueError(f"frames must have RGB channels, got {frames.shape[2]}.")
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape (B, T, 2).")
    if not 0 <= semistep < frames.shape[1] - 4:
        raise ValueError("semistep must be in [0, T - 5].")

    target_frame_idx = semistep + 4
    rgb = frames[:, target_frame_idx]
    grayscale = (
        0.299 * frames[:, :, 0]
        + 0.587 * frames[:, :, 1]
        + 0.114 * frames[:, :, 2]
    )
    differences = [
        grayscale[:, target_frame_idx - k] - grayscale[:, target_frame_idx - k - 1]
        for k in range(4)
    ]
    stacked = torch.cat([rgb, *(diff.unsqueeze(1) for diff in differences)], dim=1)
    return crop_center(stacked, crop_size=crop_size)


def make_semistep_targets(
    positions: Tensor,
    semistep: int,
    *,
    crop_size: int = 576,
    image_size: int = 600,
) -> Tensor:
    target_frame_idx = semistep + 4
    target_centers = positions[:, target_frame_idx]
    crop_offset = (image_size - crop_size) * 0.5
    return target_centers - crop_offset


def make_semistep_dummy_targets(
    dummy_positions: Tensor,
    dummy_mask: Tensor,
    semistep: int,
    *,
    crop_size: int = 576,
    image_size: int = 600,
) -> tuple[Tensor, Tensor]:
    target_frame_idx = semistep + 4
    crop_offset = (image_size - crop_size) * 0.5
    return (
        dummy_positions[:, target_frame_idx] - crop_offset,
        dummy_mask[:, target_frame_idx].to(torch.bool),
    )


def make_multi_center_heatmap_targets(
    centers: Tensor,
    mask: Tensor,
    *,
    input_size: int,
    output_height: int,
    output_width: int,
    sigma: float = 1.0,
) -> Tensor:
    if centers.ndim != 3 or centers.shape[-1] != 2:
        raise ValueError("centers must have shape (B, K, 2).")
    if mask.shape != centers.shape[:2]:
        raise ValueError("mask must have shape (B, K).")

    batch_size, object_count, _ = centers.shape
    device = centers.device
    dtype = centers.dtype
    stride_x = input_size / float(output_width)
    stride_y = input_size / float(output_height)

    grid_x = centers[..., 0] / stride_x
    grid_y = centers[..., 1] / stride_y
    valid = (
        mask.to(torch.bool)
        & (centers[..., 0] >= 0)
        & (centers[..., 0] < input_size)
        & (centers[..., 1] >= 0)
        & (centers[..., 1] < input_size)
    )

    cell_x = torch.floor(grid_x).long().clamp(0, output_width - 1)
    cell_y = torch.floor(grid_y).long().clamp(0, output_height - 1)
    ys = torch.arange(output_height, device=device, dtype=dtype).view(1, 1, output_height, 1)
    xs = torch.arange(output_width, device=device, dtype=dtype).view(1, 1, 1, output_width)
    distance2 = (xs - cell_x.to(dtype).view(batch_size, object_count, 1, 1)).square()
    distance2 = distance2 + (ys - cell_y.to(dtype).view(batch_size, object_count, 1, 1)).square()
    heatmaps = torch.exp(-distance2 / (2.0 * sigma * sigma))
    heatmaps = heatmaps * valid.to(dtype).view(batch_size, object_count, 1, 1)
    return heatmaps.max(dim=1).values.unsqueeze(1)


def make_heatmap_and_offset_targets(
    centers: Tensor,
    *,
    input_size: int,
    output_height: int,
    output_width: int,
    sigma: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor]:
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape (B, 2).")

    batch_size = centers.shape[0]
    device = centers.device
    dtype = centers.dtype
    stride_x = input_size / float(output_width)
    stride_y = input_size / float(output_height)

    grid_x = centers[:, 0] / stride_x
    grid_y = centers[:, 1] / stride_y
    cell_x = torch.floor(grid_x).long().clamp(0, output_width - 1)
    cell_y = torch.floor(grid_y).long().clamp(0, output_height - 1)

    ys = torch.arange(output_height, device=device, dtype=dtype).view(1, output_height, 1)
    xs = torch.arange(output_width, device=device, dtype=dtype).view(1, 1, output_width)
    distance2 = (xs - cell_x.to(dtype).view(batch_size, 1, 1)).square()
    distance2 = distance2 + (ys - cell_y.to(dtype).view(batch_size, 1, 1)).square()
    heatmap = torch.exp(-distance2 / (2.0 * sigma * sigma)).unsqueeze(1)

    offset = torch.zeros(batch_size, 2, output_height, output_width, device=device, dtype=dtype)
    offset_mask = torch.zeros(batch_size, 1, output_height, output_width, device=device, dtype=dtype)
    batch_indices = torch.arange(batch_size, device=device)
    offset[batch_indices, 0, cell_y, cell_x] = grid_x - cell_x.to(dtype) - 0.5
    offset[batch_indices, 1, cell_y, cell_x] = grid_y - cell_y.to(dtype) - 0.5
    offset_mask[batch_indices, 0, cell_y, cell_x] = 1.0
    return heatmap, offset, offset_mask


def softargmax_centers(
    output: HeatmapElement | Tensor,
    *,
    input_size: int,
    temperature: float = 0.25,
) -> Tensor:
    heatmap = output.heatmap if isinstance(output, HeatmapElement) else output
    offset = output.offset if isinstance(output, HeatmapElement) else None
    batch_size, _, height, width = heatmap.shape
    stride_x = input_size / float(width)
    stride_y = input_size / float(height)
    temperature = max(float(temperature), torch.finfo(heatmap.dtype).eps)

    probs = torch.softmax(heatmap.flatten(start_dim=2) / temperature, dim=-1)
    probs = probs.view(batch_size, 1, height, width)

    ys = torch.arange(height, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, height, 1)
    xs = torch.arange(width, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, 1, width)
    if offset is None:
        offset_x = torch.zeros_like(probs)
        offset_y = torch.zeros_like(probs)
    else:
        offset_x = offset[:, 0:1]
        offset_y = offset[:, 1:2]

    centers_x = ((xs + 0.5 + offset_x) * probs).sum(dim=(2, 3)) * stride_x
    centers_y = ((ys + 0.5 + offset_y) * probs).sum(dim=(2, 3)) * stride_y
    return torch.cat((centers_x, centers_y), dim=1)


def compute_tracking_loss(
    output: HeatmapElement | Tensor,
    target_centers: Tensor,
    *,
    crop_size: int,
    heatmap_loss_fn: nn.Module,
    offset_loss_fn: nn.Module,
    heatmap_sigma: float = 4.0,
    offset_loss_weight: float = 1.0,
    center_loss_weight: float = 0.01,
    softargmax_temperature: float = 0.25,
    use_distance_bias: bool = False,
    gamma: float = 0.0,
    heatmap_mode: str = "target",
    dummy_centers: Tensor | None = None,
    dummy_mask: Tensor | None = None,
) -> LossOutput:
    if heatmap_mode not in {"target", "all"}:
        raise ValueError(f"heatmap_mode must be 'target' or 'all', got {heatmap_mode!r}.")
    heatmap = output.heatmap if isinstance(output, HeatmapElement) else output
    biased_heatmap = apply_distance_bias(
        heatmap,
        gamma=gamma,
        enabled=use_distance_bias,
    )
    target_heatmap, target_offset, offset_mask = make_heatmap_and_offset_targets(
        target_centers,
        input_size=crop_size,
        output_height=heatmap.shape[-2],
        output_width=heatmap.shape[-1],
        sigma=heatmap_sigma,
    )
    if heatmap_mode == "all":
        if dummy_centers is None or dummy_mask is None:
            raise ValueError(
                "dummy_centers and dummy_mask are required when heatmap_mode='all'."
            )
        all_centers = torch.cat((target_centers.unsqueeze(1), dummy_centers), dim=1)
        target_mask = torch.ones(
            target_centers.shape[0],
            1,
            device=target_centers.device,
            dtype=torch.bool,
        )
        all_mask = torch.cat((target_mask, dummy_mask.to(torch.bool)), dim=1)
        target_heatmap = make_multi_center_heatmap_targets(
            all_centers,
            all_mask,
            input_size=crop_size,
            output_height=heatmap.shape[-2],
            output_width=heatmap.shape[-1],
            sigma=heatmap_sigma,
        )
    heatmap_loss = heatmap_loss_fn(biased_heatmap, target_heatmap)
    center_loss = heatmap_loss.new_zeros(())
    if heatmap_mode == "target" and center_loss_weight > 0:
        center_prediction = softargmax_centers(
            HeatmapElement(heatmap=biased_heatmap, offset=output.offset)
            if isinstance(output, HeatmapElement)
            else biased_heatmap,
            input_size=crop_size,
            temperature=softargmax_temperature,
        )
        center_loss = F.smooth_l1_loss(center_prediction, target_centers)

    if isinstance(output, HeatmapElement):
        offset_difference = output.offset - target_offset
        offset_loss = offset_loss_fn(
            offset_difference * offset_mask,
            torch.zeros_like(offset_difference),
        ) / offset_mask.sum().clamp_min(1.0)
        loss = heatmap_loss + offset_loss_weight * offset_loss + center_loss_weight * center_loss
    else:
        offset_loss = heatmap_loss.new_zeros(())
        loss = heatmap_loss + center_loss_weight * center_loss

    return LossOutput(
        loss=loss,
        heatmap_loss=heatmap_loss,
        offset_loss=offset_loss,
        center_loss=center_loss,
    )


def _experiment_suffix(experiment_name: str | None = None) -> str:
    if experiment_name is None or not experiment_name.strip():
        return uuid.uuid4().hex[:8]

    suffix = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", experiment_name.strip())
    suffix = suffix.strip(" ._")
    return suffix or uuid.uuid4().hex[:8]


def make_run_dir(root: str | Path = "runs", *, experiment_name: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{timestamp}_{_experiment_suffix(experiment_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "eval").mkdir()
    (run_dir / "checkpoints").mkdir()
    return run_dir


def save_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer,
    path: str | Path,
    *,
    step: int,
    config: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        checkpoint_path,
    )
    return checkpoint_path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _checkpoint_sort_key(path: Path) -> tuple[float, str]:
    return (path.stat().st_mtime, str(path))


def _run_dir_for_checkpoint(checkpoint: Path, run_root: Path) -> Path:
    relative_path = checkpoint.resolve().relative_to(run_root)
    if len(relative_path.parts) < 2:
        raise ValueError(f"Checkpoint must be inside a run directory: {checkpoint}.")
    return run_root / relative_path.parts[0]


def _find_latest_checkpoint(run_root: str | Path = "runs") -> Path:
    runs_root = Path(run_root).resolve()
    checkpoints = [path for path in runs_root.rglob("*.pt") if path.is_file()]
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoint found under {runs_root}.")

    newest_run = max(
        {_run_dir_for_checkpoint(checkpoint, runs_root) for checkpoint in checkpoints},
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    run_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if _is_relative_to(checkpoint.resolve(), newest_run.resolve())
    ]
    if not run_checkpoints:
        raise FileNotFoundError(f"No .pt checkpoint found under latest run {newest_run}.")
    return max(run_checkpoints, key=_checkpoint_sort_key)


def _resolve_params_path(params: str | Path, run_root: str | Path = "runs") -> Path:
    if str(params) == "latest":
        return _find_latest_checkpoint(run_root)

    checkpoint_path = Path(params)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    runs_root = Path(run_root).resolve()
    if checkpoint_path.suffix != ".pt":
        raise ValueError(f"params must point to a .pt file, got {checkpoint_path}.")
    if not _is_relative_to(checkpoint_path, runs_root):
        raise ValueError(f"params must be a .pt path under {runs_root}, got {checkpoint_path}.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"params checkpoint does not exist: {checkpoint_path}.")
    return checkpoint_path


def load_checkpoint(
    model: Any,
    optimizer: torch.optim.Optimizer | None,
    params: str | Path,
    *,
    device: torch.device,
    run_root: str | Path = "runs",
) -> dict[str, Any]:
    checkpoint_path = _resolve_params_path(params, run_root)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(device)
            except ValueError as e:
                print("[train][WARN] optimizer_state_dict mismatch.")
                print("[train][WARN] Model weights were loaded, but optimizer will start fresh.")
                print(f"[train][WARN] {e}")

        loaded_step = int(checkpoint.get("step", 0))
    else:
        model.load_state_dict(checkpoint)
        loaded_step = 0
        checkpoint = {"step": loaded_step}

    print(f"[train] params loaded: {checkpoint_path} (step={loaded_step})")
    return checkpoint


def load_heatmap_checkpoint(
    model: LieDA,
    params: str | Path,
    *,
    device: torch.device,
    run_root: str | Path = "runs",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint_path = _resolve_params_path(params, run_root)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        loaded_step = int(checkpoint.get("step", 0))
    else:
        state_dict = checkpoint
        loaded_step = 0
        checkpoint = {"step": loaded_step}
    model.load_heatmap_state_dict(state_dict, strict=strict)
    print(f"[train_gru] heatmap params loaded: {checkpoint_path} (step={loaded_step})")
    return checkpoint


def _detach_hidden(hidden: list[Tensor] | None) -> list[Tensor] | None:
    if hidden is None:
        return None
    return [state.detach() for state in hidden]


def _frames_to_uint8_hwc(frames: Tensor) -> np.ndarray:
    frames_cpu = frames.detach().cpu().clamp(0.0, 1.0)
    frames_hwc = frames_cpu.permute(0, 2, 3, 1).numpy()
    return (frames_hwc * 255.0).round().astype(np.uint8)


def _write_frame_image(frame: Tensor, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_uint8 = _frames_to_uint8_hwc(frame.unsqueeze(0))[0]
    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Failed to write frame image: {path}")
    return path


def _write_heatmap_image(heatmap: Tensor, path: str | Path, *, output_size: int = 576) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = torch.sigmoid(heatmap.detach().cpu()).to(torch.float32)
    min_value = values.min()
    max_value = values.max()
    normalized = (values - min_value) / (max_value - min_value).clamp_min(1e-8)
    heatmap_uint8 = (normalized.numpy() * 255.0).round().astype(np.uint8)
    heatmap_uint8 = cv2.resize(
        heatmap_uint8,
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST,
    )
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    if not cv2.imwrite(str(path), heatmap_bgr):
        raise RuntimeError(f"Failed to write heatmap image: {path}")
    return path


def _draw_centers_video(
    frames: Tensor,
    label_centers: Tensor,
    predicted_centers: Tensor,
    path: str | Path,
    *,
    fps: int = 30,
    start_frame: int = 2,
) -> None:
    frames_uint8 = _frames_to_uint8_hwc(frames)
    height, width = frames_uint8.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")

    labels = label_centers.detach().cpu().numpy()
    predictions = predicted_centers.detach().cpu().numpy()
    for frame_idx, frame_rgb in enumerate(frames_uint8):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if frame_idx >= start_frame:
            center_idx = frame_idx - start_frame
            if center_idx >= len(labels):
                continue
            lx, ly = labels[center_idx]
            px, py = predictions[center_idx]
            cv2.circle(frame_bgr, (round(lx), round(ly)), 7, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            cv2.circle(frame_bgr, (round(px), round(py)), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.line(
                frame_bgr,
                (round(lx), round(ly)),
                (round(px), round(py)),
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        writer.write(frame_bgr)

    writer.release()


@torch.no_grad()
def eval(
    model: Any,
    frames: Tensor,
    positions: Tensor,
    *,
    dummy_positions: Tensor | None = None,
    dummy_mask: Tensor | None = None,
    output_dir: str | Path,
    name: str = "eval",
    device: torch.device | str | None = None,
    crop_size: int = 576,
    heatmap_sigma: float = 4.0,
    offset_loss_weight: float = 1.0,
    center_loss_weight: float = 0.01,
    softargmax_temperature: float = 0.25,
    use_distance_bias: bool = False,
    gamma: float = 0.0,
    hit_radius: float = 55.2,
    fps: int = 30,
    train_elapsed_seconds: float | None = None,
    step_train_seconds: float | None = None,
    heatmap_mode: str = "target",
    experiment_name: str | None = None,
) -> dict[str, Any]:
    if heatmap_mode not in {"target", "all"}:
        raise ValueError(f"heatmap_mode must be 'target' or 'all', got {heatmap_mode!r}.")
    eval_start_time = time.perf_counter()
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    if frames.ndim == 5:
        if frames.shape[0] != 1:
            raise ValueError("eval expects a single sequence: use shape (300, 3, 600, 600) or (1, 300, 3, 600, 600).")
        frames = frames[0]
    if positions.ndim == 3:
        if positions.shape[0] != 1:
            raise ValueError("eval expects a single position sequence: use shape (300, 2) or (1, 300, 2).")
        positions = positions[0]
    if frames.shape != (300, 3, 600, 600):
        raise ValueError(f"Expected eval frames shape (300, 3, 600, 600), got {tuple(frames.shape)}.")
    if positions.shape != (300, 2):
        raise ValueError(f"Expected eval positions shape (300, 2), got {tuple(positions.shape)}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{name}.mp4"
    metrics_path = output_dir / f"{name}.json"
    last_frame_path = output_dir / f"{name}_last_frame.png"
    last_heatmap_path = output_dir / f"{name}_last_heatmap.png"

    was_training = model.training
    model.eval()
    frames_batch = frames.unsqueeze(0).to(device)
    positions_batch = positions.unsqueeze(0).to(device, dtype=frames_batch.dtype)
    if heatmap_mode == "all" and (dummy_positions is None or dummy_mask is None):
        raise ValueError(
            "dummy_positions and dummy_mask are required when heatmap_mode='all'."
        )
    if dummy_positions is not None and dummy_mask is not None:
        dummy_positions_batch = dummy_positions.unsqueeze(0).to(device, dtype=frames_batch.dtype)
        dummy_mask_batch = dummy_mask.unsqueeze(0).to(device)
    else:
        dummy_positions_batch = None
        dummy_mask_batch = None
    heatmap_loss_fn = nn.BCEWithLogitsLoss()
    offset_loss_fn = nn.SmoothL1Loss(reduction="sum")
    predicted_centers: list[Tensor] = []
    label_centers: list[Tensor] = []
    losses: list[float] = []
    heatmap_losses: list[float] = []
    offset_losses: list[float] = []
    center_losses: list[float] = []
    heatmap_predicted_centers: list[Tensor] = []
    heatmap_losses_from_part: list[float] = []
    heatmap_heatmap_losses_from_part: list[float] = []
    heatmap_offset_losses_from_part: list[float] = []
    heatmap_center_losses_from_part: list[float] = []
    last_predicted_heatmap: Tensor | None = None
    hidden: list[Tensor] | None = None

    for semistep in range(frames_batch.shape[1] - 4):
        inputs = make_semistep_inputs(
            frames_batch,
            positions_batch,
            semistep,
            crop_size=crop_size,
        )
        target_centers = make_semistep_targets(
            positions_batch,
            semistep,
            crop_size=crop_size,
        )
        if dummy_positions_batch is not None and dummy_mask_batch is not None:
            assert dummy_positions_batch is not None
            assert dummy_mask_batch is not None
            dummy_centers, semistep_dummy_mask = make_semistep_dummy_targets(
                dummy_positions_batch,
                dummy_mask_batch,
                semistep,
                crop_size=crop_size,
            )
        else:
            dummy_centers = None
            semistep_dummy_mask = None
        if isinstance(model, LieDA):
            output, hidden, all_output = model.forward_step(inputs, hidden)
            heatmap_part_mode = "all" if dummy_centers is not None and semistep_dummy_mask is not None else "target"
            heatmap_part_loss = compute_tracking_loss(
                all_output,
                target_centers,
                crop_size=crop_size,
                heatmap_loss_fn=heatmap_loss_fn,
                offset_loss_fn=offset_loss_fn,
                heatmap_sigma=heatmap_sigma,
                offset_loss_weight=offset_loss_weight,
                center_loss_weight=center_loss_weight,
                softargmax_temperature=softargmax_temperature,
                use_distance_bias=use_distance_bias,
                gamma=gamma,
                heatmap_mode=heatmap_part_mode,
                dummy_centers=dummy_centers,
                dummy_mask=semistep_dummy_mask,
            )
            heatmap_part_local = decode_centers(
                all_output,
                input_size=crop_size,
                use_distance_bias=use_distance_bias,
                gamma=gamma,
            )
            heatmap_part_global = heatmap_part_local + (frames_batch.shape[-1] - crop_size) * 0.5
            heatmap_predicted_centers.append(heatmap_part_global.squeeze(0).detach().cpu())
            heatmap_losses_from_part.append(float(heatmap_part_loss.loss.detach().cpu()))
            heatmap_heatmap_losses_from_part.append(float(heatmap_part_loss.heatmap_loss.detach().cpu()))
            heatmap_offset_losses_from_part.append(float(heatmap_part_loss.offset_loss.detach().cpu()))
            heatmap_center_losses_from_part.append(float(heatmap_part_loss.center_loss.detach().cpu()))
        else:
            output = model(inputs)
        output_heatmap = output.heatmap if isinstance(output, HeatmapElement) else output
        last_predicted_heatmap = output_heatmap[0, 0].detach().cpu()
        loss_output = compute_tracking_loss(
            output,
            target_centers,
            crop_size=crop_size,
            heatmap_loss_fn=heatmap_loss_fn,
            offset_loss_fn=offset_loss_fn,
            heatmap_sigma=heatmap_sigma,
            offset_loss_weight=offset_loss_weight,
            center_loss_weight=center_loss_weight,
            softargmax_temperature=softargmax_temperature,
            use_distance_bias=use_distance_bias,
            gamma=gamma,
            heatmap_mode=heatmap_mode,
            dummy_centers=dummy_centers,
            dummy_mask=semistep_dummy_mask,
        )
        predicted_local = decode_centers(
            output,
            input_size=crop_size,
            use_distance_bias=use_distance_bias,
            gamma=gamma,
        )
        crop_offset = (frames_batch.shape[-1] - crop_size) * 0.5
        predicted_global = predicted_local + crop_offset
        label_global = positions_batch[:, semistep + 4]

        predicted_centers.append(predicted_global.squeeze(0).detach().cpu())
        label_centers.append(label_global.squeeze(0).detach().cpu())
        losses.append(float(loss_output.loss.detach().cpu()))
        heatmap_losses.append(float(loss_output.heatmap_loss.detach().cpu()))
        offset_losses.append(float(loss_output.offset_loss.detach().cpu()))
        center_losses.append(float(loss_output.center_loss.detach().cpu()))

    predicted_tensor = torch.stack(predicted_centers)
    label_tensor = torch.stack(label_centers)
    distances = torch.linalg.norm(predicted_tensor - label_tensor, dim=1)
    hits = distances <= hit_radius # TODO : hit 판정을 radius로 판정하는 거 고쳐야 됨
    metrics = {
        "name": name,
        "experiment_name": experiment_name,
        "run_dir": str(output_dir.parent),
        "num_predictions": int(predicted_tensor.shape[0]),
        "accuracy": float(hits.float().mean().item()),
        "hit_radius": float(hit_radius),
        "heatmap_mode": heatmap_mode,
        "loss": float(np.mean(losses)),
        "heatmap_loss": float(np.mean(heatmap_losses)),
        "offset_loss": float(np.mean(offset_losses)),
        "center_loss": float(np.mean(center_losses)),
        "standard_distance": float(distances.mean().item()),
        "median_distance": float(distances.median().item()),
        "video_path": str(video_path),
        "last_frame_path": str(last_frame_path),
        "last_heatmap_path": str(last_heatmap_path),
        "metrics_path": str(metrics_path),
        "play_command": f"python -c \"import os; os.startfile(r'{video_path}')\"",
    }
    if heatmap_predicted_centers:
        heatmap_predicted_tensor = torch.stack(heatmap_predicted_centers)
        heatmap_distances = torch.linalg.norm(heatmap_predicted_tensor - label_tensor, dim=1)
        heatmap_hits = heatmap_distances <= hit_radius
        metrics.update(
            {
                "heatmap_eval_mode": "all" if dummy_positions_batch is not None else "target",
                "heatmap_eval_loss": float(np.mean(heatmap_losses_from_part)),
                "heatmap_heatmap_loss": float(np.mean(heatmap_heatmap_losses_from_part)),
                "heatmap_offset_loss": float(np.mean(heatmap_offset_losses_from_part)),
                "heatmap_center_loss": float(np.mean(heatmap_center_losses_from_part)),
                "heatmap_accuracy": float(heatmap_hits.float().mean().item()),
                "heatmap_standard_distance": float(heatmap_distances.mean().item()),
                "heatmap_median_distance": float(heatmap_distances.median().item()),
            }
        )
    metrics["predicted_centers"] = predicted_tensor.tolist()
    metrics["label_centers"] = label_tensor.tolist()
    if heatmap_predicted_centers:
        metrics["heatmap_predicted_centers"] = heatmap_predicted_tensor.tolist()

    _draw_centers_video(
        frames,
        label_tensor,
        predicted_tensor,
        video_path,
        fps=fps,
        start_frame=4,
    )
    if last_predicted_heatmap is None:
        raise RuntimeError("No predicted heatmap was produced during eval.")
    _write_frame_image(frames[-1], last_frame_path)
    _write_heatmap_image(last_predicted_heatmap, last_heatmap_path, output_size=crop_size)
    _sync_device(device)
    metrics["eval_time_seconds"] = float(time.perf_counter() - eval_start_time)
    if train_elapsed_seconds is not None:
        metrics["train_elapsed_seconds"] = float(train_elapsed_seconds)
    if step_train_seconds is not None:
        metrics["step_train_seconds"] = float(step_train_seconds)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if was_training:
        model.train()
    print(f"[eval] done: {name} metrics={metrics_path} video={video_path}")
    return metrics


def train_heatmap(
    model: Any,
    dataloader: DataLoader | None = None,
    *,
    device: torch.device | str | None = None,
    params: str | Path | None = None,
    steps: int = 100,
    batch: int = 16,
    lr: float = 1e-3,
    crop_size: int = 576,
    eval_step: int = 20,
    save_step: int = 20,
    run_root: str | Path = "runs",
    experiment_name: str | None = None,
    heatmap_sigma: float = 4.0,
    offset_loss_weight: float = 1.0,
    center_loss_weight: float = 0.01,
    softargmax_temperature: float = 0.25,
    wandb_active: bool = False,
    wandb_project: str = "LieDA",
    use_distance_bias: bool = False,
    gamma: float = 0.0,
    heatmap_mode: str = "target",
) -> Path:
    if heatmap_mode not in {"target", "all"}:
        raise ValueError(f"heatmap_mode must be 'target' or 'all', got {heatmap_mode!r}.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loaded_checkpoint: dict[str, Any] | None = None
    if params is not None:
        loaded_checkpoint = load_checkpoint(
            model,
            optimizer,
            params,
            device=device,
            run_root=run_root,
        )
    heatmap_loss_fn = nn.BCEWithLogitsLoss()
    offset_loss_fn = nn.SmoothL1Loss(reduction="sum")
    if dataloader is None:
        dataloader = my_dataloader(
            difficulty=1,
            batch=batch,
            pin_memory=False,
            #pin_memory=device.type == "cuda",
        )

    run_dir = make_run_dir(run_root, experiment_name=experiment_name)
    print(f"[train] run directory: {run_dir}")
    wandb_run = None
    if wandb_active:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb_active=True requires wandb. Install it with: pip install wandb"
            ) from exc
        wandb_run = wandb.init(
            project=wandb_project,
            config={
                "steps": steps,
                "batch": batch,
                "lr": lr,
                "crop_size": crop_size,
                "eval_step": eval_step,
                "save_step": save_step,
                "run_dir": str(run_dir),
                "experiment_name": experiment_name,
                "params": None if params is None else str(params),
                "loaded_step": None if loaded_checkpoint is None else loaded_checkpoint.get("step", 0),
                "heatmap_sigma": heatmap_sigma,
                "offset_loss_weight": offset_loss_weight,
                "center_loss_weight": center_loss_weight,
                "softargmax_temperature": softargmax_temperature,
                "use_distance_bias": use_distance_bias,
                "gamma": gamma,
                "heatmap_mode": heatmap_mode,
            },
        )

    data_iter = iter(dataloader)
    progress = tqdm(range(steps), desc="train", unit="step")
    checkpoint_config = {
        "batch": batch,
        "lr": lr,
        "crop_size": crop_size,
        "experiment_name": experiment_name,
        "heatmap_sigma": heatmap_sigma,
        "offset_loss_weight": offset_loss_weight,
        "center_loss_weight": center_loss_weight,
        "softargmax_temperature": softargmax_temperature,
        "use_distance_bias": use_distance_bias,
        "gamma": gamma,
        "heatmap_mode": heatmap_mode,
        "params": None if params is None else str(params),
        "loaded_step": None if loaded_checkpoint is None else loaded_checkpoint.get("step", 0),
    }
    last_saved_step = 0
    completed_steps = 0
    best_step_loss = float("inf")
    steps_without_improvement = 0
    early_stop_patience = 5
    train_start_time = time.perf_counter()
    for step in progress:
        early_stop_triggered = False
        step_start_time = time.perf_counter()
        batch_data = next(data_iter)
        frames = _batch_inputs(batch_data).to(device, non_blocking=True)
        positions = _batch_positions(batch_data).to(device, dtype=frames.dtype, non_blocking=True)
        if heatmap_mode == "all":
            dummy_positions = _batch_dummy_positions(batch_data).to(
                device,
                dtype=frames.dtype,
                non_blocking=True,
            )
            dummy_mask = _batch_dummy_mask(batch_data).to(device, non_blocking=True)
        else:
            dummy_positions = None
            dummy_mask = None

        if frames.ndim != 5 or frames.shape[1:] != (300, 3, 600, 600):
            raise ValueError(
                "Expected dataloader frames with shape (B, 300, 3, 600, 600), "
                f"got {tuple(frames.shape)}."
            )

        step_losses: list[float] = []
        step_heatmap_losses: list[float] = []
        step_offset_losses: list[float] = []
        step_center_losses: list[float] = []
        semistep_count = frames.shape[1] - 4
        for semistep in range(semistep_count):
            inputs = make_semistep_inputs(
                frames,
                positions,
                semistep,
                crop_size=crop_size,
            )
            target_centers = make_semistep_targets(
                positions,
                semistep,
                crop_size=crop_size,
            )
            if heatmap_mode == "all":
                assert dummy_positions is not None
                assert dummy_mask is not None
                dummy_centers, semistep_dummy_mask = make_semistep_dummy_targets(
                    dummy_positions,
                    dummy_mask,
                    semistep,
                    crop_size=crop_size,
                )
            else:
                dummy_centers = None
                semistep_dummy_mask = None

            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss_output = compute_tracking_loss(
                output,
                target_centers,
                crop_size=crop_size,
                heatmap_loss_fn=heatmap_loss_fn,
                offset_loss_fn=offset_loss_fn,
                heatmap_sigma=heatmap_sigma,
                offset_loss_weight=offset_loss_weight,
                center_loss_weight=center_loss_weight,
                softargmax_temperature=softargmax_temperature,
                use_distance_bias=use_distance_bias,
                gamma=gamma,
                heatmap_mode=heatmap_mode,
                dummy_centers=dummy_centers,
                dummy_mask=semistep_dummy_mask,
            )

            loss_output.loss.backward()
            optimizer.step()

            global_semistep = step * semistep_count + semistep
            metrics = {
                "loss": float(loss_output.loss.detach().cpu()),
                "heatmap_loss": float(loss_output.heatmap_loss.detach().cpu()),
                "offset_loss": float(loss_output.offset_loss.detach().cpu()),
                "center_loss": float(loss_output.center_loss.detach().cpu()),
                "step": step,
                "semistep": semistep,
            }
            step_losses.append(metrics["loss"])
            step_heatmap_losses.append(metrics["heatmap_loss"])
            step_offset_losses.append(metrics["offset_loss"])
            step_center_losses.append(metrics["center_loss"])
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    loss=metrics["loss"],
                    heatmap=metrics["heatmap_loss"],
                    offset=metrics["offset_loss"],
                    center=metrics["center_loss"],
                    semistep=semistep,
                )
            if wandb_run is not None:
                wandb_run.log(metrics, step=global_semistep)

        step_metrics = {
            "step_loss": float(np.mean(step_losses)),
            "step_heatmap_loss": float(np.mean(step_heatmap_losses)),
            "step_offset_loss": float(np.mean(step_offset_losses)),
            "step_center_loss": float(np.mean(step_center_losses)),
            "step": step + 1,
        }
        _sync_device(device)
        step_train_seconds = float(time.perf_counter() - step_start_time)
        train_elapsed_seconds = float(time.perf_counter() - train_start_time)
        step_metrics["step_train_seconds"] = step_train_seconds
        step_metrics["train_elapsed_seconds"] = train_elapsed_seconds
        print(
            f"step {step + 1}/{steps} "
            f"loss={step_metrics['step_loss']:.6f} "
            f"heatmap={step_metrics['step_heatmap_loss']:.6f} "
            f"offset={step_metrics['step_offset_loss']:.6f} "
            f"center={step_metrics['step_center_loss']:.6f} "
            f"time={step_train_seconds:.2f}s"
        )
        if wandb_run is not None:
            wandb_run.log(step_metrics, step=(step + 1) * semistep_count)

        completed_steps = step + 1
        if step_metrics["step_loss"] < best_step_loss:
            best_step_loss = step_metrics["step_loss"]
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            print(
                f"[train] no loss improvement for "
                f"{steps_without_improvement}/{early_stop_patience} steps "
                f"(best={best_step_loss:.6f})"
            )
            if steps_without_improvement >= early_stop_patience:
                print(
                    f"[train] early stop: loss did not improve for "
                    f"{early_stop_patience} steps."
                )
                early_stop_triggered = True

        if save_step > 0 and (step + 1) % save_step == 0:
            checkpoint_path = run_dir / "checkpoints" / f"step_{step + 1:06d}.pt"
            save_checkpoint(
                model,
                optimizer,
                checkpoint_path,
                step=step + 1,
                config=checkpoint_config,
            )
            last_saved_step = step + 1
            print(f"[train] checkpoint saved: {checkpoint_path}")

        if eval_step > 0 and (early_stop_triggered or (step + 1) % eval_step == 0):
            eval_metrics = eval(
                model,
                frames[0].detach().cpu(),
                positions[0].detach().cpu(),
                dummy_positions=None if dummy_positions is None else dummy_positions[0].detach().cpu(),
                dummy_mask=None if dummy_mask is None else dummy_mask[0].detach().cpu(),
                output_dir=run_dir / "eval",
                name=f"step_{step + 1:06d}",
                device=device,
                crop_size=crop_size,
                heatmap_sigma=heatmap_sigma,
                offset_loss_weight=offset_loss_weight,
                center_loss_weight=center_loss_weight,
                softargmax_temperature=softargmax_temperature,
                use_distance_bias=use_distance_bias,
                gamma=gamma,
                train_elapsed_seconds=train_elapsed_seconds,
                step_train_seconds=step_train_seconds,
                heatmap_mode=heatmap_mode,
                experiment_name=experiment_name,
            )
            print(
                f"eval step {step + 1}: "
                f"accuracy={eval_metrics['accuracy']:.4f} "
                f"loss={eval_metrics['loss']:.6f} "
                f"distance={eval_metrics['standard_distance']:.3f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/accuracy": eval_metrics["accuracy"],
                        "eval/loss": eval_metrics["loss"],
                        "eval/heatmap_loss": eval_metrics["heatmap_loss"],
                        "eval/offset_loss": eval_metrics["offset_loss"],
                        "eval/center_loss": eval_metrics["center_loss"],
                        "eval/standard_distance": eval_metrics["standard_distance"],
                        "eval/eval_time_seconds": eval_metrics["eval_time_seconds"],
                        "eval/train_elapsed_seconds": eval_metrics["train_elapsed_seconds"],
                        "eval/step_train_seconds": eval_metrics["step_train_seconds"],
                    },
                    step=(step + 1) * semistep_count,
                )

        if early_stop_triggered:
            break

    if wandb_run is not None:
        wandb_run.finish()
    if completed_steps > 0 and last_saved_step != completed_steps:
        checkpoint_path = run_dir / "checkpoints" / f"step_{completed_steps:06d}_final.pt"
        save_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            step=completed_steps,
            config=checkpoint_config,
        )
        print(f"[train] final checkpoint saved: {checkpoint_path}")
    print(f"[train] done: {run_dir}")
    return run_dir


def train_gru(
    model: LieDA,
    dataloader: DataLoader | None = None,
    *,
    device: torch.device | str | None = None,
    heatmap_params: str | Path | None = None,
    params: str | Path | None = None,
    steps: int = 100,
    batch: int = 16,
    lr: float = 1e-3,
    crop_size: int = 576,
    eval_step: int = 20,
    save_step: int = 20,
    run_root: str | Path = "runs",
    experiment_name: str | None = None,
    heatmap_sigma: float = 4.0,
    offset_loss_weight: float = 1.0,
    center_loss_weight: float = 0.01,
    softargmax_temperature: float = 0.25,
    sequence_chunk_length: int = 8,
    heatmap_freeze: bool = True,
    GRU_freeze: bool = False,
    wandb_active: bool = False,
    wandb_project: str = "LieDA",
    use_distance_bias: bool = False,
    gamma: float = 0.0,
) -> Path:
    if sequence_chunk_length < 1:
        raise ValueError("sequence_chunk_length must be at least 1.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model.to(device)
    model.set_heatmap_freeze(heatmap_freeze)
    model.set_GRU_freeze(GRU_freeze)

    loaded_heatmap_checkpoint: dict[str, Any] | None = None
    if heatmap_params is not None:
        loaded_heatmap_checkpoint = load_heatmap_checkpoint(
            model,
            heatmap_params,
            device=device,
            run_root=run_root,
        )

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters. Set heatmap_freeze=False or GRU_freeze=False.")
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr)

    loaded_checkpoint: dict[str, Any] | None = None
    if params is not None:
        loaded_checkpoint = load_checkpoint(
            model,
            optimizer,
            params,
            device=device,
            run_root=run_root,
        )

    heatmap_loss_fn = nn.BCEWithLogitsLoss()
    offset_loss_fn = nn.SmoothL1Loss(reduction="sum")
    if dataloader is None:
        dataloader = my_dataloader(
            difficulty=1,
            batch=batch,
            pin_memory=False,
            #pin_memory=device.type == "cuda",
        )

    run_dir = make_run_dir(run_root, experiment_name=experiment_name)
    print(f"[train_gru] run directory: {run_dir}")
    wandb_run = None
    if wandb_active:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb_active=True requires wandb. Install it with: pip install wandb"
            ) from exc
        wandb_run = wandb.init(
            project=wandb_project,
            config={
                "train_mode": "gru",
                "steps": steps,
                "batch": batch,
                "lr": lr,
                "crop_size": crop_size,
                "eval_step": eval_step,
                "save_step": save_step,
                "run_dir": str(run_dir),
                "experiment_name": experiment_name,
                "heatmap_sigma": heatmap_sigma,
                "offset_loss_weight": offset_loss_weight,
                "center_loss_weight": center_loss_weight,
                "softargmax_temperature": softargmax_temperature,
                "sequence_chunk_length": sequence_chunk_length,
                "heatmap_freeze": heatmap_freeze,
                "GRU_freeze": GRU_freeze,
                "use_distance_bias": use_distance_bias,
                "gamma": gamma,
                "heatmap_params": None if heatmap_params is None else str(heatmap_params),
                "params": None if params is None else str(params),
                "loaded_heatmap_step": (
                    None
                    if loaded_heatmap_checkpoint is None
                    else loaded_heatmap_checkpoint.get("step", 0)
                ),
                "loaded_step": None if loaded_checkpoint is None else loaded_checkpoint.get("step", 0),
            },
        )
    checkpoint_config = {
        "train_mode": "gru",
        "batch": batch,
        "lr": lr,
        "crop_size": crop_size,
        "experiment_name": experiment_name,
        "eval_step": eval_step,
        "save_step": save_step,
        "heatmap_sigma": heatmap_sigma,
        "offset_loss_weight": offset_loss_weight,
        "center_loss_weight": center_loss_weight,
        "softargmax_temperature": softargmax_temperature,
        "sequence_chunk_length": sequence_chunk_length,
        "heatmap_freeze": heatmap_freeze,
        "GRU_freeze": GRU_freeze,
        "use_distance_bias": use_distance_bias,
        "gamma": gamma,
        "heatmap_params": None if heatmap_params is None else str(heatmap_params),
        "params": None if params is None else str(params),
        "loaded_heatmap_step": (
            None if loaded_heatmap_checkpoint is None else loaded_heatmap_checkpoint.get("step", 0)
        ),
        "loaded_step": None if loaded_checkpoint is None else loaded_checkpoint.get("step", 0),
    }

    data_iter = iter(dataloader)
    progress = tqdm(range(steps), desc="train_gru", unit="step")
    last_saved_step = 0
    completed_steps = 0
    best_step_loss = float("inf")
    steps_without_improvement = 0
    early_stop_patience = 5
    train_start_time = time.perf_counter()
    for step in progress:
        early_stop_triggered = False
        step_start_time = time.perf_counter()
        batch_data = next(data_iter)
        frames = _batch_inputs(batch_data).to(device, non_blocking=True)
        positions = _batch_positions(batch_data).to(device, dtype=frames.dtype, non_blocking=True)
        try:
            dummy_positions = _batch_dummy_positions(batch_data).to(
                device,
                dtype=frames.dtype,
                non_blocking=True,
            )
            dummy_mask = _batch_dummy_mask(batch_data).to(device, non_blocking=True)
        except KeyError:
            dummy_positions = None
            dummy_mask = None
        if frames.ndim != 5 or frames.shape[1:] != (300, 3, 600, 600):
            raise ValueError(
                "Expected dataloader frames with shape (B, 300, 3, 600, 600), "
                f"got {tuple(frames.shape)}."
            )

        model.train()
        if heatmap_freeze:
            model.heatmap_model.eval()
        hidden: list[Tensor] | None = None
        step_losses: list[float] = []
        step_heatmap_losses: list[float] = []
        step_offset_losses: list[float] = []
        step_center_losses: list[float] = []
        semistep_count = frames.shape[1] - 4

        for chunk_start in range(0, semistep_count, sequence_chunk_length):
            chunk_end = min(chunk_start + sequence_chunk_length, semistep_count)
            optimizer.zero_grad(set_to_none=True)
            chunk_loss = frames.new_zeros(())
            chunk_size = 0
            chunk_heatmap_losses: list[float] = []
            chunk_offset_losses: list[float] = []
            chunk_center_losses: list[float] = []

            for semistep in range(chunk_start, chunk_end):
                inputs = make_semistep_inputs(
                    frames,
                    positions,
                    semistep,
                    crop_size=crop_size,
                )
                target_centers = make_semistep_targets(
                    positions,
                    semistep,
                    crop_size=crop_size,
                )
                output, hidden, _all_output = model.forward_step(inputs, hidden)
                loss_output = compute_tracking_loss(
                    output,
                    target_centers,
                    crop_size=crop_size,
                    heatmap_loss_fn=heatmap_loss_fn,
                    offset_loss_fn=offset_loss_fn,
                    heatmap_sigma=heatmap_sigma,
                    offset_loss_weight=offset_loss_weight,
                    center_loss_weight=center_loss_weight,
                    softargmax_temperature=softargmax_temperature,
                    heatmap_mode="target",
                )
                chunk_loss = chunk_loss + loss_output.loss
                chunk_size += 1
                chunk_heatmap_losses.append(float(loss_output.heatmap_loss.detach().cpu()))
                chunk_offset_losses.append(float(loss_output.offset_loss.detach().cpu()))
                chunk_center_losses.append(float(loss_output.center_loss.detach().cpu()))

            chunk_loss = chunk_loss / max(chunk_size, 1)
            chunk_loss.backward()
            optimizer.step()
            hidden = _detach_hidden(hidden)

            loss_value = float(chunk_loss.detach().cpu())
            heatmap_loss_value = float(np.mean(chunk_heatmap_losses))
            offset_loss_value = float(np.mean(chunk_offset_losses))
            center_loss_value = float(np.mean(chunk_center_losses))
            step_losses.append(loss_value)
            step_heatmap_losses.append(heatmap_loss_value)
            step_offset_losses.append(offset_loss_value)
            step_center_losses.append(center_loss_value)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    loss=loss_value,
                    heatmap=heatmap_loss_value,
                    offset=offset_loss_value,
                    center=center_loss_value,
                    chunk=f"{chunk_start}:{chunk_end}",
                )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "loss": loss_value,
                        "heatmap_loss": heatmap_loss_value,
                        "offset_loss": offset_loss_value,
                        "center_loss": center_loss_value,
                        "step": step,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,
                    },
                    step=step * semistep_count + chunk_end,
                )

        completed_steps = step + 1
        _sync_device(device)
        step_train_seconds = float(time.perf_counter() - step_start_time)
        train_elapsed_seconds = float(time.perf_counter() - train_start_time)
        step_loss = float(np.mean(step_losses))
        step_heatmap_loss = float(np.mean(step_heatmap_losses))
        step_offset_loss = float(np.mean(step_offset_losses))
        step_center_loss = float(np.mean(step_center_losses))
        print(
            f"gru step {step + 1}/{steps} "
            f"loss={step_loss:.6f} "
            f"heatmap={step_heatmap_loss:.6f} "
            f"offset={step_offset_loss:.6f} "
            f"center={step_center_loss:.6f} "
            f"time={step_train_seconds:.2f}s "
            f"elapsed={train_elapsed_seconds:.2f}s"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "step_loss": step_loss,
                    "step_heatmap_loss": step_heatmap_loss,
                    "step_offset_loss": step_offset_loss,
                    "step_center_loss": step_center_loss,
                    "step": step + 1,
                    "step_train_seconds": step_train_seconds,
                    "train_elapsed_seconds": train_elapsed_seconds,
                },
                step=(step + 1) * semistep_count,
            )

        if step_loss < best_step_loss:
            best_step_loss = step_loss
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            print(
                f"[train_gru] no loss improvement for "
                f"{steps_without_improvement}/{early_stop_patience} steps "
                f"(best={best_step_loss:.6f})"
            )
            if steps_without_improvement >= early_stop_patience:
                print(
                    f"[train_gru] early stop: loss did not improve for "
                    f"{early_stop_patience} steps."
                )
                early_stop_triggered = True

        if save_step > 0 and (step + 1) % save_step == 0:
            checkpoint_path = run_dir / "checkpoints" / f"step_{step + 1:06d}.pt"
            save_checkpoint(
                model,
                optimizer,
                checkpoint_path,
                step=step + 1,
                config=checkpoint_config,
            )
            last_saved_step = step + 1
            print(f"[train_gru] checkpoint saved: {checkpoint_path}")

        if eval_step > 0 and (early_stop_triggered or (step + 1) % eval_step == 0):
            eval_metrics = eval(
                model,
                frames[0].detach().cpu(),
                positions[0].detach().cpu(),
                dummy_positions=None if dummy_positions is None else dummy_positions[0].detach().cpu(),
                dummy_mask=None if dummy_mask is None else dummy_mask[0].detach().cpu(),
                output_dir=run_dir / "eval",
                name=f"step_{step + 1:06d}",
                device=device,
                crop_size=crop_size,
                heatmap_sigma=heatmap_sigma,
                offset_loss_weight=offset_loss_weight,
                center_loss_weight=center_loss_weight,
                softargmax_temperature=softargmax_temperature,
                use_distance_bias=use_distance_bias,
                gamma=gamma,
                train_elapsed_seconds=train_elapsed_seconds,
                step_train_seconds=step_train_seconds,
                heatmap_mode="target",
                experiment_name=experiment_name,
            )
            print(
                f"gru eval step {step + 1}: "
                f"accuracy={eval_metrics['accuracy']:.4f} "
                f"loss={eval_metrics['loss']:.6f} "
                f"distance={eval_metrics['standard_distance']:.3f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/accuracy": eval_metrics["accuracy"],
                        "eval/loss": eval_metrics["loss"],
                        "eval/heatmap_loss": eval_metrics["heatmap_loss"],
                        "eval/offset_loss": eval_metrics["offset_loss"],
                        "eval/center_loss": eval_metrics["center_loss"],
                        "eval/standard_distance": eval_metrics["standard_distance"],
                        "eval/eval_time_seconds": eval_metrics["eval_time_seconds"],
                        "eval/train_elapsed_seconds": eval_metrics["train_elapsed_seconds"],
                        "eval/step_train_seconds": eval_metrics["step_train_seconds"],
                        **(
                            {
                                "eval/heatmap_eval_loss": eval_metrics["heatmap_eval_loss"],
                                "eval/heatmap_heatmap_loss": eval_metrics["heatmap_heatmap_loss"],
                                "eval/heatmap_offset_loss": eval_metrics["heatmap_offset_loss"],
                                "eval/heatmap_center_loss": eval_metrics["heatmap_center_loss"],
                                "eval/heatmap_accuracy": eval_metrics["heatmap_accuracy"],
                                "eval/heatmap_standard_distance": eval_metrics["heatmap_standard_distance"],
                            }
                            if "heatmap_eval_loss" in eval_metrics
                            else {}
                        ),
                    },
                    step=(step + 1) * semistep_count,
                )

        if early_stop_triggered:
            break

    if wandb_run is not None:
        wandb_run.finish()
    if completed_steps > 0 and last_saved_step != completed_steps:
        checkpoint_path = run_dir / "checkpoints" / f"step_{completed_steps:06d}_final.pt"
        save_checkpoint(
            model,
            optimizer,
            checkpoint_path,
            step=completed_steps,
            config=checkpoint_config,
        )
        print(f"[train_gru] final checkpoint saved: {checkpoint_path}")
    print(f"[train_gru] done: {run_dir}")
    return run_dir


def train(
    model: Any,
    dataloader: DataLoader | None = None,
    *,
    model_type: str = "heatmap",
    **kwargs: Any,
) -> Path:
    if model_type == "heatmap":
        return train_heatmap(model, dataloader, **kwargs)
    if model_type == "gru":
        if not isinstance(model, LieDA):
            raise TypeError("model_type='gru' requires a LieDA model.")
        return train_gru(model, dataloader, **kwargs)
    raise ValueError(f"model_type must be 'heatmap' or 'gru', got {model_type!r}.")


if __name__ == "__main__":
    common = {
        "steps": 300,
        "batch": 2,
        "eval_step": 12,
        "save_step": 12,
        "heatmap_sigma": 4,
        "use_distance_bias": False,
    }

    experiment_name = input("Write experiment name or skip: ").strip() or None
    common["experiment_name"] = experiment_name

    # 1. difficulty 1: train PartialUNet heatmap model only.
    #train(
    #   PartialHeatUNet(in_channels=7),
    #   my_dataloader(difficulty=1, batch=common["batch"]),
    #   model_type="heatmap",
    #   heatmap_mode="all",
    #   **common,
    #)

    # 2. difficulty 1: load latest heatmap into LieDA, freeze heatmap, train GRU only.
    #train(
    #      LieDA(
    #          PartialHeatUNet(in_channels=7),
    #          heatmap_freeze=True,
    #          GRU_freeze=False,
    #      ),
    #      my_dataloader(difficulty=1, batch=common["batch"]),
    #      model_type="gru",
    #      heatmap_params="latest",
    #      heatmap_freeze=True,
    #      GRU_freeze=False,
    #      **common,
    #  )

    # 3. difficulty 2: train heatmap side only.
    #train(
    #    LieDA(
    #         PartialHeatUNet(in_channels=7),
    #         heatmap_freeze=False,
    #         GRU_freeze=True,
    #    ),
    #    my_dataloader(difficulty=2, batch=common["batch"]),
    #    model_type="gru",
    #    params="latest",
    #    heatmap_freeze=False,
    #    GRU_freeze=True,
    #    **common,
    # )

    # 4. difficulty 2: freeze heatmap, train GRU only.
    train(
         LieDA(
             PartialHeatUNet(in_channels=7),
             heatmap_freeze=True,
             GRU_freeze=False,
         ),
         my_dataloader(difficulty=2, batch=common["batch"]),
         model_type="gru",
         params="latest",
         heatmap_freeze=True,
         GRU_freeze=False,
         **common,
     )

    # 5. difficulty 3: train heatmap side only.
    # train(
    #     LieDA(
    #         PartialHeatUNet(in_channels=7),
    #         heatmap_freeze=False,
    #         GRU_freeze=True,
    #     ),
    #     my_dataloader(difficulty=3, batch=common["batch"]),
    #     model_type="gru",
    #     params="latest",
    #     heatmap_freeze=False,
    #     GRU_freeze=True,
    #     **common,
    # )

    # 6. difficulty 3: freeze heatmap, train GRU only.
    # train(
    #     LieDA(
    #         PartialHeatUNet(in_channels=7),
    #         heatmap_freeze=True,
    #         GRU_freeze=False,
    #     ),
    #     my_dataloader(difficulty=3, batch=common["batch"]),
    #     model_type="gru",
    #     params="latest",
    #     heatmap_freeze=True,
    #     GRU_freeze=False,
    #     **common,
    # )
