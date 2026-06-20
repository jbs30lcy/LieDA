from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
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

from models import TinyNet, PartialHeatUNet, TrackerOutput, apply_distance_bias, decode_centers
from my_dataloader import my_dataloader


@dataclass(frozen=True)
class LossOutput:
    loss: Tensor
    heatmap_loss: Tensor
    offset_loss: Tensor


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


def compute_tracking_loss(
    output: TrackerOutput | Tensor,
    target_centers: Tensor,
    *,
    crop_size: int,
    heatmap_loss_fn: nn.Module,
    offset_loss_fn: nn.Module,
    heatmap_sigma: float = 4.0,
    offset_loss_weight: float = 1.0,
    use_distance_bias: bool = False,
    gamma: float = 0.0,
    heatmap_mode: str = "target",
    dummy_centers: Tensor | None = None,
    dummy_mask: Tensor | None = None,
) -> LossOutput:
    if heatmap_mode not in {"target", "all"}:
        raise ValueError(f"heatmap_mode must be 'target' or 'all', got {heatmap_mode!r}.")
    heatmap = output.heatmap if isinstance(output, TrackerOutput) else output
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
    if isinstance(output, TrackerOutput):
        offset_difference = output.offset - target_offset
        offset_loss = offset_loss_fn(
            offset_difference * offset_mask,
            torch.zeros_like(offset_difference),
        ) / offset_mask.sum().clamp_min(1.0)
        loss = heatmap_loss + offset_loss_weight * offset_loss
    else:
        offset_loss = heatmap_loss.new_zeros(())
        loss = heatmap_loss

    return LossOutput(
        loss=loss,
        heatmap_loss=heatmap_loss,
        offset_loss=offset_loss,
    )


def make_run_dir(root: str | Path = "runs") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{timestamp}_{uuid.uuid4().hex[:8]}"
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
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        loaded_step = int(checkpoint.get("step", 0))
    else:
        model.load_state_dict(checkpoint)
        loaded_step = 0
        checkpoint = {"step": loaded_step}

    print(f"[train] params loaded: {checkpoint_path} (step={loaded_step})")
    return checkpoint


def _frames_to_uint8_hwc(frames: Tensor) -> np.ndarray:
    frames_cpu = frames.detach().cpu().clamp(0.0, 1.0)
    frames_hwc = frames_cpu.permute(0, 2, 3, 1).numpy()
    return (frames_hwc * 255.0).round().astype(np.uint8)


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
    use_distance_bias: bool = False,
    gamma: float = 0.0,
    hit_radius: float = 55.2,
    fps: int = 30,
    train_elapsed_seconds: float | None = None,
    step_train_seconds: float | None = None,
    heatmap_mode: str = "target",
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

    was_training = model.training
    model.eval()
    frames_batch = frames.unsqueeze(0).to(device)
    positions_batch = positions.unsqueeze(0).to(device, dtype=frames_batch.dtype)
    if heatmap_mode == "all":
        if dummy_positions is None or dummy_mask is None:
            raise ValueError(
                "dummy_positions and dummy_mask are required when heatmap_mode='all'."
            )
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
        if heatmap_mode == "all":
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
        output = model(inputs)
        loss_output = compute_tracking_loss(
            output,
            target_centers,
            crop_size=crop_size,
            heatmap_loss_fn=heatmap_loss_fn,
            offset_loss_fn=offset_loss_fn,
            heatmap_sigma=heatmap_sigma,
            offset_loss_weight=offset_loss_weight,
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

    predicted_tensor = torch.stack(predicted_centers)
    label_tensor = torch.stack(label_centers)
    distances = torch.linalg.norm(predicted_tensor - label_tensor, dim=1)
    hits = distances <= hit_radius # TODO : hit 판정을 radius로 판정하는 거 고쳐야 됨
    metrics = {
        "name": name,
        "num_predictions": int(predicted_tensor.shape[0]),
        "accuracy": float(hits.float().mean().item()),
        "hit_radius": float(hit_radius),
        "heatmap_mode": heatmap_mode,
        "loss": float(np.mean(losses)),
        "heatmap_loss": float(np.mean(heatmap_losses)),
        "offset_loss": float(np.mean(offset_losses)),
        "standard_distance": float(distances.mean().item()),
        "median_distance": float(distances.median().item()),
        "video_path": str(video_path),
        "metrics_path": str(metrics_path),
        "play_command": f"python -c \"import os; os.startfile(r'{video_path}')\"",
        "predicted_centers": predicted_tensor.tolist(),
        "label_centers": label_tensor.tolist(),
    }

    _draw_centers_video(
        frames,
        label_tensor,
        predicted_tensor,
        video_path,
        fps=fps,
        start_frame=4,
    )
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


def train(
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
    heatmap_sigma: float = 4.0,
    offset_loss_weight: float = 1.0,
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
            pin_memory=device.type == "cuda",
        )

    run_dir = make_run_dir(run_root)
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
                "params": None if params is None else str(params),
                "loaded_step": None if loaded_checkpoint is None else loaded_checkpoint.get("step", 0),
                "heatmap_sigma": heatmap_sigma,
                "offset_loss_weight": offset_loss_weight,
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
        "heatmap_sigma": heatmap_sigma,
        "offset_loss_weight": offset_loss_weight,
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
                "step": step,
                "semistep": semistep,
            }
            step_losses.append(metrics["loss"])
            step_heatmap_losses.append(metrics["heatmap_loss"])
            step_offset_losses.append(metrics["offset_loss"])
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    loss=metrics["loss"],
                    heatmap=metrics["heatmap_loss"],
                    offset=metrics["offset_loss"],
                    semistep=semistep,
                )
            if wandb_run is not None:
                wandb_run.log(metrics, step=global_semistep)

        step_metrics = {
            "step_loss": float(np.mean(step_losses)),
            "step_heatmap_loss": float(np.mean(step_heatmap_losses)),
            "step_offset_loss": float(np.mean(step_offset_losses)),
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
                break

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

        if eval_step > 0 and (step + 1) % eval_step == 0:
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
                use_distance_bias=use_distance_bias,
                gamma=gamma,
                train_elapsed_seconds=train_elapsed_seconds,
                step_train_seconds=step_train_seconds,
                heatmap_mode=heatmap_mode,
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
                        "eval/standard_distance": eval_metrics["standard_distance"],
                        "eval/eval_time_seconds": eval_metrics["eval_time_seconds"],
                        "eval/train_elapsed_seconds": eval_metrics["train_elapsed_seconds"],
                        "eval/step_train_seconds": eval_metrics["step_train_seconds"],
                    },
                    step=(step + 1) * semistep_count,
                )

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


if __name__ == "__main__":
    # train(model, steps=100, batch=4, eval_step=10, save_step=10, use_distance_bias=True, gamma=-0.01)

    model1 = TinyNet(in_channels=7)
    train(
        model1,
        steps=1,
        batch=1,
        eval_step=1,
        save_step=1,
        heatmap_sigma=1,
        use_distance_bias=False,
        heatmap_mode="all",
    )

    model2 = PartialHeatUNet(in_channels=7)
    train(
        model2,
        steps=1,
        batch=1,
        eval_step=1,
        save_step=1,
        heatmap_sigma=4,
        use_distance_bias=False,
        heatmap_mode="all",
    )
