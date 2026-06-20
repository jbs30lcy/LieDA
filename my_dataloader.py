from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from alchemy import make_aluminium_foil, make_ground, make_nothing, make_tiling
from make_train import make_base_image, make_target


def _image_to_tensor(image: Any) -> Tensor:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image_array).permute(2, 0, 1).contiguous()


def _make_sample(texture: Any, *, n_frames: int, output_size: int) -> dict[str, Tensor]:
    frames, positions, dummy_positions, dummy_mask = make_target(
        n_frames=n_frames,
        texture=texture,
        out_w=output_size,
        out_h=output_size,
    )
    frame_tensor = torch.stack([_image_to_tensor(frame) for frame in frames])
    position_tensor = torch.from_numpy(positions).to(torch.float32)
    dummy_position_tensor = torch.from_numpy(dummy_positions).to(torch.float32)
    dummy_mask_tensor = torch.from_numpy(dummy_mask).to(torch.bool)
    return {
        "frames": frame_tensor,
        "positions": position_tensor,
        "dummy_positions": dummy_position_tensor,
        "dummy_mask": dummy_mask_tensor,
    }


class TextureDataset(IterableDataset[dict[str, Tensor]]):
    def __init__(
        self,
        *,
        n_frames: int = 300,
        output_size: int = 600,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.output_size = output_size
        self.seed = seed

    def make_texture(self, rng: np.random.Generator) -> Any:
        raise NotImplementedError

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        base_seed = self.seed
        if base_seed is None:
            base_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        rng = np.random.default_rng(base_seed + worker_id)

        while True:
            yield _make_sample(
                self.make_texture(rng),
                n_frames=self.n_frames,
                output_size=self.output_size,
            )


class NothingDataset(TextureDataset):
    def make_texture(self, rng: np.random.Generator) -> Any:
        image, _source = make_nothing(
            output_size=(self.output_size, self.output_size),
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        return image


class AlchemyDataset(TextureDataset):
    def make_texture(self, rng: np.random.Generator) -> Any:
        maker = rng.choice((make_aluminium_foil, make_ground, make_tiling))
        image, _source = maker(
            output_size=(self.output_size, self.output_size),
            seed=int(rng.integers(0, np.iinfo(np.int32).max)),
        )
        return image


class NormalDataset(TextureDataset):
    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        *,
        n_frames: int = 300,
        output_size: int = 600,
        seed: int | None = None,
        kth_tips_dir: str | Path = "KTH_TIPS",
    ) -> None:
        super().__init__(n_frames=n_frames, output_size=output_size, seed=seed)
        self.kth_tips_dir = Path(kth_tips_dir)
        self.kth_tips_paths = [
            path
            for path in self.kth_tips_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in self.image_suffixes
        ]
        if not self.kth_tips_paths:
            raise FileNotFoundError(f"No KTH_TIPS images found in {self.kth_tips_dir}.")

    def make_texture(self, rng: np.random.Generator) -> Any:
        choice = float(rng.random())
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        output_size = (self.output_size, self.output_size)

        if choice < 0.5:
            image_path = self.kth_tips_paths[int(rng.integers(0, len(self.kth_tips_paths)))]
            image, _source = make_base_image(
                data_dir=self.kth_tips_dir,
                path=image_path,
                output_size=output_size,
            )
        elif choice < 0.8:
            image, _source = make_ground(output_size=output_size, seed=seed)
        elif choice < 0.9:
            image, _source = make_aluminium_foil(output_size=output_size, seed=seed)
        else:
            image, _source = make_tiling(output_size=output_size, seed=seed)
        return image


def my_dataloader(
    *,
    difficulty: int = 1,
    batch: int = 16,
    n_frames: int = 300,
    output_size: int = 600,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    seed: int | None = None,
) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if difficulty == 1:
        return DataLoader(
            NothingDataset(n_frames=n_frames, output_size=output_size, seed=seed),
            batch_size=batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    if difficulty == 2:
        return DataLoader(
            AlchemyDataset(n_frames=n_frames, output_size=output_size, seed=seed),
            batch_size=batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    if difficulty == 3:
        return DataLoader(
            NormalDataset(n_frames=n_frames, output_size=output_size, seed=seed),
            batch_size=batch,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    raise ValueError(f"difficulty must be 1, 2, or 3, got {difficulty}.")
