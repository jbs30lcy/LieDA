from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from alchemy import make_nothing
from make_train import make_target


def _image_to_tensor(image: Any) -> Tensor:
    image_array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image_array).permute(2, 0, 1).contiguous()


class NothingDataset(IterableDataset[dict[str, Tensor]]):
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

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        base_seed = self.seed
        if base_seed is None:
            base_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        rng = np.random.default_rng(base_seed + worker_id)

        while True:
            image, _source = make_nothing(
                output_size=(self.output_size, self.output_size),
                seed=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            frames, positions = make_target(
                n_frames=self.n_frames,
                texture=image,
                out_w=self.output_size,
                out_h=self.output_size,
            )
            frame_tensor = torch.stack([_image_to_tensor(frame) for frame in frames])
            position_tensor = torch.from_numpy(positions).to(torch.float32)
            yield {"frames": frame_tensor, "positions": position_tensor}


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
