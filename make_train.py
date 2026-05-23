from pathlib import Path
import random
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


SCALE_PATTERN = re.compile(r"scale_(\d+)_")

# 여러 개 만들 때는 candidates를 memoization해 놓는 게 나을 것 같음
# Pick a random KTH-TIPS image and preprocess it.
def make_base_image(
    data_dir: str | Path = "KTH_TIPS",
    max_scale: int = 5,
    output_size: tuple[int, int] = (800, 800),
    blur_radius: float = 0.4,
    noise_probability: float = 0.3,
    noise_std: float = 2.0,
    rng: random.Random | None = None,
) -> tuple[Image.Image, Path]:
    data_dir = Path(data_dir)
    rng = rng or random

    candidates: list[Path] = []
    for path in data_dir.rglob("*.png"):
        match = SCALE_PATTERN.search(path.name)
        if match and int(match.group(1)) <= max_scale:
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"No PNG images with scale <= {max_scale} found in {data_dir}")

    image_path = rng.choice(candidates)
    image = Image.open(image_path).convert("RGB")
    image = image.resize(output_size, Image.Resampling.BICUBIC)
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if rng.random() < noise_probability:
        pixels = np.asarray(image, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=pixels.shape)
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="RGB")

    return image, image_path

def make_background(
    n_frames: int,
    omega: float,
    base_image: Image.Image | None = None,
    frame_size: int = 600,
) -> list[Image.Image]:
    if base_image is None:
        base_image, _ = make_base_image()
    frames: list[Image.Image] = []

    for t in range(n_frames):
        left = round(100 + 100 * np.sin(t * omega))
        top = round(100 + 100 * np.cos(t * omega))
        frame = base_image.crop((left, top, left + frame_size, top + frame_size))
        frames.append(frame)

    return frames


def play_image_list(
    images: list[Image.Image],
    window_name: str = "frames",
    delay_ms: int = 33,
) -> None:
    for image in images:
        frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, frame)
        cv2.waitKey(delay_ms)

        # if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
        #     break

    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    image, image_path = make_base_image()
    print(f"Selected image: {image_path}")

    frames = make_background(n_frames=300, omega=0.05, base_image=image)
    play_image_list(frames)
