from pathlib import Path
import random
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


SCALE_PATTERN = re.compile(r"scale_(\d+)_")

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


def make_distorted_background(
    texture: Image.Image | np.ndarray,
    frame_idx: int,
    out_w: int,
    out_h: int,
    fps: int = 30,
    period_x: float = 200.0,
    period_y: float = 200.0,
    orbit_radius: float = 100.0,
    orbit_speed: float = 1.2,
    wobble_speed: float = 1.6,
) -> Image.Image:
    if isinstance(texture, Image.Image):
        texture_array = np.asarray(texture.convert("RGB"))
    else:
        texture_array = texture

    t = frame_idx / fps
    texture_h, texture_w = texture_array.shape[:2]

    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)

    x = xx - out_w * 0.5
    y = yy - out_h * 0.5

    xr = x
    yr = y

    center_x = 50.0
    center_y = 50.0

    kx = 2.0 * np.pi / period_x
    ky = 2.0 * np.pi / period_y
    phase_x = wobble_speed * t
    phase_y = wobble_speed * t * 0.7

    cx = np.cos(kx * (xr - center_x) + phase_x)
    cy = np.cos(ky * (yr - center_y) + phase_y)
    sx = np.sin(kx * (xr - center_x) + phase_x)
    sy = np.sin(ky * (yr - center_y) + phase_y)

    z = cx * cy
    gamma = 1.8

    dhdx = -kx * sx * cy
    dhdy = -ky * cx * sy

    eps = 1e-6
    scale = gamma * (np.abs(z) + eps) ** (gamma - 1.0)
    dhdx *= scale
    dhdy *= scale

    distortion_strength = 180.0
    theta = orbit_speed * t
    sample_center_x = texture_w * 0.5 + orbit_radius * np.cos(theta)
    sample_center_y = texture_h * 0.5 + orbit_radius * np.sin(theta)

    u = xr + distortion_strength * dhdx + sample_center_x
    v = yr + distortion_strength * dhdy + sample_center_y

    map_x = np.mod(u, texture_w).astype(np.float32)
    map_y = np.mod(v, texture_h).astype(np.float32)

    frame = cv2.remap(
        texture_array,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )
    return Image.fromarray(frame, mode="RGB")

def make_mask(
    mask_type: int = 1,
    size: int = 50
) -> Image.Image:
    if mask_type == 0:
        mask_type = random.randint(1, 4) # circle, square, triangle, star
    
    if mask_type == 1:
        yy, xx = np.mgrid[0:size, 0:size]
        center = (size - 1) * 0.5
        radius = size * 0.5
        mask = ((xx - center) ** 2 + (yy - center) ** 2 <= radius**2).astype(np.uint8) * 255
        return Image.fromarray(mask, mode="L")


def make_target(
    n_frames: int,
    texture: Image.Image | np.ndarray | None = None,
    out_w: int = 600,
    out_h: int = 600,
    fps: int = 30,
    period_x: float = 200.0,
    period_y: float = 200.0,
    orbit_radius: float = 100.0,
    orbit_speed: float = 1.2,
    wobble_speed: float = 1.6,
) -> list[Image.Image]:
    if texture is None:
        texture, _ = make_base_image()

    frames: list[Image.Image] = []
    mask = make_mask(mask_type=1)
    mask_left = (out_w - mask.width) // 2
    mask_top = (out_h - mask.height) // 2
    for frame_idx in range(n_frames):
        frame = make_distorted_background(
            texture,
            frame_idx,
            out_w,
            out_h,
            fps=fps,
            period_x=period_x,
            period_y=period_y,
            orbit_radius=orbit_radius,
            orbit_speed=orbit_speed,
            wobble_speed=wobble_speed,
        )
        frame.paste(Image.new("RGB", mask.size, "white"), (mask_left, mask_top), mask)
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

    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    image, image_path = make_base_image()
    print(f"Selected image: {image_path}")

    frames = make_target(n_frames=300, texture=image)
    play_image_list(frames)
