from pathlib import Path
import random
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


SCALE_PATTERN = re.compile(r"scale_(\d+)_")
LensShape = dict[str, float | int | str]

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

def make_distorted_background(
    texture: Image.Image | np.ndarray,
    frame_idx: int,
    out_w: int,
    out_h: int,
    fps: int = 30,
    period_x: float = 200.0,
    period_y: float = 300.0,
    orbit_radius: float = 100.0,
    orbit_speed: float = 1.2,
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
    cx = np.cos(kx * (xr - center_x))
    cy = np.cos(ky * (yr - center_y))
    sx = np.sin(kx * (xr - center_x))
    sy = np.sin(ky * (yr - center_y))

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


def default_lens_circles(
    source_w: int,
    source_h: int,
    count: int = 8,
    rng: random.Random | None = None,
) -> list[tuple[float, float, float]]:
    rng = rng or random
    size = min(source_w, source_h)
    circles: list[tuple[float, float, float]] = []
    for _ in range(count):
        radius = rng.uniform(0.080, 0.092) * size
        center_x = rng.uniform(radius, source_w - radius)
        center_y = rng.uniform(radius, source_h - radius)
        circles.append((center_x, center_y, radius))
    return circles


def default_lens_shapes(
    source_w: int,
    source_h: int,
    count: int = 8,
    rng: random.Random | None = None,
) -> list[LensShape]:
    return [
        {"kind": "circle", "x": x, "y": y, "radius": radius}
        for x, y, radius in default_lens_circles(source_w, source_h, count=count, rng=rng)
    ]


def regular_polygon_points(
    center_x: float,
    center_y: float,
    radius: float,
    n_points: int,
    rotation: float = -np.pi / 2,
) -> np.ndarray:
    angles = rotation + np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points = np.stack(
        [
            center_x + radius * np.cos(angles),
            center_y + radius * np.sin(angles),
        ],
        axis=1,
    )
    return np.round(points).astype(np.int32)


def star_points(
    center_x: float,
    center_y: float,
    outer_radius: float,
    inner_radius: float | None = None,
    n_points: int = 5,
    rotation: float = -np.pi / 2,
) -> np.ndarray:
    inner_radius = inner_radius or outer_radius * 0.48
    angles = rotation + np.linspace(0, 2 * np.pi, n_points * 2, endpoint=False)
    radii = np.where(np.arange(n_points * 2) % 2 == 0, outer_radius, inner_radius)
    points = np.stack(
        [
            center_x + radii * np.cos(angles),
            center_y + radii * np.sin(angles),
        ],
        axis=1,
    )
    return np.round(points).astype(np.int32)


def draw_lens_shape(mask: np.ndarray, shape: LensShape) -> None:
    kind = str(shape.get("kind", "circle")).lower()
    center_x = float(shape.get("x", 0.0))
    center_y = float(shape.get("y", 0.0))
    radius = float(shape.get("radius", shape.get("size", 0.0)))

    if kind == "circle":
        cv2.circle(mask, (round(center_x), round(center_y)), round(radius), 255, -1, lineType=cv2.LINE_AA)
    elif kind in {"rectangle", "rect", "square"}:
        width = float(shape.get("width", radius * 2.0))
        height = float(shape.get("height", width if kind == "square" else radius * 2.0))
        left = round(center_x - width * 0.5)
        right = round(center_x + width * 0.5)
        top = round(center_y - height * 0.5)
        bottom = round(center_y + height * 0.5)
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1, lineType=cv2.LINE_AA)
    elif kind == "triangle":
        points = regular_polygon_points(
            center_x,
            center_y,
            radius,
            3,
            rotation=float(shape.get("rotation", -np.pi / 2)),
        )
        cv2.fillPoly(mask, [points], 255, lineType=cv2.LINE_AA)
    elif kind == "star":
        points = star_points(
            center_x,
            center_y,
            radius,
            inner_radius=float(shape.get("inner_radius", radius * 0.48)),
            n_points=int(shape.get("points", 5)),
            rotation=float(shape.get("rotation", -np.pi / 2)),
        )
        cv2.fillPoly(mask, [points], 255, lineType=cv2.LINE_AA)
    else:
        raise ValueError(f"Unsupported lens shape kind: {kind}")


def make_lens_mask(
    out_w: int,
    out_h: int,
    shapes: list[LensShape],
    crop_left: int = 0,
    crop_top: int = 0,
) -> np.ndarray:
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    for shape in shapes:
        shifted = dict(shape)
        shifted["x"] = float(shifted.get("x", 0.0)) - crop_left
        shifted["y"] = float(shifted.get("y", 0.0)) - crop_top
        draw_lens_shape(mask, shifted)
    return mask


def apply_lens_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    rim_width: float = 12.0,
    outer_shadow_width: float = 18.0,
    magnification: float = 1.05,
    rim_refraction: float = 10.0,
) -> np.ndarray:
    out_h, out_w = frame.shape[:2]
    mask = (mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return frame

    x = 0
    y = 0
    w = out_w
    h = out_h
    left = 0
    right = out_w
    top = 0
    bottom = out_h

    local_mask = mask[top:bottom, left:right]
    inside = local_mask > 0
    inside_distance = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5).astype(np.float32)
    outside_distance = cv2.distanceTransform(255 - local_mask, cv2.DIST_L2, 5).astype(np.float32)
    signed_distance = inside_distance - outside_distance

    inner_rim = np.exp(-((inside_distance - 1.0) / rim_width) ** 2) * inside
    outer_rim = np.exp(-((outside_distance - 1.0) / outer_shadow_width) ** 2) * (~inside)
    refractive_band = np.exp(-(signed_distance / rim_width) ** 2)

    yy, xx = np.mgrid[top:bottom, left:right].astype(np.float32)
    top_light = np.clip(1.0 - (yy - y) / max(h, 1), 0.0, 1.0)
    bottom_shadow = 1.0 - top_light

    center_x = x + w * 0.5
    center_y = y + h * 0.5
    dx = xx - center_x
    dy = yy - center_y
    radius = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-6)
    normal_x = dx / radius
    normal_y = dy / radius

    magnified_x = center_x + dx / magnification
    magnified_y = center_y + dy / magnification

    # The boundary in the reference behaves like a raised meniscus: the texture
    # is magnified inside, then pinched sideways in a thin ring near the edge.
    rim_offset = rim_refraction * refractive_band
    map_x = np.where(inside, magnified_x, xx) + normal_x * rim_offset
    map_y = np.where(inside, magnified_y, yy) + normal_y * rim_offset

    map_x = np.clip(map_x, 0, out_w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, out_h - 1).astype(np.float32)
    remapped = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)

    original = frame[top:bottom, left:right].astype(np.float32)
    alpha = cv2.GaussianBlur(local_mask, (0, 0), sigmaX=0.8).astype(np.float32) / 255.0
    composed = remapped * alpha[..., None] + original * (1.0 - alpha[..., None])

    highlight = 46.0 * inner_rim * top_light
    inner_shadow = 68.0 * inner_rim * bottom_shadow + 26.0 * inner_rim
    outer_shadow = 58.0 * outer_rim
    composed += highlight[..., None]
    composed -= inner_shadow[..., None]
    composed -= outer_shadow[..., None]

    result = frame.copy()
    result[top:bottom, left:right] = np.clip(composed, 0, 255).astype(np.uint8)
    return result


def apply_lens_circle(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    draw_lens_shape(mask, {"kind": "circle", "x": center_x, "y": center_y, "radius": radius})
    return apply_lens_mask(frame, mask)


def apply_lens_shapes(
    image: Image.Image,
    shapes: list[LensShape] | None = None,
    crop_left: int = 0,
    crop_top: int = 0,
    source_size: tuple[int, int] | None = None,
) -> Image.Image:
    frame = np.asarray(image.convert("RGB")).copy()
    out_h, out_w = frame.shape[:2]

    if shapes is None:
        source_w, source_h = source_size or (out_w, out_h)
        shapes = default_lens_shapes(source_w, source_h)

    effect_margin = 80
    padded_frame = cv2.copyMakeBorder(
        frame,
        effect_margin,
        effect_margin,
        effect_margin,
        effect_margin,
        borderType=cv2.BORDER_REFLECT_101,
    )
    padded_h, padded_w = padded_frame.shape[:2]
    mask = make_lens_mask(
        padded_w,
        padded_h,
        shapes,
        crop_left=crop_left - effect_margin,
        crop_top=crop_top - effect_margin,
    )
    padded_frame = apply_lens_mask(padded_frame, mask)
    frame = padded_frame[
        effect_margin : effect_margin + out_h,
        effect_margin : effect_margin + out_w,
    ]
    return Image.fromarray(frame, mode="RGB")


def apply_lens_circles(
    image: Image.Image,
    circles: list[tuple[float, float, float]] | None = None,
    crop_left: int = 0,
    crop_top: int = 0,
    source_size: tuple[int, int] | None = None,
) -> Image.Image:
    if circles is None:
        out_w, out_h = image.size
        source_w, source_h = source_size or (out_w, out_h)
        circles = default_lens_circles(source_w, source_h)

    shapes = [
        {"kind": "circle", "x": center_x, "y": center_y, "radius": radius}
        for center_x, center_y, radius in circles
    ]
    return apply_lens_shapes(
        image,
        shapes=shapes,
        crop_left=crop_left,
        crop_top=crop_top,
        source_size=source_size,
    )


def make_target(
    n_frames: int,
    texture: Image.Image | np.ndarray | None = None,
    out_w: int = 600,
    out_h: int = 600,
    fps: int = 30,
    period_x: float = 200.0,
    period_y: float = 300.0,
    orbit_radius: float = 100.0,
    orbit_speed: float = 1.2,
) -> list[Image.Image]:
    if texture is None:
        texture, _ = make_base_image()
    if isinstance(texture, Image.Image):
        source_w, source_h = texture.size
    else:
        source_h, source_w = texture.shape[:2]
    lens_shapes = default_lens_shapes(source_w, source_h)

    frames: list[Image.Image] = []
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
        )
        t = frame_idx / fps
        theta = orbit_speed * t
        crop_left = round(source_w * 0.5 + orbit_radius * np.cos(theta) - out_w * 0.5)
        crop_top = round(source_h * 0.5 + orbit_radius * np.sin(theta) - out_h * 0.5)
        frame = apply_lens_shapes(
            frame,
            shapes=lens_shapes,
            crop_left=crop_left,
            crop_top=crop_top,
            source_size=(source_w, source_h),
        )
        frames.append(frame)

    return frames


def play_image_list(
    images: list[Image.Image],
    window_name: str = "frames",
    delay_ms: int = 33,
) -> None:
    exit_key = False
    while not exit_key:
        for image in images:
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, frame)
            cv2.waitKey(delay_ms)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                exit_key = True
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    image, image_path = make_base_image()
    print(f"Selected image: {image_path}")

    frames = make_target(n_frames=300, texture=image)
    play_image_list(frames)
