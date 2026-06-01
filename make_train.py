from pathlib import Path
import random
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


SCALE_PATTERN = re.compile(r"scale_(\d+)_")
LENS_SHAPE_KINDS = ("circle", "rectangle", "triangle", "star")
LensShape = dict[str, float | int | str]


def make_base_image(
    data_dir: str | Path = "KTH_TIPS",
    path: str | Path | None = None,
    max_scale: int | None = None,
    output_size: tuple[int, int] = (800, 800),
    blur_radius: float = 0.4,
    noise_probability: float = 0.3,
    noise_std: float = 2.0,
) -> tuple[Image.Image, Path]:
    data_dir = Path(data_dir)

    if path is None:
        candidates: list[Path] = []
        for candidate in data_dir.rglob("*.png"):
            match = SCALE_PATTERN.search(candidate.name)
            if (
                match
                and (max_scale is None or int(match.group(1)) <= max_scale)
            ):
                candidates.append(candidate)

        if not candidates:
            raise FileNotFoundError(f"No matching PNG images found in {data_dir}")

        image_path = random.choice(candidates)
    else:
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(output_size, Image.Resampling.BICUBIC)
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if random.random() < noise_probability:
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
    min_gap: float = 0.08,
    max_attempts: int = 1000,
    rng: random.Random | None = None,
) -> list[tuple[float, float, float]]:
    rng = rng or random
    size = min(source_w, source_h)
    circles: list[tuple[float, float, float]] = []
    attempts = 0
    while len(circles) < count and attempts < max_attempts:
        attempts += 1
        radius = rng.uniform(0.080, 0.092) * size
        center_x = rng.uniform(radius, source_w - radius)
        center_y = rng.uniform(radius, source_h - radius)
        if any(
            np.hypot(center_x - x, center_y - y) < radius + other_radius + min_gap * size
            for x, y, other_radius in circles
        ):
            continue
        circles.append((center_x, center_y, radius))
    return circles


def default_lens_shapes(
    source_w: int,
    source_h: int,
    count: int = 8,
    rng: random.Random | None = None,
) -> list[LensShape]:
    rng = rng or random
    shapes: list[LensShape] = []
    kind = rng.choice(LENS_SHAPE_KINDS)
    for x, y, radius in default_lens_circles(source_w, source_h, count=count, rng=rng):
        rotation = rng.uniform(0.0, 2.0 * np.pi)
        shape: LensShape = {"kind": kind, "x": x, "y": y, "radius": radius}
        if kind == "rectangle":
            shape["width"] = radius * rng.uniform(1.6, 2.5)
            shape["height"] = radius * rng.uniform(1.6, 2.5)
            shape["rotation"] = rotation
        elif kind == "triangle":
            shape["rotation"] = rotation
        elif kind == "star":
            shape["inner_radius"] = radius * rng.uniform(0.42, 0.55)
            shape["points"] = 5
            shape["rotation"] = rotation
        shapes.append(shape)
    return shapes


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
        rotation = float(shape.get("rotation", 0.0))
        half_width = width * 0.5
        half_height = height * 0.5
        corners = np.array(
            [
                [-half_width, -half_height],
                [half_width, -half_height],
                [half_width, half_height],
                [-half_width, half_height],
            ],
            dtype=np.float32,
        )
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
        points = corners @ rot.T + np.array([center_x, center_y], dtype=np.float32)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255, lineType=cv2.LINE_AA)
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


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_lens_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    rim_width: float = 12.0,
    outer_shadow_width: float = 18.0,
    magnification: float = 1.07,
    rim_refraction: float = 12.0,
    outer_line_darkness: float = 0.34,
    inner_line_brightness: float = 0.24,
    glass_rim_brightness: float = 0.025,
    outer_line_offset: float = 1.6,
    inner_line_offset: float = 2.2,
    outer_line_softness: float = 0.85,
    inner_line_softness: float = 0.95,
) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    img = frame.astype(np.float32) / 255.0
    bin_mask = (mask > 127).astype(np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    if cv2.countNonZero(bin_mask) == 0:
        return frame

    out = img.copy()
    h, w = bin_mask.shape

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask,
        connectivity=8,
    )

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        pad = int(max(rim_width * 2, outer_shadow_width * 2, rim_refraction * 2, 8))

        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad)
        y1 = min(h, y + hh + pad)

        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        if comp.sum() == 0:
            continue

        roi_out = out[y0:y1, x0:x1]
        inside_dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        outside_dist = cv2.distanceTransform(1 - comp, cv2.DIST_L2, 5)
        max_inside = float(inside_dist.max())
        if max_inside < 1e-6:
            continue

        alpha = cv2.GaussianBlur(comp.astype(np.float32), (0, 0), 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        cx, cy = centroids[i].astype(np.float32)
        dx = xx - cx
        dy = yy - cy

        gx = cv2.Sobel(inside_dist, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(inside_dist, cv2.CV_32F, 0, 1, ksize=3)
        gnorm = np.sqrt(gx * gx + gy * gy) + 1e-6
        nx = gx / gnorm
        ny = gy / gnorm

        t = np.clip(inside_dist / max_inside, 0.0, 1.0)
        local_mag = 1.0 + (magnification - 1.0) * (t**0.75)
        src_x = cx + dx / local_mag
        src_y = cy + dy / local_mag

        rim_zone = 1.0 - np.clip(inside_dist / max(rim_width, 1e-6), 0.0, 1.0)
        rim_zone = _smoothstep(0.0, 1.0, rim_zone)
        src_x += nx * rim_refraction * rim_zone
        src_y += ny * rim_refraction * rim_zone

        refracted = cv2.remap(
            img,
            src_x.astype(np.float32),
            src_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        roi_out[:] = roi_out * (1.0 - alpha[..., None]) + refracted * alpha[..., None]
        roi_out[:] += (0.045 * t * alpha)[..., None]

        shadow = cv2.GaussianBlur(
            comp.astype(np.float32),
            (0, 0),
            sigmaX=max(1.0, outer_shadow_width * 0.45),
            sigmaY=max(1.0, outer_shadow_width * 0.45),
        )
        shift_x = int(0.35 * outer_shadow_width)
        shift_y = int(0.35 * outer_shadow_width)
        shadow = np.roll(shadow, shift=(shift_y, shift_x), axis=(0, 1))
        if shift_y > 0:
            shadow[:shift_y, :] = 0
        if shift_x > 0:
            shadow[:, :shift_x] = 0

        outer_band = np.clip(1.0 - outside_dist / max(outer_shadow_width, 1e-6), 0.0, 1.0)
        shadow *= outer_band
        shadow *= 1.0 - alpha
        roi_out[:] *= 1.0 - 0.22 * shadow[..., None]

        sdf = inside_dist - outside_dist
        outer_dark_line = np.exp(
            -((sdf + outer_line_offset) ** 2)
            / (2.0 * outer_line_softness * outer_line_softness + 1e-6)
        )
        inner_bright_line = np.exp(
            -((sdf - inner_line_offset) ** 2)
            / (2.0 * inner_line_softness * inner_line_softness + 1e-6)
        )
        inner_glass_rim = np.clip(1.0 - inside_dist / max(rim_width, 1e-6), 0.0, 1.0)
        inner_glass_rim = _smoothstep(0.0, 1.0, inner_glass_rim) * alpha
        outer_ring_alpha = np.clip(outer_line_darkness * outer_dark_line * (1.0 - alpha), 0.0, 1.0)
        inner_ring_alpha = np.clip(inner_line_brightness * inner_bright_line * alpha, 0.0, 1.0)
        glass_rim_alpha = np.clip(glass_rim_brightness * inner_glass_rim, 0.0, 1.0)

        dark_color = np.array([0.035, 0.035, 0.035], dtype=np.float32)
        bright_color = np.array([0.92, 0.92, 0.92], dtype=np.float32)
        glass_color = np.array([0.72, 0.76, 0.78], dtype=np.float32)

        roi_out[:] = roi_out * (1.0 - outer_ring_alpha[..., None]) + dark_color * outer_ring_alpha[..., None]
        roi_out[:] = roi_out * (1.0 - inner_ring_alpha[..., None]) + bright_color * inner_ring_alpha[..., None]
        roi_out[:] = roi_out * (1.0 - glass_rim_alpha[..., None]) + glass_color * glass_rim_alpha[..., None]

        rim_band = np.clip(1.0 - inside_dist / max(rim_width, 1e-6), 0.0, 1.0)
        rim_band = _smoothstep(0.0, 1.0, rim_band) * alpha
        lx, ly = -0.7071, -0.7071
        ndotl = -(nx * lx + ny * ly)

        highlight = np.clip(ndotl, 0.0, 1.0) * rim_band
        darkside = np.clip(-ndotl, 0.0, 1.0) * rim_band

        roi_out[:] += 0.18 * highlight[..., None]
        roi_out[:] -= 0.14 * darkside[..., None]

        if rim_width > 1.0:
            target = 0.65 * rim_width
            sigma = max(1.0, 0.35 * rim_width)
            spec_band = np.exp(-((inside_dist - target) ** 2) / (2.0 * sigma * sigma + 1e-6))
            spec = spec_band * (np.clip(ndotl, 0.0, 1.0) ** 2) * alpha
            roi_out[:] += 0.08 * spec[..., None]

        out[y0:y1, x0:x1] = np.clip(roi_out, 0.0, 1.0)

    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def apply_lens_circle(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    outer_line_darkness: float = 0.34,
    inner_line_brightness: float = 0.24,
    glass_rim_brightness: float = 0.025,
    outer_line_offset: float = 1.6,
    inner_line_offset: float = 2.2,
    outer_line_softness: float = 0.85,
    inner_line_softness: float = 0.95,
) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    draw_lens_shape(mask, {"kind": "circle", "x": center_x, "y": center_y, "radius": radius})
    return apply_lens_mask(
        frame,
        mask,
        outer_line_darkness=outer_line_darkness,
        inner_line_brightness=inner_line_brightness,
        glass_rim_brightness=glass_rim_brightness,
        outer_line_offset=outer_line_offset,
        inner_line_offset=inner_line_offset,
        outer_line_softness=outer_line_softness,
        inner_line_softness=inner_line_softness,
    )


def apply_lens_shapes(
    image: Image.Image,
    shapes: list[LensShape] | None = None,
    crop_left: int = 0,
    crop_top: int = 0,
    source_size: tuple[int, int] | None = None,
    outer_line_darkness: float = 0.34,
    inner_line_brightness: float = 0.24,
    glass_rim_brightness: float = 0.025,
    outer_line_offset: float = 1.6,
    inner_line_offset: float = 2.2,
    outer_line_softness: float = 0.85,
    inner_line_softness: float = 0.95,
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
    padded_frame = apply_lens_mask(
        padded_frame,
        mask,
        outer_line_darkness=outer_line_darkness,
        inner_line_brightness=inner_line_brightness,
        glass_rim_brightness=glass_rim_brightness,
        outer_line_offset=outer_line_offset,
        inner_line_offset=inner_line_offset,
        outer_line_softness=outer_line_softness,
        inner_line_softness=inner_line_softness,
    )
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
    outer_line_darkness: float = 0.34,
    inner_line_brightness: float = 0.24,
    glass_rim_brightness: float = 0.025,
    outer_line_offset: float = 1.6,
    inner_line_offset: float = 2.2,
    outer_line_softness: float = 0.85,
    inner_line_softness: float = 0.95,
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
        outer_line_darkness=outer_line_darkness,
        inner_line_brightness=inner_line_brightness,
        glass_rim_brightness=glass_rim_brightness,
        outer_line_offset=outer_line_offset,
        inner_line_offset=inner_line_offset,
        outer_line_softness=outer_line_softness,
        inner_line_softness=inner_line_softness,
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
    outer_line_darkness: float = 0.34,
    inner_line_brightness: float = 0.24,
    glass_rim_brightness: float = 0.025,
    outer_line_offset: float = 1.6,
    inner_line_offset: float = 2.2,
    outer_line_softness: float = 0.85,
    inner_line_softness: float = 0.95,
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
            outer_line_darkness=outer_line_darkness,
            inner_line_brightness=inner_line_brightness,
            glass_rim_brightness=glass_rim_brightness,
            outer_line_offset=outer_line_offset,
            inner_line_offset=inner_line_offset,
            outer_line_softness=outer_line_softness,
            inner_line_softness=inner_line_softness,
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
