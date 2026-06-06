from __future__ import annotations

import colorsys

import cv2
import numpy as np
from PIL import Image


def make_aluminium_foil(
    output_size: tuple[int, int] = (800, 800),
    tile_size: int = 95,
    jitter: float = 0.62,
    wrinkle_count: int = 90,
    noise_std: float = 8.0,
    blur_radius: float = 0.45,
    seed: int | None = None,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    width, height = output_size
    margin = tile_size * 2
    work_w = width + margin * 2
    work_h = height + margin * 2

    xs = np.arange(-margin, work_w + margin + tile_size, tile_size)
    ys = np.arange(-margin, work_h + margin + tile_size, tile_size)
    points: list[tuple[float, float]] = []
    for y in ys:
        for x in xs:
            points.append(
                (
                    float(x + rng.uniform(-tile_size, tile_size) * jitter),
                    float(y + rng.uniform(-tile_size, tile_size) * jitter),
                )
            )

    rect = (0, 0, work_w, work_h)
    subdiv = cv2.Subdiv2D(rect)
    for x, y in points:
        if 0 <= x < work_w and 0 <= y < work_h:
            subdiv.insert((x, y))

    yy, xx = np.mgrid[0:work_h, 0:work_w].astype(np.float32)
    foil = np.full((work_h, work_w), 142.0, dtype=np.float32)
    creases = np.zeros_like(foil)
    triangles = subdiv.getTriangleList()

    for triangle in triangles:
        pts = triangle.reshape(3, 2)
        if (
            np.any(pts[:, 0] < 0)
            or np.any(pts[:, 0] >= work_w)
            or np.any(pts[:, 1] < 0)
            or np.any(pts[:, 1] >= work_h)
        ):
            continue

        polygon = np.round(pts).astype(np.int32)
        x0 = max(0, int(np.floor(pts[:, 0].min())) - 2)
        y0 = max(0, int(np.floor(pts[:, 1].min())) - 2)
        x1 = min(work_w, int(np.ceil(pts[:, 0].max())) + 3)
        y1 = min(work_h, int(np.ceil(pts[:, 1].max())) + 3)
        if x1 <= x0 or y1 <= y0:
            continue

        local_polygon = polygon - np.array([x0, y0], dtype=np.int32)
        local_h = y1 - y0
        local_w = x1 - x0
        mask = np.zeros((local_h, local_w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, local_polygon, 255, lineType=cv2.LINE_AA)

        angle = rng.uniform(0.0, 2.0 * np.pi)
        direction_x = np.cos(angle)
        direction_y = np.sin(angle)
        center = pts.mean(axis=0)
        local_xx = xx[y0:y1, x0:x1]
        local_yy = yy[y0:y1, x0:x1]
        projection = (local_xx - center[0]) * direction_x + (local_yy - center[1]) * direction_y
        projection /= max(tile_size * rng.uniform(0.6, 1.7), 1.0)

        base = rng.triangular(82.0, 145.0, 226.0)
        contrast = rng.uniform(38.0, 108.0)
        sheen = base + contrast * np.tanh(projection)
        sheen = sheen * rng.uniform(0.82, 1.14) + rng.normal(-4.0, 14.0)
        if rng.random() < 0.35:
            sheen = 255.0 - sheen * rng.uniform(0.75, 1.0)

        local_foil = foil[y0:y1, x0:x1]
        foil[y0:y1, x0:x1] = np.where(mask > 0, sheen, local_foil)

        edge_color = float(rng.choice([rng.uniform(8.0, 78.0), rng.uniform(195.0, 255.0)]))
        cv2.polylines(
            creases,
            [polygon],
            isClosed=True,
            color=edge_color,
            thickness=int(rng.integers(1, 4)),
            lineType=cv2.LINE_AA,
        )

    for _ in range(wrinkle_count):
        start = rng.uniform([0, 0], [work_w, work_h])
        angle = rng.uniform(0.0, 2.0 * np.pi)
        length = rng.uniform(tile_size * 0.35, tile_size * 1.6)
        end = start + np.array([np.cos(angle), np.sin(angle)]) * length
        color = float(rng.choice([rng.uniform(12.0, 105.0), rng.uniform(190.0, 255.0)]))
        cv2.line(
            creases,
            tuple(np.round(start).astype(int)),
            tuple(np.round(end).astype(int)),
            color=color,
            thickness=int(rng.integers(1, 3)),
            lineType=cv2.LINE_AA,
        )

    creases = cv2.GaussianBlur(creases, (0, 0), sigmaX=0.55)
    foil = foil * 0.76 + creases * 0.28 - 7.0
    foil += rng.normal(0.0, noise_std, size=foil.shape)
    foil = cv2.GaussianBlur(foil, (0, 0), sigmaX=blur_radius)
    foil = np.clip(foil, 0, 255).astype(np.uint8)

    rgb = np.stack(
        [
            np.clip(foil * 1.01 + 1, 0, 255),
            np.clip(foil * 1.00, 0, 255),
            np.clip(foil * 0.97 - 1, 0, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    cropped = rgb[margin : margin + height, margin : margin + width]
    return Image.fromarray(cropped, mode="RGB"), "<synthesized_aluminium_foil>"


def make_nothing(
    output_size: tuple[int, int] = (800, 800),
    seed: int | None = None,
) -> tuple[Image.Image, str]:
    rng = np.random.default_rng(seed)
    hue = float(rng.uniform(0.0, 1.0))
    saturation = float(rng.uniform(0.2, 0.8))
    value = float(rng.uniform(0.2, 0.8))
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    color = tuple(round(channel * 255.0) for channel in (red, green, blue))
    return Image.new("RGB", output_size, color), "<synthesized_nothing>"


def make_tiling(
    output_size: tuple[int, int] = (800, 800),
    tile_count: int = 120,
    seed: int | None = None,
) -> tuple[Image.Image, str]:
    rng = np.random.default_rng(seed)
    width, height = output_size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    grid_cols = int(np.ceil(np.sqrt(tile_count * width / max(height, 1))))
    grid_rows = int(np.ceil(tile_count / max(grid_cols, 1)))
    cell_w = width / grid_cols
    cell_h = height / grid_rows
    sites: list[tuple[float, float]] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            if len(sites) >= tile_count:
                break
            sites.append(
                (
                    (col + rng.uniform(0.16, 0.84)) * cell_w,
                    (row + rng.uniform(0.16, 0.84)) * cell_h,
                )
            )

    sites_array = np.asarray(sites, dtype=np.float32)
    tile_angles = rng.uniform(0.0, 2.0 * np.pi, size=len(sites_array)).astype(np.float32)
    stretch_x = rng.uniform(0.62, 1.65, size=len(sites_array)).astype(np.float32)
    stretch_y = rng.uniform(0.62, 1.65, size=len(sites_array)).astype(np.float32)
    weights = rng.normal(0.0, 420.0, size=len(sites_array)).astype(np.float32)
    warp_x = (
        18.0 * np.sin(yy / rng.uniform(44.0, 86.0) + rng.uniform(0.0, 2.0 * np.pi))
        + 12.0 * np.sin((xx + yy) / rng.uniform(78.0, 140.0))
    )
    warp_y = (
        18.0 * np.cos(xx / rng.uniform(44.0, 86.0) + rng.uniform(0.0, 2.0 * np.pi))
        + 12.0 * np.sin((xx - yy) / rng.uniform(78.0, 140.0))
    )
    sample_x = xx + warp_x
    sample_y = yy + warp_y

    labels = np.empty((height, width), dtype=np.int32)
    chunk_rows = 64
    for y0 in range(0, height, chunk_rows):
        y1 = min(height, y0 + chunk_rows)
        dx = sample_x[y0:y1, :, None] - sites_array[:, 0]
        dy = sample_y[y0:y1, :, None] - sites_array[:, 1]
        cos_a = np.cos(tile_angles)
        sin_a = np.sin(tile_angles)
        local_x = dx * cos_a + dy * sin_a
        local_y = -dx * sin_a + dy * cos_a
        distances = (local_x / stretch_x) ** 2 + (local_y / stretch_y) ** 2 - weights
        labels[y0:y1] = np.argmin(distances, axis=2)

    base_hue = float(rng.uniform(0.0, 1.0))
    image = np.zeros((height, width, 3), dtype=np.float32)
    for tile_idx in range(len(sites)):
        mask = labels == tile_idx
        if not np.any(mask):
            continue

        hue = base_hue
        saturation = float(rng.uniform(0.2, 0.8))
        value = float(rng.uniform(0.2, 0.8))
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        color = np.array([red, green, blue], dtype=np.float32) * 255.0

        site_x, site_y = sites_array[tile_idx]
        angle = rng.uniform(0.0, 2.0 * np.pi)
        direction_x = np.cos(angle)
        direction_y = np.sin(angle)
        projection = ((xx - site_x) * direction_x + (yy - site_y) * direction_y)
        projection /= rng.uniform(55.0, 140.0)
        grain = 1.0 + 0.12 * np.tanh(projection) + rng.normal(0.0, 0.018, size=(height, width))

        image[mask] = np.clip(color * grain[mask, None], 0.0, 255.0)

    edge = np.zeros((height, width), dtype=np.uint8)
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge = cv2.dilate(edge, np.ones((2, 2), dtype=np.uint8), iterations=1)
    edge_soft = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), sigmaX=0.65)
    edge_soft = np.clip(edge_soft, 0.0, 1.0)
    image *= 1.0 - 0.24 * edge_soft[..., None]
    image += rng.normal(0.0, 2.5, size=image.shape)

    image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB"), "<synthesized_tiling>"


def make_ground(
    output_size: tuple[int, int] = (800, 800),
    grain_count: int | None = None,
    grain_size: tuple[float, float] = (58.0, 128.0),
    seed: int | None = None,
) -> tuple[Image.Image, str]:
    rng = np.random.default_rng(seed)
    width, height = output_size
    margin = int(max(grain_size) * 2)
    work_w = width + margin * 2
    work_h = height + margin * 2
    grain_count = grain_count or int(work_w * work_h / 620.0)

    base_color = np.array([132.0, 91.0, 37.0], dtype=np.float32)
    canvas = np.empty((work_h, work_w, 3), dtype=np.float32)
    canvas[:] = base_color
    shadow = np.zeros((work_h, work_w), dtype=np.float32)

    for _ in range(grain_count):
        center_x = rng.uniform(-margin * 0.3, work_w + margin * 0.3)
        center_y = rng.uniform(-margin * 0.3, work_h + margin * 0.3)
        length = rng.uniform(grain_size[0], grain_size[1])
        width_ratio = rng.uniform(0.42, 0.68)
        half_l = length * 0.5
        half_w = length * width_ratio * 0.5
        angle = rng.normal(-0.65, 0.75)

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        pad = int(length * 0.82)
        x0 = max(0, int(center_x - pad))
        y0 = max(0, int(center_y - pad))
        x1 = min(work_w, int(center_x + pad) + 1)
        y1 = min(work_h, int(center_y + pad) + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx = xx - center_x
        dy = yy - center_y
        local_x = dx * cos_a + dy * sin_a
        local_y = -dx * sin_a + dy * cos_a
        radius = (local_x / half_l) ** 2 + (local_y / half_w) ** 2
        mask = np.clip((1.0 - radius) * 3.0, 0.0, 1.0)
        if mask.max() <= 0.0:
            continue

        ridge = np.clip(1.0 - np.abs(local_y) / max(half_w, 1.0), 0.0, 1.0)
        end_taper = np.clip(1.0 - np.abs(local_x) / max(half_l, 1.0), 0.0, 1.0)
        fiber = (
            0.55 * np.sin(local_y * rng.uniform(0.38, 0.78) + rng.uniform(0.0, 2.0 * np.pi))
            + 0.35 * np.sin(local_y * rng.uniform(0.95, 1.55) + local_x * 0.08)
            + rng.normal(0.0, 0.18, size=mask.shape)
        )
        light = 0.78 + 0.30 * ridge + 0.16 * end_taper + 0.09 * fiber
        shade = 1.0 - 0.34 * np.clip((local_y / max(half_w, 1.0)) + 0.15, 0.0, 1.0)

        grain_color = np.array(
            [
                rng.uniform(172.0, 210.0),
                rng.uniform(118.0, 146.0),
                rng.uniform(50.0, 72.0),
            ],
            dtype=np.float32,
        )
        grain = grain_color[None, None, :] * light[..., None] * shade[..., None]
        grain += np.array([18.0, 10.0, 1.0], dtype=np.float32) * ridge[..., None]

        alpha = (mask * rng.uniform(0.88, 1.0))[..., None]
        local_canvas = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = local_canvas * (1.0 - alpha) + grain * alpha

        contact = mask.astype(np.float32)
        contact = np.roll(contact, shift=(int(length * 0.10), int(-length * 0.06)), axis=(0, 1))
        shadow[y0:y1, x0:x1] += contact * rng.uniform(16.0, 36.0)

    canvas -= shadow[..., None] * np.array([0.82, 0.72, 0.55], dtype=np.float32)

    noise = rng.normal(0.0, 4.5, size=canvas.shape)
    canvas += noise
    luminance = (
        canvas[..., 0] * 0.299
        + canvas[..., 1] * 0.587
        + canvas[..., 2] * 0.114
    )
    warm_canvas = np.stack(
        [
            luminance * 1.15 + 16.0,
            luminance * 0.84 + 16.0,
            luminance * 0.42 + 10.0,
        ],
        axis=-1,
    )
    canvas = canvas * 0.28 + warm_canvas * 0.72
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    cropped = canvas[margin : margin + height, margin : margin + width]
    return Image.fromarray(cropped, mode="RGB"), "<synthesized_ground>"
