#!/usr/bin/env python3
"""Create a printable, back-lit relief (lithophane) from an image.

Install dependencies first:
    pip install numpy numpy-stl Pillow

Example:
    python image_to_lithophane.py --input portrait.png --output portrait.stl --width 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from stl import mesh


DEFAULT_FRAME_MM = 3.0


def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Blur a 2-D height/image array using separable Gaussian convolution."""
    radius = max(1, int(np.ceil(sigma * 3)))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(axis * axis) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    # Reflecting at the boundary avoids artificial dark/bright edges.
    padded_x = np.pad(image, ((0, 0), (radius, radius)), mode="reflect")
    horizontal = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), 1, padded_x
    )
    padded_y = np.pad(horizontal, ((radius, radius), (0, 0)), mode="reflect")
    return np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="valid"), 0, padded_y)


def load_grayscale(path: Path, max_resolution: int) -> np.ndarray:
    """Load, downsample, blur, and convert an image to weighted grayscale."""
    if max_resolution < 2:
        raise ValueError("--resolution must be at least 2")

    with Image.open(path) as source:
        # Work in RGB so palette, grayscale, and RGBA inputs behave consistently.
        rgb_image = source.convert("RGB")
        source_width, source_height = rgb_image.size
        scale = min(1.0, max_resolution / max(source_width, source_height))
        target_size = (
            max(2, round(source_width * scale)),
            max(2, round(source_height * scale)),
        )
        if target_size != rgb_image.size:
            rgb_image = rgb_image.resize(target_size, Image.Resampling.LANCZOS)
        rgb = np.asarray(rgb_image, dtype=np.float64)

    # ITU-R BT.601 luminance weights: Gray = .299R + .587G + .114B.
    grayscale = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return gaussian_blur(grayscale, sigma=1.0)


def add_flat_frame(
    heights: np.ndarray, width_mm: float, height_mm: float, frame_mm: float
) -> Tuple[np.ndarray, float, float]:
    """Surround the relief with a flat, printable frame at its minimum height."""
    image_rows, image_cols = heights.shape
    pixel_mm = min(width_mm / (image_cols - 1), height_mm / (image_rows - 1))
    frame_cells = max(1, round(frame_mm / pixel_mm))
    framed = np.pad(heights, frame_cells, mode="constant", constant_values=heights.min())
    return framed, width_mm + 2 * frame_cells * pixel_mm, height_mm + 2 * frame_cells * pixel_mm


def append_triangle(triangles: list[np.ndarray], a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    """Append one triangle with vertices ordered for outward-facing normals."""
    triangles.append(np.array((a, b, c), dtype=np.float32))


def make_closed_mesh(heights: np.ndarray, width_mm: float, height_mm: float) -> np.ndarray:
    """Build top, bottom, and four walls to create a closed solid triangle mesh."""
    rows, cols = heights.shape
    x_values = np.linspace(0.0, width_mm, cols)
    y_values = np.linspace(0.0, height_mm, rows)
    triangles: list[np.ndarray] = []

    def top(row: int, col: int) -> np.ndarray:
        return np.array((x_values[col], y_values[row], heights[row, col]))

    def bottom(row: int, col: int) -> np.ndarray:
        return np.array((x_values[col], y_values[row], 0.0))

    for row in range(rows - 1):
        for col in range(cols - 1):
            tl, tr = top(row, col), top(row, col + 1)
            bl, br = top(row + 1, col), top(row + 1, col + 1)
            append_triangle(triangles, tl, tr, br)
            append_triangle(triangles, tl, br, bl)

            # Reverse the winding on the underside so its normal faces downward.
            tl, tr = bottom(row, col), bottom(row, col + 1)
            bl, br = bottom(row + 1, col), bottom(row + 1, col + 1)
            append_triangle(triangles, tl, br, tr)
            append_triangle(triangles, tl, bl, br)

    # Join the top and bottom around all four outer edges.
    edges = [
        [(0, col) for col in range(cols)],
        [(row, cols - 1) for row in range(rows)],
        [(rows - 1, col) for col in range(cols - 1, -1, -1)],
        [(row, 0) for row in range(rows - 1, -1, -1)],
    ]
    for edge in edges:
        for (r1, c1), (r2, c2) in zip(edge, edge[1:]):
            append_triangle(triangles, bottom(r1, c1), top(r2, c2), top(r1, c1))
            append_triangle(triangles, bottom(r1, c1), bottom(r2, c2), top(r2, c2))

    return np.asarray(triangles, dtype=np.float32)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a framed lithophane STL from an image.")
    parser.add_argument("--input", required=True, type=Path, help="Input image path")
    parser.add_argument("--output", required=True, type=Path, help="Output STL path")
    parser.add_argument("--width", type=float, default=100.0, help="Print width in mm (default: 100)")
    parser.add_argument("--height", type=float, help="Print height in mm (default: preserve image ratio)")
    parser.add_argument("--min-thickness", type=float, default=0.6, help="Thin/light thickness in mm")
    parser.add_argument("--max-thickness", type=float, default=2.5, help="Thick/dark thickness in mm")
    parser.add_argument("--resolution", type=int, default=150, help="Maximum image dimension in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.width <= 0 or (args.height is not None and args.height <= 0):
        raise ValueError("--width and --height must be positive")
    if args.min_thickness <= 0 or args.max_thickness < args.min_thickness:
        raise ValueError("Thicknesses must be positive and max must be at least min")

    grayscale = load_grayscale(args.input, args.resolution)
    rows, cols = grayscale.shape
    print(f"Using {cols} x {rows} image samples")
    model_height = args.height if args.height is not None else args.width * (rows - 1) / (cols - 1)

    # Inversion makes dark image areas thick, preserving the back-lit image appearance.
    inverted = 255.0 - grayscale
    heights = args.min_thickness + (inverted / 255.0) * (args.max_thickness - args.min_thickness)
    heights = gaussian_blur(heights, sigma=0.65)  # Light smoothing reduces visible steps.
    heights, final_width, final_height = add_flat_frame(
        heights, args.width, model_height, DEFAULT_FRAME_MM
    )

    triangles = make_closed_mesh(heights, final_width, final_height)
    stl_mesh = mesh.Mesh(np.zeros(len(triangles), dtype=mesh.Mesh.dtype))
    stl_mesh.vectors[:] = triangles
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stl_mesh.save(str(args.output))
    print(f"Wrote {args.output} ({len(triangles):,} triangles, {final_width:.2f} x {final_height:.2f} mm)")


if __name__ == "__main__":
    main()
