#!/usr/bin/env python3
"""Apply a trusted alpha mask to a same-size RGB/RGBA logo image."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Logo image whose RGB colors to keep")
    parser.add_argument("mask_source", type=Path, help="RGBA image supplying the alpha mask")
    parser.add_argument("output", type=Path, help="Destination transparent PNG")
    parser.add_argument(
        "--min-alpha",
        type=int,
        default=4,
        help="Set lower alpha values to zero to remove faint background haze (default: 4)",
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    mask_source = Image.open(args.mask_source).convert("RGBA")
    if image.size != mask_source.size:
        raise ValueError(
            f"Image sizes differ: color image {image.size}, mask source {mask_source.size}"
        )

    alpha = mask_source.getchannel("A").point(
        lambda value: 0 if value < args.min_alpha else value
    )
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(args.output, format="PNG", optimize=True)


if __name__ == "__main__":
    main()
