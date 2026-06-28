from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


MASK_DIR = Path("/Users/Zhuanz/Desktop/sam3/070")
OUTPUT_DIR = MASK_DIR / "visualized_instances"
OVERVIEW_PATH = OUTPUT_DIR / "overview.png"
INSTANCE_COLORS = [
    np.array([255, 255, 0], dtype=np.uint8),    # yellow
    np.array([0, 255, 255], dtype=np.uint8),    # cyan
    np.array([255, 0, 255], dtype=np.uint8),    # magenta
    np.array([255, 64, 64], dtype=np.uint8),    # bright red
    np.array([0, 255, 0], dtype=np.uint8),      # lime
    np.array([255, 128, 0], dtype=np.uint8),    # orange
]


def load_instance_masks(mask_dir: Path) -> list[tuple[Path, np.ndarray]]:
    mask_files = sorted(mask_dir.glob("*.npy"))
    if not mask_files:
        raise FileNotFoundError(f"No .npy files found in {mask_dir}")

    masks = []
    for mask_path in mask_files:
        mask = np.load(mask_path)
        if mask.ndim != 2:
            raise ValueError(f"{mask_path} is not a 2D mask: shape={mask.shape}")
        masks.append((mask_path, mask))
    return masks


def color_for_instance(instance_id: int) -> np.ndarray:
    # Keep background black and use fixed vivid colors for easy distinction.
    return INSTANCE_COLORS[(instance_id - 1) % len(INSTANCE_COLORS)]


def colorize_instance_mask(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    instance_ids = np.unique(mask)

    for instance_id in instance_ids:
        if instance_id == 0:
            continue
        color_mask[mask == instance_id] = color_for_instance(int(instance_id))

    return color_mask


def save_colorized_png(color_mask: np.ndarray, output_path: Path) -> None:
    Image.fromarray(color_mask).save(output_path)


def save_overview(images: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, image) in zip(axes, images):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.suptitle(f"Instance masks in {MASK_DIR.name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    masks = load_instance_masks(MASK_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)

    overview_images = []
    for mask_path, mask in masks:
        color_mask = colorize_instance_mask(mask)
        save_colorized_png(color_mask, OUTPUT_DIR / f"{mask_path.stem}.png")
        overview_images.append((mask_path.stem, color_mask))

    save_overview(overview_images, OVERVIEW_PATH)


if __name__ == "__main__":
    main()
