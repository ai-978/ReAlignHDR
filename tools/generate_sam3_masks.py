from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    from generate_motion_prompt import generate_motion_prompt
except ImportError:
    from tools.generate_motion_prompt import generate_motion_prompt


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def get_palette(num_cls: int = 256) -> list[int]:
    palette = [0] * (num_cls * 3)
    for j in range(num_cls):
        lab = j
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    palette[0:3] = [0, 0, 0]
    return palette


def parse_prompt(prompt: str) -> list[str]:
    prompts = []
    seen = set()
    for value in prompt.split(","):
        category = value.strip().lower()
        if category and category not in seen:
            seen.add(category)
            prompts.append(category)
    return prompts


def scene_frame_dir(scene_dir: Path, ldr_folder_name: str | None) -> Path:
    return scene_dir / ldr_folder_name if ldr_folder_name else scene_dir


def list_scene_frames(scene_dir: Path, ldr_folder_name: str | None = None, ldr_prefix: str = "") -> list[Path]:
    frame_dir = scene_frame_dir(scene_dir, ldr_folder_name)
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    frames = sorted(
        p for p in frame_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name.startswith(ldr_prefix)
    )
    tif_frames = [p for p in frames if p.suffix.lower() in {".tif", ".tiff"}]
    return tif_frames or frames


def list_scene_dirs(input_root: Path, splits: list[str], scene_dir: Path | None) -> list[Path]:
    if scene_dir is not None:
        return [scene_dir.resolve()]

    scene_dirs: list[Path] = []
    for split in splits:
        split_dir = input_root / split
        if split_dir.is_dir():
            scene_dirs.extend(sorted(p for p in split_dir.iterdir() if p.is_dir()))
        elif input_root.is_dir() and split == "":
            scene_dirs.extend(sorted(p for p in input_root.iterdir() if p.is_dir()))
        else:
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return scene_dirs


def save_binary_mask(binary_mask: np.ndarray, source_path: Path, preview_dir: Path | None, palette: list[int]) -> Path:
    from PIL import Image

    npy_path = source_path.with_suffix(".npy")
    np.save(npy_path, binary_mask.astype(np.uint8))

    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        png_path = preview_dir / f"{source_path.stem}_mask.png"
        preview = Image.fromarray(binary_mask.astype(np.uint8), mode="P")
        preview.putpalette(palette)
        preview.save(png_path)

    return npy_path


def segment_frame(processor, image, target_prompts: list[str], min_mask_area_ratio: float) -> tuple[np.ndarray, int]:
    width, height = image.size
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    instance_count = 0

    if not target_prompts:
        return combined_mask, instance_count

    state = processor.set_image(image)
    for text_prompt in target_prompts:
        output = processor.set_text_prompt(prompt=text_prompt, state=state)
        masks = output.get("masks")
        if masks is None or masks.numel() == 0:
            continue

        masks = masks.detach().cpu().numpy().astype(bool)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError(f"SAM3 masks must have shape [N,H,W] or [N,1,H,W], got {masks.shape}")

        area_ratios = masks.reshape(masks.shape[0], -1).mean(axis=1)
        masks = masks[area_ratios >= min_mask_area_ratio]
        if masks.shape[0] == 0:
            continue

        combined_mask[np.any(masks, axis=0)] = 1
        instance_count += masks.shape[0]

    return combined_mask, instance_count


def load_prompt_for_scene(args, scene_dir: Path) -> tuple[str, str | None]:
    if args.prompt:
        return args.prompt, None

    prompt_file = Path(args.prompt_file) if args.prompt_file else scene_dir / args.prompt_file_name
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip(), None

    if not args.auto_prompt:
        raise ValueError(
            f"No prompt provided for {scene_dir}. Use --prompt, provide {prompt_file}, or enable --auto_prompt."
        )

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {args.api_key_env}")

    prompt, raw_text = generate_motion_prompt(
        scene_dir=scene_dir,
        model=args.prompt_model,
        instruction=args.prompt_instruction,
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        ldr_folder_name=args.ldr_folder_name,
        ldr_prefix=args.ldr_prefix or "",
        temperature=args.temperature,
        top_p=args.top_p,
        max_side=args.prompt_max_side,
    )
    (scene_dir / args.prompt_file_name).write_text(prompt, encoding="utf-8")
    (scene_dir / args.raw_prompt_file_name).write_text(raw_text, encoding="utf-8")
    return prompt, raw_text


def build_sam3_processor(args):
    import torch

    sam3_root = Path(args.sam3_root).resolve()
    if not sam3_root.is_dir():
        raise FileNotFoundError(
            f"SAM3 root not found: {sam3_root}. Pass --sam3_root pointing to the checked-out SAM3 package."
        )
    sys.path.insert(0, str(sam3_root))

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        compile=args.compile,
    )
    return Sam3Processor(
        model,
        resolution=args.resolution,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )


def process_scene(args, scene_dir: Path, processor, palette: list[int]) -> None:
    from PIL import Image

    frame_paths = list_scene_frames(scene_dir, args.ldr_folder_name, args.ldr_prefix or "")
    if args.expected_frames > 0 and len(frame_paths) != args.expected_frames:
        raise ValueError(f"{scene_dir} should contain {args.expected_frames} frames, found {len(frame_paths)}")

    prompt, _ = load_prompt_for_scene(args, scene_dir)
    target_prompts = parse_prompt(prompt)
    print(f"[{scene_dir.name}] prompt: {prompt!r}")

    preview_dir = None
    if args.preview_root:
        preview_dir = Path(args.preview_root) / scene_dir.name

    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            image = image.convert("RGB")
            binary_mask, instance_count = segment_frame(
                processor,
                image,
                target_prompts,
                min_mask_area_ratio=args.min_mask_area_ratio,
            )
        npy_path = save_binary_mask(binary_mask, frame_path, preview_dir, palette)
        print(f"  {frame_path.name}: merged {instance_count} instances -> {npy_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ReAlignHDR per-frame binary masks with SAM3")
    parser.add_argument("--input_root", help="Dataset root")
    parser.add_argument("--splits", nargs="+", help="Dataset splits to process")
    parser.add_argument("--scene_dir", help="Optional single scene directory")
    parser.add_argument("--ldr_folder_name", help="Optional LDR subfolder inside each scene")
    parser.add_argument("--ldr_prefix", help="Only use LDR files whose names start with this prefix")
    parser.add_argument("--expected_frames", type=int, required=True, help="Expected frame count; set 0 to disable")
    parser.add_argument("--preview_root", help="Optional root for PNG mask previews")

    parser.add_argument("--prompt", help="Comma-separated categories, e.g. 'person, car'")
    parser.add_argument("--prompt_file", help="One prompt file used for every scene")
    parser.add_argument("--prompt_file_name", required=True, help="Per-scene prompt filename")
    parser.add_argument("--raw_prompt_file_name", required=True, help="Per-scene raw prompt filename")
    parser.add_argument("--auto_prompt", action="store_true", help="Generate per-scene prompts with an API")
    parser.add_argument("--prompt_model", required=True, help="OpenAI-compatible prompt model")
    parser.add_argument("--prompt_instruction", required=True, help="Motion prompt instruction")
    parser.add_argument("--base_url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key_env", required=True, help="Environment variable containing API key")
    parser.add_argument("--timeout", type=float, required=True, help="Prompt request timeout in seconds")
    parser.add_argument("--temperature", type=float, required=True, help="Prompt sampling temperature")
    parser.add_argument("--top_p", type=float, required=True, help="Prompt nucleus sampling probability")
    parser.add_argument("--prompt_max_side", type=int, required=True, help="Maximum encoded image side length")

    parser.add_argument("--sam3_root", required=True, help="Path to the SAM3 package root")
    parser.add_argument("--confidence_threshold", type=float, required=True, help="SAM3 detection confidence threshold")
    parser.add_argument("--min_mask_area_ratio", type=float, required=True, help="Discard smaller candidate masks")
    parser.add_argument("--resolution", type=int, required=True, help="SAM3 image input resolution")
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True, help="SAM3 device")
    parser.add_argument("--compile", action="store_true", help="Compile SAM3 where supported")
    args = parser.parse_args()

    if args.scene_dir is None and (args.input_root is None or args.splits is None):
        parser.error("Use --scene_dir, or provide both --input_root and --splits.")

    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("--confidence_threshold must be in [0, 1]")
    if not 0 <= args.min_mask_area_ratio <= 1:
        raise ValueError("--min_mask_area_ratio must be in [0, 1]")
    if args.resolution < 1:
        raise ValueError("--resolution must be positive")

    scene_dir = Path(args.scene_dir).resolve() if args.scene_dir else None
    input_root = Path(args.input_root).resolve() if args.input_root else Path(".")
    scene_dirs = list_scene_dirs(input_root, args.splits or [], scene_dir)
    processor = build_sam3_processor(args)
    palette = get_palette(256)

    for current_scene_dir in scene_dirs:
        process_scene(args, current_scene_dir, processor, palette)


if __name__ == "__main__":
    main()
