from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np


IMAGE_EXTS = {".tif", ".tiff"}


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def list_scenes(input_root: Path) -> list[Path]:
    scenes = sorted(p for p in input_root.iterdir() if p.is_dir())
    if scenes:
        return scenes
    raise FileNotFoundError(f"No scene directories found in {input_root}")


def list_frames(scene_dir: Path, ldr_folder_name: str | None, ldr_prefix: str) -> list[Path]:
    frame_dir = scene_dir / ldr_folder_name if ldr_folder_name else scene_dir
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    return sorted(
        p for p in frame_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and p.name.startswith(ldr_prefix)
    )


def copy_scene_files(args, scene_dir: Path, out_scene_dir: Path) -> None:
    source_ldr_folder = args.source_ldr_folder_name or args.ldr_folder_name
    frame_paths = list_frames(scene_dir, source_ldr_folder, args.ldr_prefix or "")
    if args.expected_frames > 0 and len(frame_paths) != args.expected_frames:
        raise ValueError(f"{scene_dir} should contain {args.expected_frames} frames, found {len(frame_paths)}")

    out_frame_dir = out_scene_dir / args.ldr_folder_name if args.ldr_folder_name else out_scene_dir
    for frame_path in frame_paths:
        link_or_copy(frame_path, out_frame_dir / frame_path.name, args.mode, args.overwrite)

        mask_path = frame_path.with_suffix(".npy")
        out_mask_path = out_frame_dir / f"{frame_path.stem}.npy"
        if args.copy_existing_masks and mask_path.exists():
            link_or_copy(mask_path, out_mask_path, args.mode, args.overwrite)
        elif args.create_zero_masks and (args.overwrite or not out_mask_path.exists()):
            from PIL import Image

            with Image.open(frame_path) as image:
                width, height = image.size
            np.save(out_mask_path, np.zeros((height, width), dtype=np.uint8))

    exposure_path = scene_dir / args.exposure_file_name
    if exposure_path.exists():
        link_or_copy(exposure_path, out_scene_dir / args.exposure_file_name, args.mode, args.overwrite)
    elif args.require_exposure:
        raise FileNotFoundError(f"Missing exposure file: {exposure_path}")

    label_path = scene_dir / args.label_file_name
    if label_path.exists():
        link_or_copy(label_path, out_scene_dir / args.label_file_name, args.mode, args.overwrite)
    elif args.require_label:
        raise FileNotFoundError(f"Missing HDR label: {label_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize raw HDR scenes into the layout expected by train.py/test.py"
    )
    parser.add_argument("--input_root", required=True, help="Root containing raw scene directories")
    parser.add_argument("--output_root", required=True, help="Target dataset root")
    parser.add_argument("--split", required=True, help="Output split name")
    parser.add_argument("--scene_names", nargs="*", help="Optional subset of scene directory names")
    parser.add_argument("--source_ldr_folder_name", help="Optional source LDR subfolder")
    parser.add_argument("--ldr_folder_name", help="Optional target LDR subfolder")
    parser.add_argument("--ldr_prefix", help="Only use LDR files whose names start with this prefix")
    parser.add_argument("--exposure_file_name", required=True, help="Exposure filename")
    parser.add_argument("--label_file_name", required=True, help="HDR label filename")
    parser.add_argument("--expected_frames", type=int, required=True, help="Expected LDR frame count; set 0 to disable")
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink", "hardlink"),
        required=True,
        help="How to place files in the output dataset",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing target files")
    parser.add_argument("--copy_existing_masks", action="store_true", help="Copy per-frame .npy masks if present")
    parser.add_argument("--create_zero_masks", action="store_true", help="Create all-zero masks as placeholders")
    parser.add_argument("--require_exposure", action="store_true", help="Fail when exposure file is missing")
    parser.add_argument("--allow_missing_exposure", dest="require_exposure", action="store_false")
    parser.add_argument("--require_label", action="store_true", help="Fail when HDR label is missing")
    parser.add_argument("--allow_missing_label", dest="require_label", action="store_false")
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_split_dir = Path(args.output_root).resolve() / args.split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    scenes = list_scenes(input_root)
    if args.scene_names:
        wanted = set(args.scene_names)
        scenes = [scene for scene in scenes if scene.name in wanted]
        missing = sorted(wanted - {scene.name for scene in scenes})
        if missing:
            raise FileNotFoundError(f"Requested scenes not found: {missing}")

    for scene_dir in scenes:
        out_scene_dir = output_split_dir / scene_dir.name
        out_scene_dir.mkdir(parents=True, exist_ok=True)
        copy_scene_files(args, scene_dir, out_scene_dir)
        print(f"{scene_dir} -> {out_scene_dir}")


if __name__ == "__main__":
    main()
