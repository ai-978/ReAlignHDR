from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def has_scene_frames(scene_dir: Path, ldr_folder_name: str | None, ldr_prefix: str | None) -> bool:
    frame_dir = scene_dir / ldr_folder_name if ldr_folder_name else scene_dir
    if not frame_dir.is_dir():
        return False
    prefix = ldr_prefix or ""
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.name.startswith(prefix)
        for path in frame_dir.iterdir()
    )


def list_scene_dirs(
    input_root: Path,
    splits: list[str] | None,
    scene_names: set[str] | None,
    ldr_folder_name: str | None,
    ldr_prefix: str | None,
) -> list[Path]:
    def discover(root: Path) -> list[Path]:
        if has_scene_frames(root, ldr_folder_name, ldr_prefix):
            return [root]

        discovered = []
        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            discovered.extend(discover(child))
        return discovered

    if splits:
        candidates = []
        for split in splits:
            split_dir = input_root / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
            candidates.extend(discover(split_dir))
    else:
        candidates = discover(input_root)

    scene_dirs = []
    for scene_dir in candidates:
        if scene_names is not None and scene_dir.name not in scene_names:
            continue
        if has_scene_frames(scene_dir, ldr_folder_name, ldr_prefix):
            scene_dirs.append(scene_dir)

    if not scene_dirs:
        raise ValueError(f"No scene directories found under {input_root}")
    return scene_dirs


def append_optional(command: list[str], name: str, value: str | None) -> None:
    if value is not None:
        command.extend([name, value])


def run_motion_prompt(args, script_dir: Path, scene_dir: Path, prompt_path: Path, raw_prompt_path: Path) -> None:
    command = [
        sys.executable,
        str(script_dir / "generate_motion_prompt.py"),
        "--scene_dir",
        str(scene_dir),
        "--model",
        args.prompt_model,
        "--instruction",
        args.prompt_instruction,
        "--base_url",
        args.base_url,
        "--api_key_env",
        args.api_key_env,
        "--timeout",
        str(args.timeout),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--max_side",
        str(args.prompt_max_side),
        "--save_prompt_to",
        str(prompt_path),
        "--save_raw_to",
        str(raw_prompt_path),
    ]
    append_optional(command, "--ldr_folder_name", args.ldr_folder_name)
    append_optional(command, "--ldr_prefix", args.ldr_prefix)
    subprocess.run(command, check=True)


def run_sam3_masks(args, script_dir: Path, scene_dir: Path, prompt_path: Path) -> None:
    command = [
        sys.executable,
        str(script_dir / "generate_sam3_masks.py"),
        "--scene_dir",
        str(scene_dir),
        "--expected_frames",
        str(args.expected_frames),
        "--prompt_file",
        str(prompt_path),
        "--prompt_file_name",
        args.prompt_file_name,
        "--raw_prompt_file_name",
        args.raw_prompt_file_name,
        "--prompt_model",
        args.prompt_model,
        "--prompt_instruction",
        args.prompt_instruction,
        "--base_url",
        args.base_url,
        "--api_key_env",
        args.api_key_env,
        "--timeout",
        str(args.timeout),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--prompt_max_side",
        str(args.prompt_max_side),
        "--sam3_root",
        args.sam3_root,
        "--confidence_threshold",
        str(args.confidence_threshold),
        "--min_mask_area_ratio",
        str(args.min_mask_area_ratio),
        "--resolution",
        str(args.resolution),
        "--device",
        args.device,
    ]
    append_optional(command, "--preview_root", args.preview_root)
    append_optional(command, "--ldr_folder_name", args.ldr_folder_name)
    append_optional(command, "--ldr_prefix", args.ldr_prefix)
    if args.compile:
        command.append("--compile")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MLLM prompts and SAM3 masks for every scene in a ReAlignHDR dataset"
    )
    parser.add_argument("--input_root", required=True, help="Dataset root containing scene directories or split folders")
    parser.add_argument("--splits", nargs="*", help="Optional split folders under the dataset root")
    parser.add_argument("--scene_names", nargs="*", help="Optional subset of scene directory names")
    parser.add_argument("--ldr_folder_name", help="Optional LDR subfolder inside each scene")
    parser.add_argument("--ldr_prefix", help="Only use LDR files whose names start with this prefix")
    parser.add_argument("--expected_frames", type=int, required=True, help="Expected frame count; set 0 to disable")
    parser.add_argument("--preview_root", help="Optional root for PNG mask previews")
    parser.add_argument("--skip_existing_prompt", action="store_true", help="Reuse existing per-scene prompt files")

    parser.add_argument("--prompt_file_name", required=True, help="Per-scene normalized prompt filename")
    parser.add_argument("--raw_prompt_file_name", required=True, help="Per-scene raw prompt filename")
    parser.add_argument("--prompt_model", required=True, help="OpenAI-compatible prompt model")
    parser.add_argument("--prompt_instruction", required=True, help="Motion prompt instruction")
    parser.add_argument("--base_url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key_env", required=True, help="Environment variable containing the API key")
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

    input_root = Path(args.input_root).resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {input_root}")

    scene_dirs = list_scene_dirs(
        input_root=input_root,
        splits=args.splits,
        scene_names=set(args.scene_names) if args.scene_names else None,
        ldr_folder_name=args.ldr_folder_name,
        ldr_prefix=args.ldr_prefix,
    )

    script_dir = Path(__file__).resolve().parent
    total = len(scene_dirs)
    for index, scene_dir in enumerate(scene_dirs, start=1):
        prompt_path = scene_dir / args.prompt_file_name
        raw_prompt_path = scene_dir / args.raw_prompt_file_name
        print(f"[{index}/{total}] {scene_dir}")

        if args.skip_existing_prompt and prompt_path.is_file():
            print(f"  Reusing prompt: {prompt_path}")
        else:
            print("  Generating motion prompt")
            run_motion_prompt(args, script_dir, scene_dir, prompt_path, raw_prompt_path)

        print("  Generating SAM3 masks")
        run_sam3_masks(args, script_dir, scene_dir, prompt_path)


if __name__ == "__main__":
    main()
