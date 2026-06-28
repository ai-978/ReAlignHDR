from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any


MOTION_INSTRUCTION_PROMPT = (
    "Identify the object categories that exhibit clear relative motion across the three frames. "
    "Return only a comma-separated list of object categories."
)

SUPPORTED_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_scene_frames(scene_dir: Path, ldr_folder_name: str | None = None, ldr_prefix: str = "") -> list[Path]:
    frame_dir = scene_dir / ldr_folder_name if ldr_folder_name else scene_dir
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

    tif_frames = sorted(
        p for p in frame_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
        and p.name.startswith(ldr_prefix)
    )
    frames = tif_frames or sorted(
        p for p in frame_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTS
        and p.name.startswith(ldr_prefix)
    )
    if len(frames) != 3:
        raise ValueError(f"Expected exactly 3 image frames in {frame_dir}, found {len(frames)}")
    return frames


def image_to_data_url(image_path: Path, max_side: int = 1024, jpeg_quality: int = 90) -> str:
    from PIL import Image

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    outputs = payload.get("output", [])
    for item in outputs:
        if isinstance(item, dict) and isinstance(item.get("content"), list):
            for content in item["content"]:
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        text = item.get("text") if isinstance(item, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()

    raise RuntimeError(f"Could not extract text from Responses API payload: {json.dumps(payload)[:1000]}")


def normalize_prompt_text(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        return ""

    text = text.strip("` \n\t")
    if text.lower() in {"none", "no motion", "no clear motion", "no moving object", "empty"}:
        return ""

    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                if isinstance(data.get("prompt"), str):
                    text = data["prompt"]
                elif isinstance(data.get("categories"), list):
                    text = ", ".join(str(item) for item in data["categories"])
        except json.JSONDecodeError:
            pass

    text = re.sub(r"^[A-Za-z ]*:\s*", "", text).strip()
    text = text.replace("\n", ",")
    text = text.replace(";", ",")

    parts = []
    seen = set()
    for part in text.split(","):
        item = part.strip().strip(".").strip('"').strip("'")
        item = re.sub(r"^\d+[\.\)]\s*", "", item)
        item = re.sub(r"^(a|an|the)\s+", "", item, flags=re.IGNORECASE)
        item = item.lower().strip()
        if not item or item in {"none", "no motion", "no clear motion"}:
            continue
        if item not in seen:
            seen.add(item)
            parts.append(item)

    return ", ".join(parts)


def generate_motion_prompt(
    scene_dir: Path,
    model: str,
    instruction: str,
    api_key: str,
    base_url: str,
    timeout: float,
    ldr_folder_name: str | None = None,
    ldr_prefix: str = "",
    temperature: float | None = None,
    top_p: float | None = None,
    max_side: int | None = None,
) -> tuple[str, str]:
    import httpx

    if temperature is None:
        raise ValueError("temperature must be specified explicitly.")
    if top_p is None:
        raise ValueError("top_p must be specified explicitly.")
    if max_side is None:
        raise ValueError("max_side must be specified explicitly.")

    frame_paths = list_scene_frames(scene_dir, ldr_folder_name=ldr_folder_name, ldr_prefix=ldr_prefix)
    content = [{"type": "input_text", "text": instruction}]
    for frame_path in frame_paths:
        content.append({"type": "input_image", "image_url": image_to_data_url(frame_path, max_side=max_side)})

    response = httpx.post(
        f"{base_url.rstrip('/')}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": [{"role": "user", "content": content}],
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": 128,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    raw_text = extract_response_text(payload)
    prompt = normalize_prompt_text(raw_text)
    return prompt, raw_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a motion-category prompt for one ReAlignHDR scene")
    parser.add_argument("--scene_dir", required=True, help="Scene directory containing the three LDR frames")
    parser.add_argument("--ldr_folder_name", help="Optional LDR subfolder inside each scene")
    parser.add_argument("--ldr_prefix", help="Only use LDR files whose names start with this prefix")
    parser.add_argument("--model", required=True, help="OpenAI-compatible model name")
    parser.add_argument("--instruction", required=True, help="Motion-identification instruction")
    parser.add_argument("--base_url", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key_env", required=True, help="Environment variable containing the API key")
    parser.add_argument("--timeout", type=float, required=True, help="HTTP timeout in seconds")
    parser.add_argument("--temperature", type=float, required=True, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, required=True, help="Nucleus sampling probability")
    parser.add_argument("--max_side", type=int, required=True, help="Maximum encoded image side length")
    parser.add_argument("--save_raw_to", help="Optional file path for raw model output")
    parser.add_argument("--save_prompt_to", help="Optional file path for normalized prompt")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {args.api_key_env}")

    prompt, raw_text = generate_motion_prompt(
        scene_dir=Path(args.scene_dir).resolve(),
        model=args.model,
        instruction=args.instruction,
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
        ldr_folder_name=args.ldr_folder_name,
        ldr_prefix=args.ldr_prefix or "",
        temperature=args.temperature,
        top_p=args.top_p,
        max_side=args.max_side,
    )

    if args.save_raw_to:
        raw_path = Path(args.save_raw_to)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw_text, encoding="utf-8")

    if args.save_prompt_to:
        prompt_path = Path(args.save_prompt_to)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")

    print(prompt)


if __name__ == "__main__":
    main()
