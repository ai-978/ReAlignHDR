import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from PIL import Image


DEFAULT_INSTRUCTION = (
    "Identify the object categories that exhibit clear relative motion across the three frames."
)

DEFAULT_OUTPUT_INSTRUCTION = (
    "Return only a comma-separated list of object categories suitable for segmentation prompts. "
    "Use short singular nouns such as 'person, bicycle, dog'. "
    "Include only categories with clear relative motion across the three frames. "
    "If no object category shows clear relative motion, return an empty string."
)

SUPPORTED_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_scene_frames(scene_dir: Path) -> list[Path]:
    frames = sorted(
        p for p in scene_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if len(frames) < 3:
        raise ValueError(f"Expected at least 3 image frames in {scene_dir}, found {len(frames)}")
    return frames[:3]


def image_to_data_url(image_path: Path, max_side: int = 1024, jpeg_quality: int = 90) -> str:
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
) -> tuple[str, str]:
    frame_paths = list_scene_frames(scene_dir)
    content = [
        {
            "type": "input_text",
            "text": (
                f"{instruction}\n\n"
                f"{DEFAULT_OUTPUT_INSTRUCTION}\n"
                f"Scene directory: {scene_dir.name}"
            ),
        }
    ]

    for frame_path in frame_paths:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(frame_path),
            }
        )

    response = httpx.post(
        f"{base_url.rstrip('/')}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": [{"role": "user", "content": content}],
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
    parser = argparse.ArgumentParser(description="Generate motion-category prompt for one scene with GPT-5.4")
    parser.add_argument("--scene_dir", required=True, help="Path to one scene directory containing the 3 LDR images")
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI model name")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Primary motion-identification instruction")
    parser.add_argument("--base_url", default="https://api.openai.com/v1", help="OpenAI API base URL")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Environment variable containing API key")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP request timeout in seconds")
    parser.add_argument("--save_raw_to", default=None, help="Optional file path to save the raw model output")
    parser.add_argument("--save_prompt_to", default=None, help="Optional file path to save the normalized prompt")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {args.api_key_env}")

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    prompt, raw_text = generate_motion_prompt(
        scene_dir=scene_dir,
        model=args.model,
        instruction=args.instruction,
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
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
