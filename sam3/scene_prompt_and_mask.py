import argparse
import os
import subprocess
import sys
from pathlib import Path

from generate_motion_prompt import DEFAULT_INSTRUCTION, generate_motion_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GPT-5.4 motion prompt and run SAM3 mask generation")
    parser.add_argument("--scene_dir", required=True, help="Path to one scene directory")
    parser.add_argument("--output_root", required=True, help="Root directory for SAM3 preview outputs")
    parser.add_argument("--model", default="gpt-5.4", help="OpenAI model name")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Motion-identification instruction")
    parser.add_argument("--base_url", default="https://api.openai.com/v1", help="OpenAI API base URL")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Environment variable containing API key")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP request timeout in seconds")
    parser.add_argument("--save_prompt_to", default=None, help="Optional path to save the normalized prompt")
    parser.add_argument("--save_raw_to", default=None, help="Optional path to save the raw model output")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {args.api_key_env}")

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

    print(f"Motion prompt: {prompt!r}")

    segmask_path = Path(__file__).resolve().parent / "segmask.py"
    cmd = [
        sys.executable,
        str(segmask_path),
        "--scene_dir",
        str(scene_dir),
        "--output_root",
        args.output_root,
        "--prompt",
        prompt,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
