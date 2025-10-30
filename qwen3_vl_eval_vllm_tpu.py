import argparse
import base64
import io
import json
import time
from typing import List

import requests
from PIL import Image


def encode_image_to_base64_jpeg(image_path: str, quality: int = 95) -> str:
    image = Image.open(image_path)
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def send_chat_with_image(
    server_url: str,
    model: str,
    prompt: str,
    image_b64_url: str,
    max_new_tokens: int,
    temperature: float,
    seed: int = 42,
) -> str:
    url = server_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful vision-language assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_b64_url}},
                ],
            },
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(payload), headers=headers, timeout=300)
    response.raise_for_status()
    data = response.json()
    # vLLM OpenAI-compatible response
    return data["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL-4B-Instruct via TPU vLLM OpenAI API")
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://127.0.0.1:8000/v1",
        help="Base URL of the OpenAI-compatible server (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Model identifier as served by vLLM (override if your server uses a different name)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/nfs/aidm_nfs/gene/testing_vlms/im2.png",
        help="Path to the evaluation image",
    )
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Prompts mirroring Gemma example (no special image token needed for OpenAI API)
    prompts: List[str] = [
        "What type of tile is the character facing in the image? E.g. grass, dirt, stone, etc.",
        "How much health points does the character have?",
        "Are there any animals in the image?",
        "Are there any plants in the image?",
        "Are there any trees in the image?",
        "Is there any water in the image?",
    ]

    image_b64_url = encode_image_to_base64_jpeg(args.image, quality=95)

    for idx, prompt in enumerate(prompts, start=1):
        start_time = time.perf_counter()
        out_text = send_chat_with_image(
            server_url=args.server_url,
            model=args.model,
            prompt=prompt,
            image_b64_url=image_b64_url,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - start_time
        print(out_text)
        ordinal = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"][idx - 1]
        print(f"{ordinal} inference: {elapsed:.3f} s")


if __name__ == "__main__":
    main()




