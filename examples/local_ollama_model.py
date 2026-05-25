"""Local Ollama split-execution example for AskUI.

This example bypasses the unified `act()` route because local Ollama models
currently do not reliably support both image input and tool schemas in the same
SDK path. Instead, it uses:

1. A local vision model to inspect the current screen and return coordinates.
2. A local text model to generate AskUI automation code for those coordinates.
3. The AskUI controller to execute the generated action directly.

Run with:
    python examples/local_ollama_model.py --target "the red close button"

Set these environment variables if needed:
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_VISION_MODEL=qwen3.5:7b
    OLLAMA_TOOL_MODEL=llama3.1:8b
"""

import argparse
import base64
import io
import os
import re
from typing import Any

import requests

from askui.tools.askui import AskUiControllerClient


class OllamaError(RuntimeError):
    """Raised when an Ollama request fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local two-step Ollama + AskUI pipeline."
    )
    parser.add_argument(
        "--target",
        default="the active target",
        help="Natural-language description of the UI element you want to act on.",
    )
    parser.add_argument(
        "--action",
        choices=("click", "move"),
        default="click",
        help="Which action to perform after coordinates are extracted.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Base URL for the local Ollama API.",
    )
    parser.add_argument(
        "--vision-model",
        default=os.environ.get("OLLAMA_VISION_MODEL", "qwen3.5:7b"),
        help="Ollama model to use for the vision step.",
    )
    parser.add_argument(
        "--tool-model",
        default=os.environ.get("OLLAMA_TOOL_MODEL", "llama3.1:8b"),
        help="Ollama model to use for the text/action step.",
    )
    return parser.parse_args()


def _request_ollama(url: str, payload: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    if not response.ok:
        error_msg = f"Ollama request failed: {response.status_code} {response.text}"
        raise OllamaError(error_msg)
    return response.json()


def image_to_base64(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_coordinates(text: str) -> tuple[int, int]:
    match = re.search(r"(-?\d+)\s*(?:,|x|\bX\b|y|\bY\b)\s*(-?\d+)", text)
    if match is None:
        raise OllamaError(
            f"Could not extract coordinates from model output: {text!r}"
        )
    return int(match.group(1)), int(match.group(2))


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:python)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def vision_step(base_url: str, vision_model: str, image: Any, target: str) -> str:
    prompt = (
        "Return only the exact pixel coordinates for the target. "
        f"Target: {target}. "
        "Use the format [x, y] or x, y."
    )
    payload = {
        "model": vision_model,
        "prompt": prompt,
        "images": [image_to_base64(image)],
        "stream": False,
    }
    data = _request_ollama(f"{base_url}/api/generate", payload)
    response = data.get("response")
    if not isinstance(response, str) or not response.strip():
        raise OllamaError("Vision model returned an empty response")
    return response.strip()


def tool_step(base_url: str, tool_model: str, coordinates_text: str, action: str) -> str:
    prompt = (
        "You are an AskUI automation assistant. "
        "Output only Python code, no markdown fences, and nothing else. "
        "Use the existing `controller` object. "
        "Only use `controller.mouse_move(x, y)` and `controller.click()`. "
        f"Coordinates text: {coordinates_text}. "
        f"Desired action: {action}."
    )
    payload = {
        "model": tool_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an AskUI automation assistant. "
                    "Return only Python code using the existing `controller` object."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    data = _request_ollama(f"{base_url}/api/chat", payload)
    message = data.get("message", {}).get("content")
    if not isinstance(message, str) or not message.strip():
        raise OllamaError("Tool model returned an empty response")

    code = _strip_code_fences(message)
    if not code:
        raise OllamaError("Tool model returned an empty code snippet")
    return code


def execute_action(controller: AskUiControllerClient, action: str, x: int, y: int) -> None:
    if action == "move":
        controller.mouse_move(x, y)
        return

    controller.mouse_move(x, y)
    controller.click()


def run_pipeline(
    target: str,
    action: str,
    base_url: str,
    vision_model: str,
    tool_model: str,
) -> None:
    with AskUiControllerClient() as controller:
        image = controller.screenshot()
        coordinates_text = vision_step(base_url, vision_model, image, target)
        x, y = extract_coordinates(coordinates_text)
        generated_code = tool_step(
            base_url,
            tool_model,
            f"Coordinates: {x}, {y}",
            action,
        )

        print("Generated AskUI code:")
        print(generated_code)
        execute_action(controller, action, x, y)


def main() -> None:
    args = parse_args()
    run_pipeline(
        target=args.target,
        action=args.action,
        base_url=args.base_url,
        vision_model=args.vision_model,
        tool_model=args.tool_model,
    )


if __name__ == "__main__":
    main()
