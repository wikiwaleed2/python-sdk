"""End-to-end test for opening Calculator and performing a calculation."""

import subprocess
import sys
import time

import pytest

from askui.tools.askui import AskUiControllerClient


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only calculator test")
def test_calculator_addition_with_keyboard_and_clipboard() -> None:
    """Open Calculator, type an expression, copy the result, and assert it."""
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Set-Clipboard -Value ''",
        ],
        check=False,
    )

    process = subprocess.Popen(["calc.exe"])

    try:
        with AskUiControllerClient() as controller:
            time.sleep(2)
            controller.type("1+2=")
            time.sleep(1)
            controller.keyboard_tap("c", modifier_keys=["control"])
            time.sleep(1)

        clipboard = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-Clipboard",
            ],
            capture_output=True,
            check=False,
            text=True,
        ).stdout.strip()

        assert clipboard == "3"
    finally:
        process.kill()
        process.wait(timeout=5)
