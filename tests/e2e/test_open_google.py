"""End-to-end test that opens a local browser and searches Google."""

import random
import re
import string

from playwright.sync_api import sync_playwright

from askui.tools.askui import AskUiControllerClient


def _screen_position_from_box(page, box: dict[str, float]) -> tuple[int, int]:
    """Convert a Playwright viewport box into a global screen coordinate."""
    screen_origin = page.evaluate("() => ({ x: window.screenX, y: window.screenY })")
    x = int(screen_origin["x"] + box["x"] + box["width"] / 2)
    y = int(screen_origin["y"] + box["y"] + box["height"] / 2)
    return x, y


def test_open_google_in_local_browser() -> None:
    """Open Chromium, type a random query, and click the search button."""
    query = "askui" + "".join(random.choices(string.ascii_lowercase, k=6))
    controller = AskUiControllerClient()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            page.goto("https://www.msn.com", wait_until="load")
            page.bring_to_front()

            search_input = page.locator('textarea[name="q"]')
            search_input.wait_for()
            search_input_box = search_input.bounding_box()
            assert search_input_box is not None

            controller.connect()
            search_input_position = _screen_position_from_box(page, search_input_box)
            controller.mouse_move(*search_input_position)
            controller.click()
            controller.type(query)

            search_button = page.locator('input[name="btnK"]').first
            search_button.wait_for(state="visible")
            search_button_box = search_button.bounding_box()
            assert search_button_box is not None

            search_button_position = _screen_position_from_box(page, search_button_box)
            controller.mouse_move(*search_button_position)
            controller.click()

            page.wait_for_function(
                "() => location.href.includes('/search') || document.title.includes('Search')",
                timeout=10000,
            )
            search_box_value = page.locator('textarea[name="q"]').input_value()
            page_title = page.title()
            page_url = page.url
        finally:
            controller.disconnect()
            browser.close()

    assert "google.com/search" in page_url
    assert query.lower() in page_title.lower()
    assert search_box_value == query
