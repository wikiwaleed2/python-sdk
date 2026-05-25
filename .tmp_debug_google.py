from playwright.sync_api import sync_playwright

p = sync_playwright().start()
b = p.chromium.launch(headless=False)
page = b.new_page()
page.goto("https://www.google.com", wait_until="load")
print("TITLE", page.title())
print("ROLE_BUTTON_COUNT", page.get_by_role("button").count())
print("BTN_K_COUNT", page.locator('input[name="btnK"]').count())
print("SEARCH_INPUT_VISIBLE", page.locator('textarea[name="q"]').is_visible())
print("SEARCH_INPUT_COUNT", page.locator('textarea[name="q"]').count())
print("HTML_SNIPPET", page.locator("body").inner_html()[:2000])
b.close()
p.stop()
