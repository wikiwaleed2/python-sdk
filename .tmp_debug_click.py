from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto('https://www.google.com', wait_until='load')
    search_input = page.locator('textarea[name="q"]')
    search_input.wait_for()
    search_input.fill('askui test')
    btn = page.locator('input[name="btnK"]').first
    btn.wait_for(state='visible')
    print('box', btn.bounding_box())
    btn.click()
    page.wait_for_timeout(3000)
    print('url', page.url)
    print('title', page.title())
    browser.close()
