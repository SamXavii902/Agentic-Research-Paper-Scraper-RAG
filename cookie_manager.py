import asyncio
from playwright.async_api import async_playwright

async def get_ieee_cookie():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        print("Opening IEEE Xplore...")
        await page.goto("https://ieeexplore.ieee.org/")
        
        print("\n*** ACTION REQUIRED ***")
        print("Please log in to your institutional account (NIT Rourkela) in the browser window.")
        print("Once you are logged in and the page has fully loaded, come back here and press ENTER.")
        input("Press ENTER to save session state...")
        
        await context.storage_state(path="ieee_state.json")
        print("Session state saved to ieee_state.json!")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(get_ieee_cookie())
