"""Capture each dashboard tab with a headless browser (review aid only)."""
import sys, time
from playwright.sync_api import sync_playwright

out = sys.argv[1]
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1400, "height": 7000})
    page.goto("http://localhost:8501", wait_until="networkidle")
    page.wait_for_selector("text=RISKY | 30-Day Hospital Readmission Analytics", timeout=120000)
    time.sleep(5)
    for i, name in enumerate(["Dataset & EDA", "Model comparison", "Threshold exploration", "Calibration"], 1):
        page.get_by_role("tab", name=name).click()
        time.sleep(6)
        page.screenshot(path=f"{out}/{i}_{name.replace(' & ', '_').replace(' ', '_')}.png", full_page=True)
    b.close()
