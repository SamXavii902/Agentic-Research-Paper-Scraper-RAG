import asyncio
import os
import time
from playwright.async_api import async_playwright

class IEEEScraper:
    def __init__(self, download_dir="data/papers/ieee", state_file="ieee_state.json", log_callback=None):
        self.download_dir = download_dir
        self.state_file = state_file
        self.log_callback = log_callback
        os.makedirs(self.download_dir, exist_ok=True)

    async def _download_pdf(self, context, page, title):
        safe_title = "".join(x for x in title if x.isalnum() or x in " -_")[:100]
        filepath = os.path.join(self.download_dir, f"{safe_title}.pdf")
        
        if os.path.exists(filepath):
            print(f"Already downloaded: {filepath}")
            if self.log_callback:
                self.log_callback(f"[IEEE] Cached: {title[:60]}...")
            return filepath
            
        print(f"Attempting to download: {title}")
        if self.log_callback:
            self.log_callback(f"[IEEE] Downloading: {title[:60]}...")
        
        try:
            # Look for the PDF button (usually contains /stamp/stamp.jsp)
            pdf_button = page.locator('a[href*="/stamp/stamp.jsp"]')
            if await pdf_button.count() == 0:
                print(f"Could not find PDF button for {title}")
                return None
            
            stamp_href = await pdf_button.first.get_attribute("href")
            if not stamp_href.startswith("http"):
                stamp_href = f"https://ieeexplore.ieee.org{stamp_href}"
                
            print(f"Opening PDF viewer: {stamp_href}")
            await page.goto(stamp_href, wait_until="domcontentloaded")
            
            # The actual PDF is embedded in an iframe on the stamp page
            try:
                await page.wait_for_selector("iframe", timeout=15000)
            except:
                pass # might already be loaded or iframe not found
                
            pdf_url = None
            for frame in page.frames:
                if frame.url and (".pdf" in frame.url.lower() or "arnumber=" in frame.url.lower() and "stamp" not in frame.url.lower()):
                    pdf_url = frame.url
                    break
                    
            if not pdf_url:
                iframe = page.locator("iframe").first
                if await iframe.count() > 0:
                    pdf_url = await iframe.get_attribute("src")
                    
            if pdf_url:
                if not pdf_url.startswith("http"):
                    pdf_url = f"https://ieeexplore.ieee.org{pdf_url}"
                
                print(f"Fetching PDF bytes directly...")
                # Download using the authenticated context
                response = await context.request.get(pdf_url)
                if response.ok:
                    pdf_bytes = await response.body()
                    with open(filepath, "wb") as f:
                        f.write(pdf_bytes)
                    print(f"Successfully downloaded to {filepath}")
                    return filepath
                else:
                    print(f"Failed to fetch PDF: HTTP {response.status}")
            else:
                print("Could not extract PDF URL from iframe.")
                
            return None
        except Exception as e:
            print(f"Failed to download {title}: {e}")
            return None

    async def _extract_metadata(self, page):
        """Extract structured metadata from an IEEE paper page."""
        metadata = {}
        import re
        
        try:
            # Wait briefly for dynamic content to render
            await page.wait_for_timeout(2000)
            
            # --- YEAR ---
            # Strategy 1: "Published in:" line often contains "2024 IEEE 7th International Conference..."
            published_in_el = await page.query_selector('.doc-abstract-publishedin, .stats-document-abstract-publishedIn')
            if published_in_el:
                pub_text = await published_in_el.inner_text()
                year_match = re.search(r'\b(20\d{2})\b', pub_text)
                if year_match:
                    metadata['year'] = int(year_match.group())
            
            # Strategy 2: "Date of Conference" or "Date of Publication" fields
            if 'year' not in metadata:
                date_el = await page.query_selector('.doc-abstract-pubdate')
                if date_el:
                    date_text = await date_el.inner_text()
                    year_match = re.search(r'\b(20\d{2})\b', date_text)
                    if year_match:
                        metadata['year'] = int(year_match.group())
            
            # Strategy 3: Scan the full page text for conference year in header area
            if 'year' not in metadata:
                try:
                    header_text = await page.inner_text('xpl-root')
                    # Look for patterns like "2024 IEEE", "IEEE 2024"
                    conf_match = re.search(r'\b(20(?:1[5-9]|2[0-5]))\s*IEEE|IEEE\s*(20(?:1[5-9]|2[0-5]))\b', header_text[:2000])
                    if conf_match:
                        metadata['year'] = int(conf_match.group(1) or conf_match.group(2))
                except:
                    pass

            # --- CITATIONS ---
            # Strategy 1: Metrics banner (shows citation count prominently)
            cite_selectors = [
                '.document-banner-metric-count',
                '.citing-paper-count', 
                '.citation-count',
                '[data-tealium_data*="citedby"]'
            ]
            for selector in cite_selectors:
                cite_el = await page.query_selector(selector)
                if cite_el:
                    cite_text = (await cite_el.inner_text()).strip()
                    cite_match = re.search(r'(\d[\d,]*)', cite_text)
                    if cite_match:
                        metadata['citations'] = int(cite_match.group(1).replace(',', ''))
                        break
            
            # If no citation element found, default to 0 (NOT None — prevents OpenAlex fallback)
            if 'citations' not in metadata:
                metadata['citations'] = 0

            # --- ABSTRACT ---
            abstract_el = await page.query_selector('.abstract-text div, .abstract-desktop-div div')
            if abstract_el:
                metadata['abstract'] = (await abstract_el.inner_text()).strip()

            # --- AUTHORS ---
            author_els = await page.query_selector_all('.authors-info span.fw-bold, .authors-accordion-container a')
            authors = []
            for el in author_els[:10]:
                name = (await el.inner_text()).strip()
                if name and len(name) > 1:
                    authors.append(name)
            if authors:
                metadata['authors'] = authors

            # --- DOI ---
            doi_el = await page.query_selector('a[href*="doi.org"]')
            if doi_el:
                metadata['doi'] = (await doi_el.get_attribute('href'))

            print(f"  IEEE metadata: year={metadata.get('year')}, citations={metadata.get('citations')}, doi={metadata.get('doi', 'N/A')[:40]}")

        except Exception as e:
            print(f"Metadata extraction failed: {e}")

        return metadata

    async def search_and_download(self, query, max_results=3):
        if not os.path.exists(self.state_file):
            print(f"Error: {self.state_file} not found. Please run cookie_manager.py first.")
            return []

        async with async_playwright() as p:
            # Setting headless=False so we can see what's happening (bot detection, etc.)
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(storage_state=self.state_file, accept_downloads=True)
            page = await context.new_page()
            
            # Go to IEEE search — sort by Most Cited to get the best papers first
            print(f"Searching IEEE for: {query} (sorted by Most Cited)")
            import urllib.parse
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://ieeexplore.ieee.org/search/searchresult.jsp?newsearch=true&queryText={encoded_query}&sortType=paper-citations"
            
            await page.goto(search_url, wait_until="domcontentloaded")
            
            # Wait for results to load with better error handling
            try:
                await page.wait_for_selector("xpl-results-item, .List-results-items, .result-item", timeout=20000)
            except Exception as e:
                print(f"Timeout waiting for results: {e}")
                await page.screenshot(path="ieee_error_screenshot.png")
                html = await page.content()
                with open("ieee_error_page.html", "w", encoding="utf-8") as f:
                    f.write(html)
                print("Saved 'ieee_error_screenshot.png' and 'ieee_error_page.html' for debugging.")
                await browser.close()
                return []
            
            # Fallback: try clicking the "Most Cited" sort option if URL param didn't work
            try:
                sort_dropdown = page.locator('select.sort-order-select, .results-sort select, [aria-label*="Sort"]')
                if await sort_dropdown.count() > 0:
                    await sort_dropdown.first.select_option(label="Most Cited")
                    await page.wait_for_timeout(3000)
                    print("  Applied 'Most Cited' sort via dropdown.")
            except Exception:
                pass  # URL param likely worked already
            
            # Find paper links (xpl-results-item usually contains an h2 with an anchor tag for the title)
            # This avoids grabbing author names which are also 'a' tags
            paper_links = await page.locator("xpl-results-item h2 a, xpl-results-item h3 a").all()
            
            downloaded_papers = []
            seen_hrefs = set()
            
            # Limit results
            processed_count = 0
            for link in paper_links:
                if processed_count >= max_results:
                    break
                    
                title = await link.inner_text()
                href = await link.get_attribute("href")
                
                if href and title.strip() and href not in seen_hrefs:
                    seen_hrefs.add(href)
                    processed_count += 1
                    
                    full_url = f"https://ieeexplore.ieee.org{href}"
                    
                    # Open paper in new tab
                    new_page = await context.new_page()
                    await new_page.goto(full_url, wait_until="domcontentloaded")
                    
                    # Extract metadata from the paper page before downloading
                    metadata = await self._extract_metadata(new_page)
                    
                    # Try to download passing the context
                    filepath = await self._download_pdf(context, new_page, title)
                    if filepath:
                        paper_data = {
                            "pdf_path": filepath,
                            "title": title.strip(),
                            "source": "ieee",
                            "year": metadata.get("year"),
                            "citations": metadata.get("citations", 0),
                            "authors": metadata.get("authors", []),
                            "doi": metadata.get("doi"),
                        }
                        # Use page abstract if available
                        if metadata.get("abstract"):
                            paper_data["page_abstract"] = metadata["abstract"]
                        downloaded_papers.append(paper_data)
                        
                    await new_page.close()
                    # Polite delay (reduced to speed up pipeline)
                    await asyncio.sleep(1.5)
                    
            await browser.close()
            return downloaded_papers

if __name__ == "__main__":
    scraper = IEEEScraper()
    asyncio.run(scraper.search_and_download("hybrid cryptography steganography", max_results=1))
