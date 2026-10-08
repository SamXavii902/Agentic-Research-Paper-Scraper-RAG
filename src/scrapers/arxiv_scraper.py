import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os
import time
import requests
import re
from difflib import SequenceMatcher

class ArxivScraper:
    def __init__(self, download_dir="data/papers/arxiv", log_callback=None):
        self.download_dir = download_dir
        self.log_callback = log_callback
        os.makedirs(self.download_dir, exist_ok=True)
        self.base_url = "http://export.arxiv.org/api/query?"

    def _get_citation_counts(self, candidates):
        """
        Batch-fetch citation counts from OpenAlex for a list of candidate papers.
        Uses title similarity validation to avoid wrong-paper matches.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def fetch_citation(paper):
            title = paper.get("title", "")
            try:
                safe_title = urllib.parse.quote(title)
                url = f"https://api.openalex.org/works?search={safe_title}&mailto=research_bot_{int(time.time())}@gmail.com"
                resp = requests.get(url, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    if "results" in data and len(data["results"]) > 0:
                        top = data["results"][0]
                        api_title = top.get("title", "")
                        # Validate title similarity to avoid wrong-paper match
                        a = re.sub(r'[^a-z0-9\s]', '', title.lower()).strip()
                        b = re.sub(r'[^a-z0-9\s]', '', api_title.lower()).strip()
                        sim = SequenceMatcher(None, a, b).ratio()
                        if sim >= 0.65:
                            paper["citations"] = top.get("cited_by_count", 0)
                        else:
                            paper["citations"] = None
                    else:
                        paper["citations"] = None
                else:
                    paper["citations"] = None
            except Exception:
                paper["citations"] = None
            # Removed time.sleep(0.3) since we batch them and OpenAlex handles some concurrency
            return paper

        # Reduced max_workers to 2 to prevent 429 Too Many Requests from OpenAlex
        with ThreadPoolExecutor(max_workers=2) as executor:
            # We submit all candidates. Max workers=2 is extremely polite to OpenAlex API
            futures = [executor.submit(fetch_citation, p) for p in candidates]
            for future in as_completed(futures):
                # The paper dict is updated in-place, so we just wait for all to finish
                future.result()

    def search_and_download(self, query, max_results=10):
        print(f"Searching arXiv for: {query} (will rank by citations)")
        # Format the query for arXiv API by combining terms with AND
        # Ignore common stop words
        terms = [t for t in query.split() if t.lower() not in ('and', 'or', 'the', 'of', 'in', 'for', 'a', 'an')]
        arxiv_query = "all:" + " ".join(terms[:4]) # Keep it broad
        encoded_query = urllib.parse.quote(arxiv_query)
        
        # Fetch 3x more candidates than needed so we can pick the best
        fetch_count = max_results * 3
        url = f"{self.base_url}search_query={encoded_query}&start=0&max_results={fetch_count}&sortBy=relevance&sortOrder=descending"
        
        try:
            time.sleep(2)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 AgenticScraper/1.0'})
            response = urllib.request.urlopen(req)
            xml_data = response.read()
            root = ET.fromstring(xml_data)
        except Exception as e:
            err_msg = f"Failed to search arXiv: {e}"
            print(err_msg)
            if self.log_callback:
                self.log_callback(f"[ArXiv] <span style='color:red'>{err_msg}</span>")
            return []
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        
        # Phase 1: Parse all candidates (no downloads yet)
        candidates = []
        for entry in root.findall('atom:entry', namespace):
            title = entry.find('atom:title', namespace).text.replace('\n', '').strip()
            
            published_raw = entry.find('atom:published', namespace)
            year = int(published_raw.text[:4]) if published_raw is not None and published_raw.text else None
            
            abstract_el = entry.find('atom:summary', namespace)
            abstract = abstract_el.text.replace('\n', ' ').strip() if abstract_el is not None and abstract_el.text else ""
            
            authors = []
            for author_el in entry.findall('atom:author', namespace):
                name_el = author_el.find('atom:name', namespace)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())
            
            categories = []
            for cat_el in entry.findall('atom:category', namespace):
                term = cat_el.attrib.get('term')
                if term:
                    categories.append(term)
            
            pdf_url = None
            for link in entry.findall('atom:link', namespace):
                if link.attrib.get('title') == 'pdf':
                    pdf_url = link.attrib.get('href')
                    break
                    
            if pdf_url:
                if not pdf_url.endswith('.pdf'):
                    pdf_url += ".pdf"
                candidates.append({
                    "title": title,
                    "year": year,
                    "authors": authors,
                    "abstract": abstract,
                    "categories": categories,
                    "source": "arxiv",
                    "pdf_url": pdf_url,
                    "citations": 0
                })
        
        print(f"  Found {len(candidates)} arXiv candidates. Fetching citation counts...")
        if self.log_callback:
            self.log_callback(f"[ArXiv] Found {len(candidates)} candidates. Ranking by citations...")
        
        # Phase 2: Get citation counts from OpenAlex
        self._get_citation_counts(candidates)
        
        # Phase 3: Sort by citations (descending) and take top N
        for p in candidates:
            if p.get("citations") is None:
                p["citations"] = 0
        candidates.sort(key=lambda p: p.get("citations") or 0, reverse=True)
        top_papers = candidates[:max_results]
        
        print(f"  Top {len(top_papers)} papers by citations:")
        for i, p in enumerate(top_papers):
            print(f"    {i+1}. [{p.get('citations') or 0} cites] {p['title'][:70]}...")
        
        # Phase 4: Download only the top-ranked papers
        downloaded_papers = []
        for paper in top_papers:
            title = paper["title"]
            safe_title = "".join(x for x in title if x.isalnum() or x in " -_")[:100]
            filepath = os.path.join(self.download_dir, f"{safe_title}.pdf")
            
            if not os.path.exists(filepath):
                print(f"Downloading: {title[:70]}...")
                if self.log_callback:
                    self.log_callback(f"[ArXiv] Downloading: {title[:60]}...")
                try:
                    urllib.request.urlretrieve(paper["pdf_url"], filepath)
                    time.sleep(3)
                except Exception as e:
                    print(f"Failed to download {paper['pdf_url']}: {e}")
                    continue
            else:
                print(f"Already exists: {filepath}")
                if self.log_callback:
                    self.log_callback(f"[ArXiv] Cached: {title[:60]}...")
            
            paper["pdf_path"] = filepath
            del paper["pdf_url"]  # Clean up internal field
            downloaded_papers.append(paper)
                    
        return downloaded_papers

if __name__ == "__main__":
    scraper = ArxivScraper()
    # Test
    scraper.search_and_download("Graph Neural Networks Drug Discovery", max_results=5)
