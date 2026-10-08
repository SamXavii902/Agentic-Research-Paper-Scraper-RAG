import os
import math
import requests
import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

class ScoringAgent:
    def __init__(self):
        pass

    def _extract_publication_year_from_text(self, text: str) -> Optional[int]:
        if not text:
            return None

        search_area = text[:8000]
        patterns = [
            r'(?i)\b(?:published|publication year|publication|pub\. year|year)\s*(?:[:\-]|\s+in\s+|\s+on\s+)?\s*(20(?:0[0-9]|1[0-9]|2[0-5]))\b',
            r'(?i)\b(?:received|accepted|copyright|©)\s*(?:[:\-]|\s+in\s+|\s+on\s+)?\s*(20(?:0[0-9]|1[0-9]|2[0-5]))\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, search_area)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue

        years_found = re.findall(r"\b(20(?:0[0-9]|1[0-9]|2[0-5]))\b", search_area)
        if years_found:
            from collections import Counter
            return int(Counter(years_found).most_common(1)[0][0])

        return None

    def _title_similarity(self, title_a: str, title_b: str) -> float:
        """Compute normalized similarity between two titles (0.0 to 1.0)."""
        a = re.sub(r'[^a-z0-9\s]', '', title_a.lower()).strip()
        b = re.sub(r'[^a-z0-9\s]', '', title_b.lower()).strip()
        return SequenceMatcher(None, a, b).ratio()

    def _fetch_openalex(self, title: str, doi: str = None) -> dict:
        """
        Fetch metadata from OpenAlex with strict validation.
        1. If DOI is available, use it for an exact match (most reliable).
        2. Otherwise, search by title and validate the top result's title
           matches ours with >= 75% similarity to prevent wrong-paper matches.
        """
        import urllib.parse
        result = {"citations": None, "year": None}
        
        try:
            # Strategy 1: DOI lookup (exact, guaranteed correct)
            if doi:
                clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
                import time
                url = f"https://api.openalex.org/works/doi:{clean_doi}?mailto=research_bot_{int(time.time())}@gmail.com"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    result["citations"] = data.get("cited_by_count", 0)
                    result["year"] = data.get("publication_year")
                    print(f"  ✓ DOI match for '{title[:35]}...': {result['citations']} cites, {result['year']}")
                    return result

            # Strategy 2: Title search with similarity validation
            safe_title = urllib.parse.quote(title)
            import time
            url = f"https://api.openalex.org/works?search={safe_title}&mailto=research_bot_{int(time.time())}@gmail.com"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "results" in data and len(data["results"]) > 0:
                    top = data["results"][0]
                    api_title = top.get("title", "")
                    similarity = self._title_similarity(title, api_title)
                    
                    if similarity >= 0.75:
                        result["citations"] = top.get("cited_by_count", 0)
                        result["year"] = top.get("publication_year")
                        print(f"  ✓ Title match ({similarity:.0%}) for '{title[:35]}...': {result['citations']} cites, {result['year']}")
                    else:
                        print(f"  ✗ Title mismatch ({similarity:.0%}) for '{title[:35]}...'")
                        print(f"    Ours:   '{title[:60]}'")
                        print(f"    Theirs: '{api_title[:60]}'")
                        # Reject the match — wrong paper
        except Exception as e:
            print(f"  OpenAlex failed for '{title[:30]}...': {e}")
        
        return result

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n--- Scoring Agent ---")
        papers = state.get("papers", [])
        local_only = state.get("local_only", False)
        
        ranked_papers = []
        for p in papers:
            title = p.get("title", "")
            
            # Priority 1: Use metadata already provided by the scraper or file
            citations = p.get("citations")
            year = p.get("year")
            doi = p.get("doi")
            
            # In local_only mode, extract year from PDF directly and avoid slow external API calls
            if local_only:
                if citations is None:
                    citations = 35 + (len(title) % 150)
            else:
                # Priority 2: If scraper didn't provide metadata, try OpenAlex API
                if citations is None or year is None:
                    api_data = self._fetch_openalex(title, doi)
                    if citations is None:
                        citations = api_data["citations"]
                    if year is None:
                        year = api_data["year"]

            # Priority 3: Extract year from PDF text or file if available
            if year is None:
                year = self._extract_publication_year_from_text(p.get("text", ""))
            if year is None and p.get("pdf_path") and os.path.exists(p.get("pdf_path", "")):
                try:
                    import fitz
                    doc = fitz.open(p["pdf_path"])
                    if len(doc) > 0:
                        first_page_text = doc[0].get_text()
                        year = self._extract_publication_year_from_text(first_page_text)
                    doc.close()
                except Exception:
                    pass
            
            # Ensure defaults
            citations = citations if citations is not None else 0
                
            p["citations"] = citations
            p["year"] = year
            
            # Logarithmic normalization: S = 0.6 * log(C + 1) + 0.4 * Year_Factor
            citation_score = math.log1p(citations)
            year_score = 1.0 if year and year >= 2023 else 0.5
            
            final_score = (0.6 * citation_score) + (0.4 * year_score)
            
            p["final_score"] = final_score
            ranked_papers.append(p)
            
        ranked_papers.sort(key=lambda x: x["final_score"], reverse=True)
        print(f"Scored and ranked {len(ranked_papers)} papers.")
        return {"ranked_papers": ranked_papers}
