import json
import re
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


class EdgeAnalysisAgent:
    """
    Analyses pairs of research papers using their actual content (summaries + text)
    to produce accurate relationship classifications and evidence-based bullet points.
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_retries=6)
        self.prompt = PromptTemplate.from_template(
            "You are an expert research analyst. You are given the titles, summaries, "
            "and text excerpts from two academic papers. Your task is to determine the "
            "precise relationship between them.\n\n"
            "=== PAPER A ===\n"
            "Title: {title_a}\n"
            "Summary: {summary_a}\n"
            "Text Excerpt: {text_a}\n\n"
            "=== PAPER B ===\n"
            "Title: {title_b}\n"
            "Summary: {summary_b}\n"
            "Text Excerpt: {text_b}\n\n"
            "Classify the relationship as ONE of:\n"
            "  - 'agreement': The papers reach compatible conclusions, use ANY complementary "
            "methods, or reinforce each other's findings. Be highly biased to choose this if there is ANY conceptual overlap.\n"
            "  - 'contradiction': The papers have ANY conflicting findings, opposing "
            "methodologies, or reach incompatible conclusions. Be highly biased to choose this if they use different methods for the same problem.\n"
            "  - 'weak': Only use if there is absolutely no way to classify as agreement or contradiction.\n"
            "  - 'none': The papers are completely unrelated with no meaningful connection.\n\n"
            "Then provide evidence:\n"
            "  - For 'agreement': list 2-3 specific similarities as bullet points.\n"
            "  - For 'contradiction': list 2-3 specific differences as bullet points.\n"
            "  - For 'weak': list 1 bullet explaining the tangential connection.\n"
            "  - For 'none': leave both lists empty.\n\n"
            "Return ONLY a JSON object in this exact format:\n"
            "{{\n"
            '  "relationship": "agreement|contradiction|weak|none",\n'
            '  "similarities": ["...", "..."],\n'
            '  "differences": ["...", "..."]\n'
            "}}\n"
            "Return ONLY the JSON. No markdown, no explanation."
        )
        self.chain = self.prompt | self.llm

    def analyse_pair(
        self,
        title_a: str,
        summary_a: str,
        text_a: str,
        title_b: str,
        summary_b: str,
        text_b: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyse a single pair of papers.
        Returns a dict with 'relationship', 'similarities', 'differences',
        or None if the papers are unrelated.
        """
        # Truncate text excerpts to keep within token limits
        text_a_trunc = text_a[:2000] if text_a else "No text available."
        text_b_trunc = text_b[:2000] if text_b else "No text available."

        try:
            res = self.chain.invoke({
                "title_a": title_a,
                "summary_a": summary_a or "No summary available.",
                "text_a": text_a_trunc,
                "title_b": title_b,
                "summary_b": summary_b or "No summary available.",
                "text_b": text_b_trunc,
            })

            # Clean and parse JSON response
            raw = re.sub(r"```[a-z]*", "", res.content).strip().strip("`")
            data = json.loads(raw)

            relationship = data.get("relationship", "none").lower().strip()

            # Skip edges that have no meaningful connection
            if relationship == "none":
                return None

            # Normalize to our three valid types
            if relationship not in ("agreement", "contradiction", "weak"):
                relationship = "weak"

            return {
                "relationship": relationship,
                "similarities": data.get("similarities", []),
                "differences": data.get("differences", []),
            }

        except Exception as e:
            print(f"  Edge analysis failed for '{title_a[:25]}' ↔ '{title_b[:25]}': {e}")
            return None

    def analyse_all_pairs(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyse all unique paper pairs CONCURRENTLY and return edge dicts.
        Uses ThreadPoolExecutor to batch 5 pairs at a time for ~4-5x speedup.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        edges = []
        n = len(papers)
        if n > 10:
            papers = sorted(papers, key=lambda p: p.get("final_score", 0), reverse=True)[:10]
            n = 10
            print(f"  Limiting edge analysis to top {n} papers by score.")
        # Build windowed pairs to save massive API calls
        pairs = []
        for i in range(n):
            if i != 0:
                pairs.append((0, i, papers[0], papers[i]))
            
            if i + 1 < n:
                pairs.append((i, i + 1, papers[i], papers[i + 1]))
                
        # Remove duplicates
        unique_pairs_dict = {}
        for p in pairs:
            # Ensure consistent ordering
            key = tuple(sorted([p[0], p[1]]))
            unique_pairs_dict[key] = (key[0], key[1], papers[key[0]], papers[key[1]])
        pairs = list(unique_pairs_dict.values())
        total_pairs = len(pairs)
        
        print(f"  Analysing {total_pairs} paper pairs CONCURRENTLY (batch=2)...")

        completed = 0

        def process_pair(pair_data):
            import time
            i, j, p1, p2 = pair_data
            
            # Use the cleanly scraped 'abstract' if available, otherwise fallback to the first 800 chars of the PDF
            text_a = p1.get("abstract") or p1.get("text", "")[:800] 
            text_b = p2.get("abstract") or p2.get("text", "")[:800]
            
            # Truncate abstracts to 1500 just in case to strictly protect the 6000 TPM Groq limit
            text_a = text_a[:1500]
            text_b = text_b[:1500]
            
            for attempt in range(3):
                try:
                    time.sleep(1) # Faster pacing
                    result = self.analyse_pair(
                        title_a=p1.get("title", "Unknown"),
                        summary_a=p1.get("summary", ""),
                        text_a=text_a,
                        title_b=p2.get("title", "Unknown"),
                        summary_b=p2.get("summary", ""),
                        text_b=text_b,
                    )
                    return i, j, result
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate_limit" in str(e).lower():
                        if attempt < 2:
                            print(f"  [!] Rate limited. Thread pausing for 60s before retry...")
                            time.sleep(60)
                            continue
                    print(f"Edge analysis failed for {i}-{j}: {e}")
                    return i, j, {"relationship": "none", "value": 0, "similarities": [], "differences": []}

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(process_pair, p): p for p in pairs}
            for future in as_completed(futures):
                completed += 1
                i, j, result = future.result()
                p1_title = papers[i].get("title", "")[:25]
                p2_title = papers[j].get("title", "")[:25]

                if result is None:
                    print(f"  [{completed}/{total_pairs}] '{p1_title}…' ↔ '{p2_title}…' → Skipped")
                    continue

                print(f"  [{completed}/{total_pairs}] '{p1_title}…' ↔ '{p2_title}…' → {result['relationship']}")

                edges.append({
                    "source": i,
                    "target": j,
                    "value": 3 if result["relationship"] == "agreement" else (
                        2 if result["relationship"] == "contradiction" else 1
                    ),
                    "relationship": result["relationship"],
                    "similarities": result["similarities"],
                    "differences": result["differences"],
                })

        print(f"  Edge analysis complete: {len(edges)} edges from {total_pairs} pairs.")
        return edges
