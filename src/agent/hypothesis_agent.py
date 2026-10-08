from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import json
import re


class HypothesisAgent:
    """
    Generates research hypotheses, future work suggestions, and unexplored areas
    by analysing the collected papers' summaries, titles, and relationships.
    """

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, request_timeout=45, max_retries=6)
        self.prompt = PromptTemplate.from_template(
            "You are a senior research strategist. Below are the titles, summaries, "
            "and edge relationships of papers collected on the topic: '{query}'.\n\n"
            "{papers_context}\n\n"
            "Based on your analysis of these papers, generate exactly 5 research insights.\n\n"
            "You MUST use each of these 5 categories EXACTLY ONCE (one insight per category):\n"
            "  1. 'Hypothesis': A testable scientific claim that follows logically from the evidence.\n"
            "  2. 'Future Work': A concrete next step or extension that the field should pursue.\n"
            "  3. 'Unexplored': A gap where no existing work combines two or more ideas — a paradigm bridge.\n"
            "  4. 'Methodology Gap': A missing technique, tool, or benchmark that would accelerate progress.\n"
            "  5. 'Cross-Domain': An idea that bridges this field with a completely different discipline.\n\n"
            "Also assign a 'novelty' score from 50 to 98 reflecting how novel/non-obvious the idea is:\n"
            "  - 50-65: Incremental, fairly obvious extension\n"
            "  - 66-80: Meaningful contribution, non-trivial combination\n"
            "  - 81-98: Highly novel, paradigm-shifting insight\n\n"
            "Rules:\n"
            "- Each insight should be 1-2 sentences, specific and actionable.\n"
            "- Reference actual methods, datasets, or findings from the papers.\n"
            "- Do NOT repeat any category. Each of the 5 must be different.\n"
            "- Do NOT produce generic statements. Be specific to THIS set of papers.\n\n"
            "Return ONLY a JSON array in this format:\n"
            "[{{\n"
            '  "category": "Hypothesis|Future Work|Unexplored|Methodology Gap|Cross-Domain",\n'
            '  "text": "The specific insight text...",\n'
            '  "novelty": 75\n'
            "}}]\n\n"
            "Return ONLY the JSON array. No markdown, no explanation."
        )
        self.chain = self.prompt | self.llm

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n--- Hypothesis Agent ---")
        papers = state.get("ranked_papers", [])
        query = state.get("query", "")
        graph_data = state.get("graph_data", {})

        if not papers:
            print("No papers to generate hypotheses from.")
            state["hypotheses"] = []
            return state

        # Build context from papers: title + summary
        paper_lines = []
        for i, p in enumerate(papers[:10]):
            title = p.get("title", f"Paper {i}")
            summary = p.get("summary", "")
            citations = p.get("citations", 0)
            year = p.get("year", "Unknown")
            line = f"[{i+1}] \"{title}\" ({year}, {citations} citations)"
            if summary:
                # Take first 200 chars of summary for context
                line += f"\n    Summary: {summary[:200]}..."
            paper_lines.append(line)

        # Add edge relationships for richer context
        edges = graph_data.get("links", [])
        if edges:
            paper_lines.append("\n--- Key Relationships ---")
            for edge in edges[:15]:
                src_idx = edge.get("source", 0)
                tgt_idx = edge.get("target", 0)
                rel = edge.get("relationship", "weak")
                insight = edge.get("insight", "")
                src_title = papers[src_idx]["title"][:40] if src_idx < len(papers) else "?"
                tgt_title = papers[tgt_idx]["title"][:40] if tgt_idx < len(papers) else "?"
                paper_lines.append(f"  {src_title}... ↔ {tgt_title}... [{rel}]: {insight[:80]}")

        papers_context = "\n".join(paper_lines)

        try:
            result = self.chain.invoke({
                "query": query,
                "papers_context": papers_context
            })

            # Parse JSON response
            raw = re.sub(r"```[a-z]*", "", result.content).strip().strip("`")
            hypotheses = json.loads(raw)

            # Validate and normalize
            valid = []
            for h in hypotheses:
                category = h.get("category", "Hypothesis")
                if category not in ("Hypothesis", "Future Work", "Unexplored", "Methodology Gap", "Cross-Domain"):
                    category = "Hypothesis"
                novelty = max(50, min(98, int(h.get("novelty", 70))))
                valid.append({
                    "category": category,
                    "text": h.get("text", ""),
                    "novelty": novelty
                })

            state["hypotheses"] = valid[:5]
            print(f"Generated {len(valid[:5])} research hypotheses.")
            for h in valid[:5]:
                print(f"  [{h['category']}] (novelty: {h['novelty']}%) {h['text'][:80]}...")

        except Exception as e:
            print(f"Hypothesis generation failed: {e}")
            state["hypotheses"] = []

        return state
