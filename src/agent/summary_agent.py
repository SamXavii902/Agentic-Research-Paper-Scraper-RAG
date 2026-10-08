import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any

class SummaryAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_retries=6)
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n--- Summary Agent ---")
        papers = state.get("ranked_papers", [])
        
        if not papers:
            print("No papers to summarize.")
            return state

        prompt = PromptTemplate.from_template(
            "You are an expert academic research analyst. Below is the text extracted from a research paper titled '{title}'.\n"
            "Your task is to write a comprehensive 'Core Summary' consisting of exactly 3 paragraphs:\n\n"
            "Paragraph 1 — OBJECTIVE & CONTEXT: What specific problem does this paper address? "
            "Why is it important? What gap in existing research does it fill?\n\n"
            "Paragraph 2 — METHODOLOGY: What techniques, algorithms, datasets, or experimental "
            "setups are used? Mention specific tools, models, or frameworks by name.\n\n"
            "Paragraph 3 — KEY FINDINGS & TAKEAWAYS: What are the main results? "
            "What can a reader learn from this paper? How does it advance the field?\n\n"
            "Rules:\n"
            "- Do NOT start any paragraph with 'This paper' or 'The authors'. Jump straight into the substance.\n"
            "- Be specific — use numbers, method names, and concrete details from the text.\n"
            "- Each paragraph should be 3-4 sentences.\n"
            "- Separate paragraphs with a blank line.\n\n"
            "Extracted Text:\n{text}\n\n"
            "Core Summary:"
        )
        
        chain = prompt | self.llm
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        print(f"Generating summaries for {len(papers)} papers CONCURRENTLY...")
        
        def process_paper(p):
            import time
            title = p.get("title", "Unknown Title")
            raw_text = p.get("text", p.get("snippet", ""))
            
            if not raw_text:
                return p, "No content available to summarize."
                
            # Increased context window for richer summaries
            context_text = raw_text[:12000]
            
            for attempt in range(3):
                try:
                    # Pace requests slightly
                    time.sleep(2)
                    result = chain.invoke({
                        "title": title,
                        "text": context_text
                    })
                    return p, result.content.strip()
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if attempt < 2:
                            print(f"Rate limited on '{title}'. Sleeping 60s to respect free tier quota...")
                            time.sleep(60)
                            continue
                    print(f"Summary generation failed for '{title}': {e}")
                    return p, raw_text[:280] + "..." if len(raw_text) > 280 else raw_text

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_paper = {executor.submit(process_paper, p): p for p in papers}
            for future in as_completed(future_to_paper):
                p, summary_text = future.result()
                p["summary"] = summary_text

        print("Summaries generated successfully.")
        return {"ranked_papers": papers}
