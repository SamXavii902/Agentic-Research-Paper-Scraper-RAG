import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.rag.vector_store import RAGStore
from typing import Dict, Any

class DraftingAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_retries=6)
        self.rag_store = RAGStore()
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        section = state.get("current_section")
        print(f"\n--- Drafting Agent: {section} ---")
        
        docs = self.rag_store.similarity_search(section, k=10)
        context = "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in docs])
        
        prompt = PromptTemplate.from_template(
            "You are an academic researcher writing a literature review section titled: '{section_title}'.\n"
            "Use the following retrieved text chunks to write a 2-paragraph synthesis.\n"
            "You MUST cite your sources directly in the text exactly as they appear in the Context metadata (e.g., [Source: paper_name.pdf]). Do not just write '[Source]'.\n\n"
            "Context:\n{context}\n\n"
            "Draft:"
        )
        chain = prompt | self.llm
        result = chain.invoke({
            "section_title": section,
            "context": context if context else "No relevant literature found in the database."
        })
        
        drafts = state.get("draft_sections", {})
        drafts[section] = result.content
        return {"draft_sections": drafts}
