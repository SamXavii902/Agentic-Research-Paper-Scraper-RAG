from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any

class RevisionAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_retries=6)
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n--- Revision Agent ---")
        section = state.get("current_section")
        draft = state.get("draft_sections", {}).get(section, "")
        critique = state.get("critique", {}).get("feedback", "")
        
        prompt = PromptTemplate.from_template(
            "You are an academic revision agent. Rewrite the following draft to strictly fix the issues identified in the critique.\n"
            "Maintain a formal academic tone and ensure citations [Source: <filename>] are preserved or added where missing.\n\n"
            "Critique Feedback:\n{critique}\n\n"
            "Original Draft:\n{draft}\n\n"
            "Revised Draft:"
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"draft": draft, "critique": critique})
        
        drafts = state.get("draft_sections", {})
        drafts[section] = result.content
        return {"draft_sections": drafts}
