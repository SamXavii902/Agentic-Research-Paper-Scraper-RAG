from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any

class CriticAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, max_retries=6)
        
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        section = state.get("current_section")
        draft = state.get("draft_sections", {}).get(section, "")
        iteration = state.get("iteration_count", 0)
        print(f"\n--- Critic Agent (Iteration {iteration}) ---")
        
        # Guard against infinite loops
        if iteration >= 2:
            print("Max iterations reached. Forcing pass.")
            return {"critique": {"passed": True, "feedback": "Max iterations reached."}, "iteration_count": iteration + 1}
            
        prompt = PromptTemplate.from_template(
            "You are a strict peer reviewer. Analyze this draft section for a literature review.\n"
            "Criteria:\n"
            "1. It MUST contain actual citations in the format [Source: <filename>].\n"
            "2. It must logically synthesize the text without sounding like cheap AI filler.\n"
            "If it fails any criteria, explain exactly what needs fixing. If it passes, reply exactly with the word 'PASS'.\n\n"
            "Draft:\n{draft}"
        )
        
        chain = prompt | self.llm
        result = chain.invoke({"draft": draft})
        feedback = result.content.strip()
        
        passed = feedback == "PASS"
        if passed:
            print("Draft passed critique!")
        else:
            print(f"Critique Feedback: {feedback[:100]}...")
            
        return {"critique": {"passed": passed, "feedback": feedback}, "iteration_count": iteration + 1}
