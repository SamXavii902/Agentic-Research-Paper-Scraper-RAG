import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from src.rag.vector_store import RAGStore

# Load environment variables (API Key)
load_dotenv()

class LiteratureReviewAgent:
    def __init__(self):
        # Initialize OpenAI API
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment.")
            
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.2,
            api_key=api_key
        , max_retries=6)
        self.rag_store = RAGStore()

    def generate_outline(self, topic):
        """Generate a basic outline based on the topic."""
        print(f"Generating outline for: {topic}")
        prompt = PromptTemplate.from_template(
            "You are an expert academic researcher. Generate a concise 3-section outline for a literature review on the topic: '{topic}'. "
            "Return ONLY the section titles separated by newlines."
        )
        chain = prompt | self.llm
        result = chain.invoke({"topic": topic})
        
        # Parse outline
        outline = [line.strip().lstrip('-').lstrip('1234567890.').strip() for line in result.content.strip().split('\n') if line.strip()]
        return outline[:3] # Keep it short for testing

    def draft_section(self, section_title):
        """Draft a section using RAG."""
        print(f"Drafting section: {section_title}")
        
        # Query ChromaDB
        docs = self.rag_store.similarity_search(section_title, k=3)
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
            "section_title": section_title,
            "context": context if context else "No relevant literature found in the database."
        })
        
        return result.content

    def verify_citations(self, draft):
        """A simple verification step to check for hallucinations."""
        # For a full implementation, this would query the DB again to verify claims.
        # For now, we'll use an LLM-as-a-judge approach.
        prompt = PromptTemplate.from_template(
            "You are a strict proofreader. Read the following draft.\n"
            "If the draft contains citations in the format [Source: <filename>], you must return the EXACT draft text, word for word, with absolutely no modifications or commentary.\n"
            "If it does not contain proper citations, append this exact warning to the end of the draft: '\n\nWarning: Missing proper citations.' and return the draft with the warning appended.\n"
            "Do NOT just say 'The draft contains citations'. You must output the actual draft text.\n\n"
            "Draft:\n{draft}"
        )
        chain = prompt | self.llm
        result = chain.invoke({"draft": draft})
        return result.content

    def run_synthesis(self, topic, output_file="literature_review.md"):
        outline = self.generate_outline(topic)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Literature Review: {topic}\n\n")
            
            for section in outline:
                f.write(f"## {section}\n\n")
                draft = self.draft_section(section)
                verified_draft = self.verify_citations(draft)
                f.write(f"{verified_draft}\n\n")
                
        print(f"Literature review saved to {output_file}")

if __name__ == "__main__":
    # Test
    # agent = LiteratureReviewAgent()
    # agent.run_synthesis("Hybrid Cryptography and Steganography")
    pass
