import asyncio
import os
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END

# Project imports
from src.scrapers.arxiv_scraper import ArxivScraper
from src.scrapers.ieee_scraper import IEEEScraper
from src.processors.pdf_parser import PDFParser
from src.rag.vector_store import RAGStore

# Memory & State
from src.memory.state import ResearchState
from src.memory.cache_manager import initialize_cache

# Agents
from src.agent.scoring_agent import ScoringAgent
from src.graphs.graph_builder import GraphAgent
from src.graphs.graph_builder import GraphAgent

load_dotenv()

def retrieval_node(state):
    print("\n--- Retrieval Node ---")
    keyword = state["query"]
    print(f"Starting retrieval pipeline for: '{keyword}'...")
    log_cb = print
        
    rag = RAGStore()
    
    local_only = state.get("local_only", False)

    # If UI requested local only, return all PDFs without keyword matching or scraping
    if local_only:
        print("Forced Local Mode: Bypassing all scrapers and using available local PDFs.")
        all_local_files = []
        for d in ["data/papers/ieee", "data/papers/arxiv"]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith(".pdf"):
                        all_local_files.append(os.path.join(d, f))
        
        valid_files = [p for p in all_local_files if os.path.exists(p) and os.path.getsize(p) > 1000]
        papers = [{"pdf_path": p, "title": os.path.basename(p).replace(".pdf", "")} for p in valid_files[:8]]
        print(f"Found {len(papers)} local PDFs to process.")
        return {"papers": papers, "cached": False}

    # RAG CACHE CHECK
    # Use relevance scores to ensure we only count highly relevant cached papers
    # A score of 0.7 or higher typically means strong semantic similarity
    try:
        results = rag.vector_store.similarity_search_with_relevance_scores(keyword, k=15)
        relevant_docs = [doc for doc, score in results if score >= 0.65]
    except Exception as e:
        # Fallback if relevance score metric fails (some embeddings don't support it well)
        results = rag.vector_store.similarity_search_with_score(keyword, k=15)
        # For L2 distance, lower is better. Assuming threshold < 0.8
        relevant_docs = [doc for doc, score in results if score < 0.8]
    
    unique_titles = list(set([doc.metadata.get("title", "Unknown Title") for doc in relevant_docs if doc.metadata.get("title")]))
    
    if len(unique_titles) >= 3:
        if log_cb:
            log_cb(f"<span style='color:var(--accent-emerald)'>RAG Cache Hit:</span> Found {len(unique_titles)} highly relevant pre-vectorized papers. Bypassing scrapers.")
        
        cached_papers = [{"title": t, "final_score": 0.95} for t in unique_titles]
        return {"papers": [], "ranked_papers": cached_papers, "cached": True}

    # LOCAL FILESYSTEM CACHE CHECK
    query_terms = [t.lower() for t in keyword.split() if t.lower() not in ('and', 'or', 'the', 'of', 'in', 'for', 'a', 'an', 'with', 'using')]
    local_matching_files = []
    for d in ["data/papers/ieee", "data/papers/arxiv"]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith(".pdf"):
                    matches = sum(1 for term in query_terms if term in f.lower())
                    # If at least half the significant keywords match, consider it relevant
                    if matches >= max(1, len(query_terms) // 2):
                        local_matching_files.append(os.path.join(d, f))
                        
    if len(local_matching_files) >= 3:
        if log_cb:
            log_cb(f"<span style='color:var(--accent-emerald)'>Local Cache Hit:</span> Found {len(local_matching_files)} matching PDFs locally. Bypassing scrapers.")
            
        papers = [{"pdf_path": p, "title": os.path.basename(p).replace(".pdf", "")} for p in local_matching_files[:10]]
        return {"papers": papers, "cached": False} # Not vectorized yet, so cached=False

    # arXiv scraper — primary source for open-access papers
    arxiv = ArxivScraper(log_callback=log_cb)
    arxiv_papers = arxiv.search_and_download(keyword, max_results=10)
    
    ieee = IEEEScraper(log_callback=print)
    ieee_papers = asyncio.run(ieee.search_and_download(keyword, max_results=10))
    
    # Both scrapers now return metadata-rich dicts, merge directly
    all_papers = arxiv_papers + ieee_papers
    print(f"Total papers acquired: {len(all_papers)}")
    
    return {"papers": all_papers, "cached": False}

def chunking_node(state):
    print("\n--- Chunking & Vectorization Node ---")
    papers = state["ranked_papers"]
    
    parser = PDFParser()
    rag = RAGStore()
    
    for p in papers:
        file_path = p.get("pdf_path", "")
        if not file_path:
            continue
        print(f"Processing {file_path}...")
        text = parser.parse_pdf(file_path)
        if text:
            # Store text back on paper dict so downstream agents have full text
            p["text"] = text
            # Extract real publication year from the PDF text
            from src.graphs.graph_builder import extract_publication_year_from_text, extract_clean_abstract_from_paper
            doc_year = extract_publication_year_from_text(text)
            if doc_year:
                p["year"] = doc_year
            doc_abstract = extract_clean_abstract_from_paper(p)
            if doc_abstract:
                p["abstract"] = doc_abstract
            rag.add_document(text, metadata={"source": os.path.basename(file_path), "title": p.get("title", "")})
            
    print("Vectorization complete.")
    
    # Hardcoded outline for simplicity, but could be generated by an OutlineAgent
    outline = ["Introduction", "Applications and Impact", "Challenges and Future Directions"]
    
    return {"outline": outline, "current_section": outline[0], "iteration_count": 0, "ranked_papers": papers}



def build_graph():
    workflow = StateGraph(ResearchState)
    
    # Initialize agents
    scoring = ScoringAgent()
    graph = GraphAgent()
    
    # Add nodes
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("score", scoring.process)
    workflow.add_node("chunk", chunking_node)
    workflow.add_node("graph", graph.process)
    
    # Define edges
    workflow.set_entry_point("retrieve")
    
    def should_score(state):
        if state.get("cached"):
            return "graph"
        return "score"
        
    workflow.add_conditional_edges(
        "retrieve",
        should_score,
        {
            "score": "score",
            "graph": "graph"
        }
    )
    workflow.add_edge("score", "chunk")
    workflow.add_edge("chunk", "graph")
    workflow.add_edge("graph", END)
    
    return workflow.compile()

def main():
    print("=== Autonomous LangGraph Research Agent ===")
    
    # Enable LLM caching to save API costs
    initialize_cache()
    
    app = build_graph()
    
    initial_state = {
        "query": "Land subsidence monitoring in Mine sites using INSAR images",
        "papers": [],
        "ranked_papers": [],
        "chunks": [],
        "outline": [],
        "current_section": "",
        "draft_sections": {},
        "critique": {},
        "iteration_count": 0,
        "cached": False,
        "graph_data": {},
        "hypotheses": []
    }
    
    # Run the graph
    print("\nStarting LangGraph workflow execution...")
    final_state = app.invoke(initial_state)
    
    # Save basic output file (no drafts generated)
    output_file = "final_review_langgraph.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Literature Graph Ready: {final_state['query']}\n\n")
        f.write("The knowledge graph and vector store have been successfully populated. The chat interface is now ready for use.")
            
    print(f"\nWorkflow complete! Final output saved to {output_file}")

if __name__ == "__main__":
    main()
