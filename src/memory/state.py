from typing import TypedDict, List, Dict, Any, Optional

class ResearchState(TypedDict):
    """
    The shared state passed between all agents in the LangGraph workflow.
    """
    query: str
    papers: List[Dict[str, Any]]        # Raw scraped metadata + pdf paths
    ranked_papers: List[Dict[str, Any]] # Papers sorted by Quality Scoring Agent
    chunks: List[Any]                   # RAG chunks from the parsed PDFs
    outline: List[str]                  # Generated section titles
    current_section: str                # The section currently being processed
    draft_sections: Dict[str, str]      # section_title -> drafted text
    critique: Dict[str, Any]            # Feedback from the Critic Agent
    iteration_count: int                # Prevents infinite revision loops
    cached: bool                        # True if RAG cache was hit
    local_only: Optional[bool]          # True if UI explicitly requested local files only
    graph_data: Dict[str, Any]          # JSON payload containing nodes and links for D3
    hypotheses: List[Dict[str, Any]]    # Generated research hypotheses from HypothesisAgent
