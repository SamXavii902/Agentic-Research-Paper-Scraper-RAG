import json
import re
from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


class SidebarOps:
    """
    Backend logic for the 5 sidebar operation buttons.
    Each method operates on already-loaded papers (no re-scraping).
    """

    def __init__(self):
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, request_timeout=45, max_retries=6)

    # ──────────────────────────────────────────
    # 1. SEMANTIC QUERY — search the RAG vector store
    # ──────────────────────────────────────────
    def semantic_query(self, query: str, rag_store) -> str:
        """Search the vector store and return top matching chunks with paper context."""
        if not rag_store:
            return "<span style='color:var(--accent-magenta)'>No vector store available. Run a search first.</span>"

        try:
            results = rag_store.similarity_search(query, k=5)
            if not results:
                return f"No relevant chunks found for query: <b>{query}</b>"

            html_parts = [
                f"<b style='color:var(--accent-cyan)'>Semantic Search Results</b> for: <i>\"{query}\"</i><br><br>"
            ]
            for i, doc in enumerate(results):
                source = doc.metadata.get("source", "Unknown")
                title = doc.metadata.get("title", source)
                # Trim content to a reasonable display length
                content = doc.page_content[:400].strip()
                if len(doc.page_content) > 400:
                    content += "..."

                html_parts.append(
                    f"<div style='margin-bottom:12px;padding:8px;border-left:2px solid var(--accent-cyan);'>"
                    f"<b style='color:var(--accent-cyan)'>[{i+1}]</b> "
                    f"<span style='color:var(--text-sub)'>{title}</span><br>"
                    f"<span style='font-size:11px'>{content}</span>"
                    f"</div>"
                )

            return "".join(html_parts)

        except Exception as e:
            return f"<span style='color:red'>Semantic search failed: {e}</span>"

    # ──────────────────────────────────────────
    # 2. CONTRADICTION ENGINE — find conflicts from edge data
    # ──────────────────────────────────────────
    def find_contradictions(
        self, graph_data: Dict[str, Any], papers: List[Dict[str, Any]]
    ) -> str:
        """Extract contradiction edges from graph data and present them."""
        links = graph_data.get("links", [])
        nodes = graph_data.get("nodes", [])

        contradictions = [l for l in links if l.get("relationship") == "contradiction"]

        if not contradictions:
            # Also check for weak edges that might hint at methodological differences
            weak_with_diffs = [
                l for l in links
                if l.get("relationship") == "weak" and l.get("differences")
            ]
            if not weak_with_diffs:
                return (
                    "<b style='color:var(--accent-cyan)'>Contradiction Engine</b><br><br>"
                    "No contradictions detected in the current corpus. "
                    "All papers appear to be in methodological agreement."
                )
            contradictions = weak_with_diffs[:3]

        html_parts = [
            f"<b style='color:var(--accent-magenta)'>⚡ Contradiction Engine</b><br>"
            f"<span style='color:var(--text-sub)'>Found {len(contradictions)} conflict(s)</span><br><br>"
        ]

        for idx, edge in enumerate(contradictions):
            src_id = edge.get("source", 0)
            tgt_id = edge.get("target", 0)

            src_title = nodes[src_id]["title"] if src_id < len(nodes) else "Paper A"
            tgt_title = nodes[tgt_id]["title"] if tgt_id < len(nodes) else "Paper B"

            rel_type = edge.get("relationship", "contradiction")
            sims = edge.get("similarities", [])
            diffs = edge.get("differences", [])

            html_parts.append(
                f"<div style='margin-bottom:14px;padding:8px 10px;border-left:2px solid var(--accent-magenta);'>"
                f"<b style='color:var(--accent-magenta)'>Conflict {idx+1}</b><br>"
                f"<span style='color:var(--accent-cyan);font-size:11px'>{src_title[:60]}...</span><br>"
                f"<span style='color:var(--text-sub);font-size:10px'>vs.</span><br>"
                f"<span style='color:var(--accent-cyan);font-size:11px'>{tgt_title[:60]}...</span><br><br>"
            )

            if diffs:
                html_parts.append("<span style='color:var(--accent-magenta);font-size:10px'>DIFFERENCES:</span><br>")
                for d in diffs[:3]:
                    html_parts.append(f"<span style='font-size:11px'>• {d}</span><br>")

            if sims:
                html_parts.append("<br><span style='color:var(--accent-cyan);font-size:10px'>SIMILARITIES:</span><br>")
                for s in sims[:2]:
                    html_parts.append(f"<span style='font-size:11px'>• {s}</span><br>")

            html_parts.append("</div>")

        return "".join(html_parts)

    # ──────────────────────────────────────────
    # 3. SYNTHESIZE LITERATURE — LLM-generated structured review
    # ──────────────────────────────────────────
    def synthesize_review(self, papers: List[Dict[str, Any]], query: str) -> str:
        """Generate a concise structured literature review from paper data."""
        if not papers:
            return "<span style='color:var(--accent-magenta)'>No papers loaded.</span>"

        context_lines = []
        for i, p in enumerate(papers[:10]):
            title = p.get("title", f"Paper {i}")
            summary = p.get("summary", "")[:300]
            year = p.get("year", "N/A")
            citations = p.get("citations", 0)
            context_lines.append(
                f"[{i+1}] \"{title}\" ({year}, {citations} cites)\n  {summary}"
            )

        prompt = PromptTemplate.from_template(
            "You are a research assistant. Given these papers on '{query}', "
            "write a structured 3-paragraph literature review:\n\n"
            "Paragraph 1: Overview of the research landscape and key themes.\n"
            "Paragraph 2: Methodological approaches and their strengths/weaknesses.\n"
            "Paragraph 3: Research gaps and future directions.\n\n"
            "Papers:\n{papers}\n\n"
            "Write the review in HTML format with <b> for key terms and <br> for line breaks. "
            "Keep it under 300 words. Be specific — reference actual paper findings."
        )

        try:
            chain = prompt | self.llm
            result = chain.invoke({"query": query, "papers": "\n".join(context_lines)})
            return (
                f"<b style='color:var(--accent-cyan)'>📄 Literature Synthesis</b><br>"
                f"<span style='color:var(--text-sub)'>Topic: {query}</span><br><br>"
                f"{result.content}"
            )
        except Exception as e:
            return f"<span style='color:red'>Synthesis failed: {e}</span>"

    # ──────────────────────────────────────────
    # 4. PREDICT TRAJECTORIES — analyze citation trends
    # ──────────────────────────────────────────
    def predict_trajectories(self, papers: List[Dict[str, Any]], query: str) -> str:
        """Analyze publication years, citation counts, and methods to predict research trends."""
        if not papers:
            return "<span style='color:var(--accent-magenta)'>No papers loaded.</span>"

        # Build a rich timeline context
        timeline_data = []
        for p in sorted(papers[:10], key=lambda x: x.get("year", 0)):
            title = p.get("title", "Unknown")[:60]
            year = p.get("year", "N/A")
            citations = p.get("citations", 0)
            timeline_data.append(f"  {year} | {citations:>5} cites | {title}")

        prompt = PromptTemplate.from_template(
            "You are a research trend analyst. Given these papers sorted chronologically "
            "on the topic '{query}':\n\n"
            "Year | Citations | Title\n{timeline}\n\n"
            "Analyze the trajectory of this research field:\n"
            "1. Which methods/approaches are gaining traction? (cite specific papers)\n"
            "2. What is the next likely breakthrough direction? (be specific)\n"
            "3. Estimate a probability (%) for your top prediction.\n"
            "4. Which papers are outliers (high citations for their age)?\n\n"
            "Format as HTML with <b> tags for key terms. Be specific and data-driven."
        )

        try:
            chain = prompt | self.llm
            result = chain.invoke({
                "query": query,
                "timeline": "\n".join(timeline_data)
            })
            return (
                f"<b style='color:var(--accent-cyan)'>🕐 Research Trajectory Analysis</b><br>"
                f"<span style='color:var(--text-sub)'>Analyzing {len(papers)} papers</span><br><br>"
                f"{result.content}"
            )
        except Exception as e:
            return f"<span style='color:red'>Trajectory prediction failed: {e}</span>"

    # ──────────────────────────────────────────
    # 5. FIND BRIDGES — graph topology analysis
    # ──────────────────────────────────────────
    def find_bridges(
        self, graph_data: Dict[str, Any], papers: List[Dict[str, Any]]
    ) -> str:
        """
        Find bridge papers that connect otherwise disconnected clusters.
        Uses pure graph topology — no LLM call needed.
        """
        nodes = graph_data.get("nodes", [])
        links = graph_data.get("links", [])

        if not nodes or not links:
            return "<span style='color:var(--accent-magenta)'>No graph data available.</span>"

        n = len(nodes)

        # Build adjacency list
        adj: Dict[int, set] = {i: set() for i in range(n)}
        edge_meta: Dict[tuple, Dict] = {}
        for link in links:
            s, t = link.get("source", 0), link.get("target", 0)
            if s < n and t < n:
                adj[s].add(t)
                adj[t].add(s)
                edge_meta[(min(s, t), max(s, t))] = link

        # Compute degree centrality and betweenness (approximate)
        # A bridge node connects groups that wouldn't be connected otherwise
        degree = {i: len(adj[i]) for i in range(n)}

        # Find nodes whose removal would increase connected components
        bridge_nodes = []
        for node_id in range(n):
            if degree[node_id] < 2:
                continue
            # Remove node and check connectivity
            remaining = set(range(n)) - {node_id}
            if not remaining:
                continue
            visited = set()
            stack = [next(iter(remaining))]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                for neighbor in adj[curr]:
                    if neighbor != node_id and neighbor not in visited:
                        stack.append(neighbor)

            disconnected = remaining - visited
            if disconnected:
                bridge_nodes.append({
                    "id": node_id,
                    "title": nodes[node_id]["title"] if node_id < len(nodes) else f"Paper {node_id}",
                    "degree": degree[node_id],
                    "clusters_connected": len(disconnected),
                    "connected_to": [
                        nodes[nb]["title"][:40] if nb < len(nodes) else f"Paper {nb}"
                        for nb in adj[node_id]
                    ]
                })

        # Also find high-degree hub nodes (connect many papers)
        if not bridge_nodes:
            # Fallback: show hub nodes with highest degree
            hubs = sorted(range(n), key=lambda i: degree[i], reverse=True)[:3]
            bridge_nodes = [
                {
                    "id": h,
                    "title": nodes[h]["title"] if h < len(nodes) else f"Paper {h}",
                    "degree": degree[h],
                    "clusters_connected": 0,
                    "connected_to": [
                        nodes[nb]["title"][:40] if nb < len(nodes) else f"Paper {nb}"
                        for nb in adj[h]
                    ]
                }
                for h in hubs if degree[h] > 0
            ]

        # Format output
        html_parts = [
            f"<b style='color:var(--accent-cyan)'>🔗 Bridge & Hub Analysis</b><br>"
            f"<span style='color:var(--text-sub)'>{len(nodes)} nodes, {len(links)} edges analyzed</span><br><br>"
        ]

        for b in bridge_nodes[:5]:
            is_bridge = b["clusters_connected"] > 0
            label = "BRIDGE" if is_bridge else "HUB"
            color = "var(--accent-magenta)" if is_bridge else "var(--accent-cyan)"

            html_parts.append(
                f"<div style='margin-bottom:12px;padding:8px 10px;border-left:2px solid {color};'>"
                f"<b style='color:{color}'>[{label}]</b> "
                f"<span style='font-size:11px'>{b['title'][:70]}</span><br>"
                f"<span style='color:var(--text-sub);font-size:10px'>"
                f"Connections: {b['degree']}"
            )
            if is_bridge:
                html_parts.append(f" | Bridges {b['clusters_connected']} disconnected node(s)")
            html_parts.append("</span><br>")

            if b["connected_to"]:
                html_parts.append(
                    "<span style='font-size:10px;color:var(--text-sub)'>Links to:</span><br>"
                )
                for ct in b["connected_to"][:4]:
                    html_parts.append(f"<span style='font-size:10px'>  → {ct}</span><br>")

            html_parts.append("</div>")

        return "".join(html_parts)

    # ──────────────────────────────────────────
    # 6. CHAT WITH PAPERS (NotebookLM / Gemini style)
    # ──────────────────────────────────────────
    def chat_with_papers(self, papers: List[Dict[str, Any]], query: str, rag_store) -> str:
        """Answer an arbitrary user query based on the loaded papers."""
        if not papers:
            return "<span style='color:var(--accent-magenta)'>No papers loaded to chat with. Please run a search first.</span>"

        # Try to get highly specific context from RAG store first
        rag_context = ""
        if rag_store:
            try:
                results = rag_store.similarity_search(query, k=4)
                for doc in results:
                    rag_context += f"Source: {doc.metadata.get('title', 'Unknown')}\nSnippet: {doc.page_content}\n\n"
            except Exception as e:
                print(f"RAG search failed during chat: {e}")

        # If RAG is empty or not available, use the paper summaries
        if not rag_context.strip():
            for i, p in enumerate(papers[:15]):
                title = p.get("title", f"Paper {i}")
                summary = p.get("summary", "")[:500]
                rag_context += f"Source: {title}\nSummary: {summary}\n\n"

        prompt = PromptTemplate.from_template(
            "You are an expert research assistant. You are chatting with a user about a set of retrieved academic papers.\n"
            "Below is the context extracted from the papers. Use ONLY this context to answer the user's question.\n"
            "If the answer cannot be determined from the context, state that clearly.\n\n"
            "Context:\n{context}\n\n"
            "User Question: {query}\n\n"
            "Format your answer beautifully in HTML using <b> for bold, <i> for italics, and bullet points (<ul><li>) where appropriate. "
            "Always cite your sources by mentioning the paper title when you state a fact."
        )

        try:
            chain = prompt | self.llm
            result = chain.invoke({"query": query, "context": rag_context})
            return result.content
        except Exception as e:
            return f"<span style='color:red'>Chat failed: {e}</span>"

    async def async_chat_stream(self, papers: List[Dict[str, Any]], query: str, rag_store, action: str):
        """Asynchronously stream a chat response, specialized based on the action mode."""
        import asyncio
        loop = asyncio.get_running_loop()
        
        print(f"[Chat Stream] Started. Action: {action}, Query: {query}")
        
        if not papers:
            yield "<span style='color:var(--accent-magenta)'>No papers loaded to chat with. Please run a search first.</span>"
            return

        rag_context = ""

        # MODE: CITE (@ REF)
        if action == "ref":
            print("[Chat Stream] CITE mode")
            # Extract [Paper Title] from the query
            titles = re.findall(r"\[([^\]]+)\]", query)
            if titles:
                for title in titles:
                    # Find matching papers
                    matched = next((p for p in papers if p.get("title") == title), None)
                    if matched:
                        summary = matched.get("summary", "")
                        # Try to get first few paragraphs of actual text
                        text = matched.get("text", "")[:3000]
                        rag_context += f"Source: {title}\nSummary: {summary}\nExcerpt: {text}\n\n"
            
            if not rag_context.strip():
                rag_context = "No specific references found in context. Please provide a general answer."
                
            prompt = PromptTemplate.from_template(
                "You are an expert research assistant. The user is asking about specific cited papers.\n"
                "Context of cited papers:\n{context}\n\n"
                "User Question: {query}\n\n"
                "Format your answer beautifully using HTML tags like <b>, <i>, and <ul><li>. Focus ONLY on the referenced papers.\n"
                "CRITICAL: Return ONLY the HTML snippet. Do NOT wrap your answer in <html>, <body>, or <head> tags, and do NOT include any <style> tags.\n"
                "AT THE END, generate 3 follow-up questions the user could ask to explore further. Format them EXACTLY as a single HTML <ul> list with the class 'follow-up-questions' and each question as an <li>."
            )

        # MODE: SEARCH
        elif action == "search":
            print("[Chat Stream] SEARCH mode")
            if rag_store:
                try:
                    # Deep retrieval - MUST run in executor to avoid blocking event loop with local embeddings
                    print("[Chat Stream] Running similarity search...")
                    results = await loop.run_in_executor(None, rag_store.similarity_search, query, 8)
                    for doc in results:
                        rag_context += f"Source: {doc.metadata.get('title', 'Unknown')}\nSnippet: {doc.page_content}\n\n"
                except Exception as e:
                    print(f"[Chat Stream] RAG Error: {e}")
            
            if not rag_context.strip():
                for p in papers[:10]:
                    rag_context += f"Source: {p.get('title', 'Unknown')}\nSummary: {p.get('summary', '')[:500]}\n\n"
                    
            prompt = PromptTemplate.from_template(
                "You are an expert research assistant conducting a deep search across the literature.\n"
                "Context:\n{context}\n\n"
                "User Question: {query}\n\n"
                "Provide a comprehensive, synthesized answer formatted with HTML tags (<b>, <i>, <ul><li>). Cite multiple sources where possible.\n"
                "CRITICAL: Return ONLY the HTML snippet. Do NOT wrap your answer in <html>, <body>, or <head> tags, and do NOT include any <style> tags.\n"
                "AT THE END, generate 3 follow-up questions the user could ask to explore further. Format them EXACTLY as a single HTML <ul> list with the class 'follow-up-questions' and each question as an <li>."
            )

        # MODE: THINK (Chain of Thought)
        elif action == "think":
            print("[Chat Stream] THINK mode")
            if rag_store:
                try:
                    results = await loop.run_in_executor(None, rag_store.similarity_search, query, 5)
                    for doc in results:
                        rag_context += f"Source: {doc.metadata.get('title', 'Unknown')}\nSnippet: {doc.page_content}\n\n"
                except Exception as e:
                    print(f"[Chat Stream] RAG Error: {e}")
            
            if not rag_context.strip():
                for p in papers[:5]:
                    rag_context += f"Source: {p.get('title', 'Unknown')}\nSummary: {p.get('summary', '')[:800]}\n\n"
                    
            prompt = PromptTemplate.from_template(
                "You are a brilliant academic critic thinking deeply about a problem.\n"
                "Context:\n{context}\n\n"
                "User Question: {query}\n\n"
                "Format your answer using HTML tags. First, provide a <div class='thought-process'> section using <i> italics to explain your deep reasoning step-by-step, "
                "then provide the final 'Conclusion' in a separate paragraph.\n"
                "CRITICAL: Return ONLY the HTML snippet. Do NOT wrap your answer in <html>, <body>, or <head> tags, and do NOT include any <style> tags.\n"
                "AT THE END, generate 3 follow-up questions the user could ask to explore further. Format them EXACTLY as a single HTML <ul> list with the class 'follow-up-questions' and each question as an <li>."
            )

        # MODE: NORMAL (null or other)
        else:
            print("[Chat Stream] NORMAL mode")
            if rag_store:
                try:
                    results = await loop.run_in_executor(None, rag_store.similarity_search, query, 4)
                    for doc in results:
                        rag_context += f"Source: {doc.metadata.get('title', 'Unknown')}\nSnippet: {doc.page_content}\n\n"
                except Exception as e:
                    print(f"[Chat Stream] RAG Error: {e}")
                    
            if not rag_context.strip():
                for p in papers[:5]:
                    rag_context += f"Source: {p.get('title', 'Unknown')}\nSummary: {p.get('summary', '')[:400]}\n\n"
                    
            prompt = PromptTemplate.from_template(
                "You are a helpful research assistant. Answer the user based on the context.\n"
                "Context:\n{context}\n\n"
                "User Question: {query}\n\n"
                "Format beautifully using HTML tags. Cite sources by title.\n"
                "CRITICAL: Return ONLY the HTML snippet. Do NOT wrap your answer in <html>, <body>, or <head> tags, and do NOT include any <style> tags.\n"
                "AT THE END, generate 3 follow-up questions the user could ask to explore further. Format them EXACTLY as a single HTML <ul> list with the class 'follow-up-questions' and each question as an <li>."
            )

        print(f"[Chat Stream] Context prepared, length: {len(rag_context)}")
        
        # Stream the response
        chain = prompt | self.llm
        try:
            print("[Chat Stream] Initiating LLM astream...")
            # astream returns an async generator over chunks
            async for chunk in chain.astream({"query": query, "context": rag_context}):
                if chunk.content:
                    yield chunk.content
            print("[Chat Stream] Stream completed.")
        except Exception as e:
            print(f"[Chat Stream] LLM Stream Error: {e}")
            yield f"<span style='color:red'>Stream failed: {e}</span>"
