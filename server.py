from dotenv import load_dotenv
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

from main import build_graph
from src.memory.cache_manager import initialize_cache
from src.agent.sidebar_operations import SidebarOps
from src.rag.vector_store import RAGStore

app = FastAPI()

# Enable LLM caching to save API costs
initialize_cache()
langgraph_app = build_graph()
sidebar_ops = SidebarOps()

# Cached state from the last pipeline run — sidebar operations use this
last_run_state = {
    "papers": [],
    "graph_data": {},
    "query": "",
    "rag_store": None,
}

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.get("/")
async def get():
    with open("knowledge_graph.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/api/search")
async def start_search(payload: dict):
    load_dotenv(override=True) # Fresh API keys
    query = payload.get("query", "")
    if not query:
        return {"status": "error", "message": "Query is empty"}
    
    local_only = payload.get("local_only", False)
    # We will trigger the background task to run the graph and stream to websocket
    asyncio.create_task(run_graph_and_stream(query, local_only))
    return {"status": "started"}

@app.get("/api/state")
async def get_state():
    # Return the cached graph data and query so frontend can restore instantly on refresh
    return {
        "status": "success",
        "query": last_run_state.get("query", ""),
        "graph_data": last_run_state.get("graph_data", {}),
        "hypotheses": last_run_state.get("hypotheses", [])
    }

@app.post("/api/operation")
async def run_operation(payload: dict):
    op_id = payload.get("operation", "")
    query = payload.get("query", "") or last_run_state.get("query", "")

    papers = last_run_state.get("papers", [])
    graph_data = last_run_state.get("graph_data", {})
    rag_store = last_run_state.get("rag_store")

    if not papers and op_id != "semantic":
        await manager.broadcast(json.dumps({
            "type": "operation_result",
            "operation": op_id,
            "html": "<span style='color:var(--accent-magenta)'>No papers loaded. Run a search first.</span>"
        }))
        return {"status": "no_data"}

    asyncio.create_task(run_sidebar_operation(op_id, query, papers, graph_data, rag_store))
    return {"status": "started"}

@app.post("/api/chat")
async def chat_endpoint(payload: dict):
    load_dotenv(override=True) # Fresh API keys
    query = payload.get("query", "")
    action = payload.get("action", None)
    
    if not query:
        return {"status": "error", "message": "Query is empty"}
    
    papers = last_run_state.get("papers", [])
    rag_store = last_run_state.get("rag_store")
    
    # Auto-restore RAG store from disk cache if server was restarted
    if not rag_store:
        try:
            rag_store = RAGStore()
            last_run_state["rag_store"] = rag_store
        except Exception as e:
            print(f"Failed to restore RAG store from cache: {e}")

    # Delegate entirely to the async generator in SidebarOps
    return StreamingResponse(
        sidebar_ops.async_chat_stream(papers, query, rag_store, action),
        media_type="text/plain"
    )

async def run_sidebar_operation(op_id, query, papers, graph_data, rag_store):
    """Run a sidebar operation in a background thread and broadcast results."""
    await manager.broadcast(json.dumps({
        "type": "log",
        "message": f"Running <b>{op_id}</b> operation..."
    }))

    try:
        loop = asyncio.get_running_loop()

        if op_id == "semantic":
            if not rag_store:
                try:
                    rag_store = RAGStore()
                except Exception:
                    pass
            result_html = await loop.run_in_executor(
                None, sidebar_ops.semantic_query, query, rag_store
            )
        elif op_id == "contradiction":
            result_html = await loop.run_in_executor(
                None, sidebar_ops.find_contradictions, graph_data, papers
            )
        elif op_id == "synthesize":
            result_html = await loop.run_in_executor(
                None, sidebar_ops.synthesize_review, papers, query
            )
        elif op_id == "predict":
            result_html = await loop.run_in_executor(
                None, sidebar_ops.predict_trajectories, papers, query
            )
        elif op_id == "bridges":
            result_html = await loop.run_in_executor(
                None, sidebar_ops.find_bridges, graph_data, papers
            )
        else:
            result_html = f"<span style='color:red'>Unknown operation: {op_id}</span>"

        await manager.broadcast(json.dumps({
            "type": "operation_result",
            "operation": op_id,
            "html": result_html
        }))
    except Exception as e:
        await manager.broadcast(json.dumps({
            "type": "operation_result",
            "operation": op_id,
            "html": f"<span style='color:red'>Operation failed: {e}</span>"
        }))

async def run_graph_and_stream(query: str, local_only: bool = False):
    await manager.broadcast(json.dumps({
        "type": "log",
        "message": f"Initializing autonomous research agent for query: '{query}'..." + (" (LOCAL ONLY)" if local_only else "")
    }))
    await manager.broadcast(json.dumps({"type": "rag_step", "step": 0})) # Query Expansion

    try:
        queue = asyncio.Queue()
        
        loop = asyncio.get_running_loop()
        
        initial_state = {
            "query": query,
            "papers": [],
            "ranked_papers": [],
            "chunks": [],
            "outline": [],
            "current_section": "",
            "draft_sections": {},
            "critique": {},
            "iteration_count": 0,
            "local_only": local_only,
            "graph_data": {},
            "hypotheses": []
        }
        
        def graph_runner(loop):
            try:
                for output in langgraph_app.stream(initial_state, {"recursion_limit": 80}):
                    node_name = list(output.keys())[0]
                    asyncio.run_coroutine_threadsafe(queue.put({"node": node_name, "state": output[node_name]}), loop)
                        
                asyncio.run_coroutine_threadsafe(queue.put({"node": "END"}), loop)
            except Exception as e:
                print("GRAPH RUNNER EXCEPTION:", e)
                import traceback
                traceback.print_exc()
                asyncio.run_coroutine_threadsafe(queue.put({"error": str(e)}), loop)
        
        # Launch the graph runner in a background thread WITHOUT awaiting it
        asyncio.create_task(asyncio.to_thread(graph_runner, loop))
        
        step_mapping = {
            "retrieve": 1,
            "score": 2,
            "summary": 2,
            "chunk": 3,
            "draft": 4,
            "critique": 4,
            "revise": 4,
            "advance": 4,
            "graph": 4,
            "hypothesize": 5
        }
        
        while True:
            msg = await queue.get()
            if "error" in msg:
                await manager.broadcast(json.dumps({"type": "log", "message": f"<span style='color:red'>Error: {msg['error']}</span>"}))
                break
            if msg.get("node") == "END":
                await manager.broadcast(json.dumps({"type": "log", "message": "Workflow complete!"}))
                await manager.broadcast(json.dumps({"type": "rag_step", "step": 5})) # Synthesis/Generation
                await manager.broadcast(json.dumps({"type": "operation_complete", "operation": "synthesize"}))
                break
                
            if "log" in msg:
                await manager.broadcast(json.dumps({"type": "log", "message": msg["log"]}))
                continue
                
            node = msg["node"]
            state = msg["state"]
            
            # Send step update
            if node in step_mapping:
                await manager.broadcast(json.dumps({"type": "rag_step", "step": step_mapping[node]}))
                
            # Log specifics depending on node
            if node == "retrieve":
                papers = state.get("papers", [])
                await manager.broadcast(json.dumps({"type": "log", "message": f"Retrieved {len(papers)} papers from arXiv and IEEE."}))
                await manager.broadcast(json.dumps({"type": "metric", "key": "Active Nodes", "value": len(papers)}))
            elif node == "score":
                ranked = state.get("ranked_papers", [])
                await manager.broadcast(json.dumps({"type": "log", "message": f"Scoring complete. Retained top {len(ranked)} papers."}))
            elif node == "chunk":
                chunks = state.get("chunks", [])
                await manager.broadcast(json.dumps({"type": "log", "message": f"Vectorized {len(chunks)} chunks into RAG memory."}))
            else:
                await manager.broadcast(json.dumps({"type": "console", "message": f"Executing <b>{node}</b>..."}))
            
            # If graph is generated, send graph data and cache state
            if node == "graph" and "graph_data" in state:
                last_run_state["graph_data"] = state["graph_data"]
                await manager.broadcast(json.dumps({
                    "type": "graph_data",
                    "data": state["graph_data"]
                }))
            
            # Cache papers and query from any node that carries them
            if "ranked_papers" in state and state["ranked_papers"]:
                last_run_state["papers"] = state["ranked_papers"]
                last_run_state["query"] = query
            
            # Cache RAG store reference from chunk node
            if node == "chunk":
                try:
                    last_run_state["rag_store"] = RAGStore()
                except Exception:
                    pass

            # If hypotheses are generated, send them to frontend and cache them
            if node == "hypothesize" and "hypotheses" in state:
                last_run_state["hypotheses"] = state["hypotheses"]
                await manager.broadcast(json.dumps({
                    "type": "hypotheses",
                    "data": state["hypotheses"]
                }))
                
    except Exception as e:
         await manager.broadcast(json.dumps({"type": "log", "message": f"Failed: {str(e)}"}))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # We can handle messages from client if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
