from main import build_graph

def test():
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
        "local_only": True,
        "log_callback": print
    }
    
    print("Testing graph compilation and streaming with local_only=True...")
    try:
        for output in app.stream(initial_state):
            node_name = list(output.keys())[0]
            print("Completed Node Output:", node_name)
            if node_name == "graph":
                graph_data = output[node_name].get("graph_data", {})
                print(f"Graph nodes: {len(graph_data.get('nodes', []))}, links: {len(graph_data.get('links', []))}")
                for node in graph_data.get("nodes", [])[:3]:
                    print(f"  Node: '{node['title'][:35]}' | Year: {node.get('year')} | Abstract: '{node.get('abstract', '')[:60]}...'")
    except Exception as e:
        print("Exception:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
