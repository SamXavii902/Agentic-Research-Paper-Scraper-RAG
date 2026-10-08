import json
import re
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


def extract_publication_year_from_text(text: str) -> Optional[int]:
    if not text:
        return None

    search_area = text[:8000]
    patterns = [
        r'(?i)\b(?:published|publication year|publication|pub\. year|year)\s*(?:[:\-]|\s+in\s+|\s+on\s+)?\s*(20(?:0[0-9]|1[0-9]|2[0-5]))\b',
        r'(?i)\b(?:received|accepted|copyright|©)\s*(?:[:\-]|\s+in\s+|\s+on\s+)?\s*(20(?:0[0-9]|1[0-9]|2[0-5]))\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, search_area)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue

    years_found = re.findall(r"\b(20(?:0[0-9]|1[0-9]|2[0-5]))\b", search_area)
    if years_found:
        from collections import Counter
        return int(Counter(years_found).most_common(1)[0][0])

    return None


def extract_clean_abstract_from_paper(p: Dict[str, Any]) -> str:
    """
    Extract a clean, concise, coherent abstract about the specific paper,
    avoiding header metadata, publisher notices, authors, emails, and garbage text.
    """
    # 1. Scraper / Metadata abstract
    existing_abstract = p.get("abstract")
    if existing_abstract and len(existing_abstract.strip()) >= 50:
        cleaned = " ".join(existing_abstract.split()).strip()
        if len(cleaned) > 500:
            cleaned = cleaned[:500].rsplit(" ", 1)[0] + "..."
        return cleaned

    # 2. LLM summary if generated
    summary = p.get("summary")
    if summary and len(summary.strip()) >= 50:
        cleaned = " ".join(summary.split()).strip()
        if len(cleaned) > 500:
            cleaned = cleaned[:500].rsplit(" ", 1)[0] + "..."
        return cleaned

    # 3. Extract directly from PDF full text
    raw_text = p.get("text", "") or p.get("snippet", "")
    if not raw_text:
        return f"Full text indexed for '{p.get('title', 'this paper')}'. Abstract extracted by pipeline."

    prefix = raw_text[:9000]

    # Pattern A: Standard academic paper format: Abstract / Summary / Background / Overview
    patterns = [
        r'(?i)\b(?:abstract|summary)\b[\s:—–\-.\n]+(.*?)(?=(?:\b(?:statement of need|index\s+terms|keywords|key\s+words|1[\.\s]+introduction|i[\.\s]+introduction|introduction)\b|\n\s*\n\s*\n))',
        r'(?i)\b(?:background|overview)\b[\s:—–\-.\n]+(.*?)(?=(?:\b(?:statement of need|index\s+terms|keywords|1[\.\s]+introduction|i[\.\s]+introduction|introduction)\b|\n\s*\n\s*\n))',
        r'(?i)\babstract\b[\s:—–\-.\n]+([A-Z0-9][^\n]+(?:\n[^\n]+){1,8})',
        r'(?i)\bintroduction\b[\s:—–\-.\n]+([A-Z][^\n]+(?:\n[^\n]+){1,6})'
    ]

    for pat in patterns:
        match = re.search(pat, prefix, re.DOTALL)
        if match:
            candidate = " ".join(match.group(1).split()).strip()
            candidate = re.sub(r'^(?:—|-|:|\.)\s*', '', candidate)
            if len(candidate) >= 80 and not re.search(r'(?i)^(?:department|university|laboratory|copyright|author)', candidate[:40]):
                if len(candidate) > 500:
                    candidate = candidate[:500].rsplit(" ", 1)[0] + "..."
                return candidate

    # Pattern B: Heuristic fallback — locate first real paragraph with narrative prose
    paragraphs = [para.strip() for para in raw_text.split('\n\n') if len(para.strip()) > 90]
    for para in paragraphs[:8]:
        cleaned_para = " ".join(para.split()).strip()
        if re.search(r'(?i)(?:university|department|laboratory|ieee|copyright|all rights reserved|@|\.edu|\.org|vol\.|no\.|doi:|license)', cleaned_para[:80]):
            continue
        if re.search(r'[A-Z].*\.\s+[A-Z]', cleaned_para):
            if len(cleaned_para) > 500:
                cleaned_para = cleaned_para[:500].rsplit(" ", 1)[0] + "..."
            return cleaned_para

    title = p.get("title", "this study")
    return f"This paper examines {title}, presenting core methodology, empirical analysis, and domain findings."


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>The Agentic Nexus</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #fafafc;
            --dot-color: #d1d1d1;
            --text-main: #111111;
            --text-sub: #555555;
            --card-bg: rgba(255, 255, 255, 0.75);
            --card-border: rgba(220, 220, 235, 0.9);
            --card-shadow: 0 8px 32px rgba(0,0,0,0.06);
            --node-title: #1a1a1a;
            --edge-color: #b0b0c0;
            --edge-highlight: #ff0071;
            --glow-pink: rgba(255,0,113,0.10);
            --glow-purple: rgba(138,43,226,0.08);
            --sim-accent: #0071ff;
            --diff-accent: #ff6b35;
        }
        body.dark-theme {
            --bg-color: #0a0a0e;
            --dot-color: #222230;
            --text-main: #ececf0;
            --text-sub: #8888a0;
            --card-bg: rgba(18, 18, 28, 0.85);
            --card-border: rgba(255,255,255,0.07);
            --card-shadow: 0 8px 40px rgba(0,0,0,0.5);
            --node-title: #f0f0f8;
            --edge-color: #2a2a44;
            --glow-pink: rgba(255,0,113,0.06);
            --glow-purple: rgba(138,43,226,0.05);
        }
        *, *::before, *::after { box-sizing: border-box; }
        body {
            background-color: var(--bg-color);
            background-image: radial-gradient(var(--dot-color) 1px, transparent 1px);
            background-size: 22px 22px;
            margin: 0; overflow: hidden;
            font-family: 'Inter', sans-serif;
            transition: background-color 0.4s, background-image 0.4s;
        }
        .glow-blob {
            position: absolute; border-radius: 50%;
            width: 700px; height: 700px;
            pointer-events: none; z-index: 0;
            filter: blur(80px);
            transition: background 0.4s;
        }
        #glow1 { background: radial-gradient(circle, var(--glow-pink) 0%, transparent 70%); top: 5%; left: 10%; }
        #glow2 { background: radial-gradient(circle, var(--glow-purple) 0%, transparent 70%); bottom: 5%; right: 10%; }

        /* ======= SVG LAYER ======= */
        svg#edges-svg {
            position: absolute; top: 0; left: 0;
            width: 100vw; height: 100vh;
            z-index: 1; overflow: visible;
        }
        .edge-visible {
            fill: none;
            stroke: var(--edge-color);
            stroke-linecap: round;
            transition: stroke 0.25s, stroke-width 0.25s;
        }
        .edge-active {
            stroke: var(--edge-highlight) !important;
        }
        .edge-interaction {
            fill: none; stroke: transparent; stroke-width: 28px; cursor: pointer;
        }

        /* ======= NODE CARDS ======= */
        #nodes-layer {
            position: absolute; top: 0; left: 0;
            width: 100vw; height: 100vh;
            pointer-events: none; z-index: 2;
        }
        .node-card {
            position: absolute;
            background: var(--card-bg);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            box-shadow: var(--card-shadow);
            padding: 18px 20px;
            cursor: grab;
            display: flex; flex-direction: column; gap: 8px;
            pointer-events: auto;
            transform: translate(-50%, -50%);
            transition: border-color 0.2s, box-shadow 0.2s, background 0.4s;
            width: 270px;
            user-select: none;
        }
        .node-card:hover { box-shadow: 0 12px 48px rgba(0,0,0,0.12); }
        .node-card:active { cursor: grabbing; }
        .node-card.apex {
            border: 1.5px solid #ff0071;
            box-shadow: 0 0 32px rgba(255,0,113,0.18), var(--card-shadow);
        }
        .n-badge {
            display: inline-block;
            padding: 3px 10px; border-radius: 6px;
            font-size: 10px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.6px;
            width: fit-content;
        }
        .badge-red   { background: rgba(255,0,113,0.10); color: #ff0071; }
        .badge-blue  { background: rgba(0,113,255,0.10); color: #0071ff; }
        .n-title {
            font-size: 14px; font-weight: 600; line-height: 1.45;
            color: var(--node-title);
            transition: color 0.4s;
        }
        .n-score {
            font-size: 12px; font-weight: 500; color: var(--text-sub);
            transition: color 0.4s;
        }

        /* ======= DUAL TOOLTIP CARDS ======= */
        #tooltip-root {
            position: fixed; top: 0; left: 0;
            pointer-events: none; z-index: 200;
        }
        .t-card {
            position: absolute;
            background: var(--card-bg);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            box-shadow: var(--card-shadow);
            padding: 16px 18px;
            width: 260px;
            opacity: 0;
            transform-origin: top center;
            animation: dropIn 0.28s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        }
        @keyframes dropIn {
            0%   { opacity: 0; transform: translateY(-18px) scale(0.88); }
            100% { opacity: 1; transform: translateY(0)     scale(1.00); }
        }
        .t-card.hiding {
            animation: dropOut 0.18s ease-in forwards;
        }
        @keyframes dropOut {
            0%   { opacity: 1; transform: translateY(0)     scale(1.00); }
            100% { opacity: 0; transform: translateY(-10px) scale(0.92); }
        }
        .t-card-title {
            font-size: 11px; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.5px;
            margin-bottom: 10px;
            display: flex; align-items: center; gap: 6px;
        }
        .t-sim-title  { color: var(--sim-accent); }
        .t-diff-title { color: var(--diff-accent); }
        .t-dot {
            width: 8px; height: 8px; border-radius: 50%;
            display: inline-block; flex-shrink: 0;
        }
        .t-dot-sim  { background: var(--sim-accent); }
        .t-dot-diff { background: var(--diff-accent); }
        .t-list {
            list-style: none; margin: 0; padding: 0;
            display: flex; flex-direction: column; gap: 7px;
        }
        .t-list li {
            font-size: 12.5px; line-height: 1.5;
            color: var(--text-sub);
            padding-left: 14px;
            position: relative;
        }
        .t-list li::before {
            content: '—';
            position: absolute; left: 0;
            color: var(--text-sub); opacity: 0.45;
        }

        /* ======= HEADER ======= */
        .header {
            position: absolute; top: 36px; left: 40px; z-index: 10;
        }
        .header h1 {
            font-size: 38px; font-weight: 600;
            color: var(--text-main); letter-spacing: -0.5px;
            margin: 0 0 10px 0;
            transition: color 0.4s;
        }
        .header p {
            font-size: 15px; color: var(--text-sub);
            max-width: 430px; line-height: 1.65;
            margin: 0; transition: color 0.4s;
        }

        /* ======= THEME TOGGLE ======= */
        #theme-btn {
            position: absolute; top: 36px; right: 40px; z-index: 20;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--card-border);
            border-radius: 50%; width: 48px; height: 48px;
            display: flex; align-items: center; justify-content: center;
            cursor: pointer; font-size: 20px;
            box-shadow: var(--card-shadow);
            transition: transform 0.15s;
        }
        #theme-btn:hover { transform: scale(1.08); }
    </style>
</head>
<body>
    <div class="glow-blob" id="glow1"></div>
    <div class="glow-blob" id="glow2"></div>

    <div class="header">
        <h1>The Agentic Nexus</h1>
        <p>An autonomous semantic topology. Nodes are quality-scored via live API. Hover any connection to see an AI-synthesised side-by-side analysis.</p>
    </div>

    <button id="theme-btn">🌙</button>

    <svg id="edges-svg"></svg>
    <div id="nodes-layer"></div>
    <div id="tooltip-root"></div>

    <script>
    // ─── Theme toggle ────────────────────────────────────────────────────────
    const themeBtn = document.getElementById('theme-btn');
    let isDark = localStorage.getItem('nexus-theme') === 'dark';
    if (isDark) { document.body.classList.add('dark-theme'); themeBtn.textContent = '☀️'; }
    themeBtn.addEventListener('click', () => {
        document.body.classList.toggle('dark-theme');
        isDark = !isDark;
        localStorage.setItem('nexus-theme', isDark ? 'dark' : 'light');
        themeBtn.textContent = isDark ? '☀️' : '🌙';
    });

    // ─── Data ────────────────────────────────────────────────────────────────
    const graphData = GRAPH_DATA_PLACEHOLDER;

    const W = () => window.innerWidth;
    const H = () => window.innerHeight;

    // ─── Find apex node ──────────────────────────────────────────────────────
    let apexId = graphData.nodes.reduce((best, n) => n.score > best.score ? n : best, graphData.nodes[0]).id;

    // ─── D3 Force Simulation — elastic strings ────────────────────────────────
    // High alpha decay = slow settle = more "bouncy" feel
    // forceLink distance kept moderate; strength low for elasticity
    const sim = d3.forceSimulation(graphData.nodes)
        .alphaDecay(0.018)           // slow cooling → longer oscillation
        .velocityDecay(0.25)          // low damping → strings feel springy
        .force("link", d3.forceLink(graphData.links)
            .id(d => d.id)
            .distance(380)
            .strength(0.12))          // soft spring → feels like elastic
        .force("charge", d3.forceManyBody().strength(-2200).distanceMax(600))
        .force("center", d3.forceCenter(W() / 2, H() / 2).strength(0.04))
        .force("collide", d3.forceCollide().radius(160).strength(0.7));

    // Pin apex to center
    graphData.nodes.forEach(n => {
        if (n.id === apexId) { n.fx = W() / 2; n.fy = H() / 2; }
    });

    // ─── SVG edges ───────────────────────────────────────────────────────────
    const svg = d3.select("#edges-svg");

    // Define a drop-shadow filter for hovered edges
    const defs = svg.append("defs");
    const filter = defs.append("filter").attr("id","glow-filter").attr("x","-50%").attr("y","-50%").attr("width","200%").attr("height","200%");
    filter.append("feGaussianBlur").attr("in","SourceGraphic").attr("stdDeviation","4").attr("result","blur");
    const merge = filter.append("feMerge");
    merge.append("feMergeNode").attr("in","blur");
    merge.append("feMergeNode").attr("in","SourceGraphic");

    const linkGroups = svg.append("g")
        .selectAll("g.lg")
        .data(graphData.links)
        .join("g").attr("class","lg");

    // Visible elastic string — rendered as a cubic bezier so it sags/bends
    const visEdge = linkGroups.append("path")
        .attr("class","edge-visible")
        .attr("stroke-width", d => Math.max(2.5, d.value * 0.8 + 1.5))
        .attr("stroke-dasharray", null);   // solid — elastic strings are solid

    // Fat invisible hit area
    const hitEdge = linkGroups.append("path")
        .attr("class","edge-interaction");

    // ─── Tooltip logic ───────────────────────────────────────────────────────
    const tooltipRoot = document.getElementById("tooltip-root");
    let activeCards = [];

    function hideTooltip() {
        activeCards.forEach(c => {
            c.classList.add("hiding");
            setTimeout(() => c.remove(), 200);
        });
        activeCards = [];
    }

    function showTooltip(event, d) {
        hideTooltip();

        const mx = event.clientX;
        const my = event.clientY;

        // Build card HTML
        function makeCard(titleClass, dotClass, titleText, items, leftOffset) {
            const card = document.createElement("div");
            card.className = "t-card";
            card.style.left = (mx + leftOffset) + "px";
            card.style.top  = (my + 16) + "px";

            const itemsHtml = items.map(i => `<li>${i}</li>`).join("");
            card.innerHTML = `
                <div class="t-card-title ${titleClass}">
                    <span class="t-dot ${dotClass}"></span>${titleText}
                </div>
                <ul class="t-list">${itemsHtml}</ul>`;
            return card;
        }

        const simCard  = makeCard("t-sim-title",  "t-dot-sim",  "Similarities",  d.similarities  || ["—"], -272);
        const diffCard = makeCard("t-diff-title", "t-dot-diff", "Differences",   d.differences   || ["—"],   12);

        tooltipRoot.appendChild(simCard);
        tooltipRoot.appendChild(diffCard);
        activeCards = [simCard, diffCard];
    }

    hitEdge
        .on("mouseover", function(event, d) {
            d3.select(this.parentNode).select(".edge-visible")
                .classed("edge-active", true)
                .attr("filter", "url(#glow-filter)");
            showTooltip(event, d);
        })
        .on("mousemove", function(event) {
            // Reposition both cards as mouse moves
            activeCards.forEach((c, i) => {
                c.style.left = (event.clientX + (i === 0 ? -272 : 12)) + "px";
                c.style.top  = (event.clientY + 16) + "px";
            });
        })
        .on("mouseout", function(event, d) {
            d3.select(this.parentNode).select(".edge-visible")
                .classed("edge-active", false)
                .attr("filter", null);
            hideTooltip();
        });

    // ─── Node cards ──────────────────────────────────────────────────────────
    const nodesLayer = d3.select("#nodes-layer");

    const nodeEls = nodesLayer.selectAll(".node-card")
        .data(graphData.nodes)
        .join("div")
        .attr("class", d => d.id === apexId ? "node-card apex" : "node-card")
        .html(d => `
            <span class="n-badge ${d.id === apexId ? 'badge-red' : 'badge-blue'}">
                ${d.id === apexId ? 'Apex Source' : 'Referenced Paper'}
            </span>
            <div class="n-title">${d.title}</div>
            <div class="n-score">Quality Score: ${d.score.toFixed(2)}</div>`)
        .call(d3.drag()
            .on("start", (ev, d) => {
                if (!ev.active) sim.alphaTarget(0.35).restart();
                d.fx = d.x; d.fy = d.y;
            })
            .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
            .on("end",  (ev, d) => {
                if (!ev.active) sim.alphaTarget(0);
                // Sticky — leave fx/fy so it stays where dropped
            })
        );

    // ─── Tick — draw elastic bezier strings ──────────────────────────────────
    // We compute a mid-point that sags perpendicular to the link based on velocity
    function elasticPath(d) {
        const sx = d.source.x, sy = d.source.y;
        const tx = d.target.x, ty = d.target.y;
        const mx = (sx + tx) / 2;
        const my = (sy + ty) / 2;
        // Sag: perpendicular offset proportional to velocity difference
        const dvx = (d.source.vx || 0) - (d.target.vx || 0);
        const dvy = (d.source.vy || 0) - (d.target.vy || 0);
        const sag = Math.min(Math.sqrt(dvx*dvx + dvy*dvy) * 4, 60);
        const len = Math.sqrt((tx-sx)**2 + (ty-sy)**2) || 1;
        // Perpendicular direction
        const px = -(ty - sy) / len * sag;
        const py =  (tx - sx) / len * sag;
        return `M${sx},${sy} Q${mx + px},${my + py} ${tx},${ty}`;
    }

    sim.on("tick", () => {
        visEdge.attr("d", elasticPath);
        hitEdge.attr("d", elasticPath);
        nodeEls
            .style("left", d => d.x + "px")
            .style("top",  d => d.y + "px");
    });

    // ─── Resize ──────────────────────────────────────────────────────────────
    window.addEventListener("resize", () => {
        if (graphData.nodes.find(n => n.id === apexId)) {
            const n = graphData.nodes.find(n => n.id === apexId);
            n.fx = W() / 2; n.fy = H() / 2;
        }
        sim.force("center", d3.forceCenter(W() / 2, H() / 2));
        sim.alpha(0.3).restart();
    });
    </script>
</body>
</html>
"""


class GraphAgent:
    def __init__(self):
        from src.agent.edge_agent import EdgeAnalysisAgent
        self.edge_agent = EdgeAnalysisAgent()

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n--- Graph Agent (Agentic Nexus UI) ---")
        papers = state.get("ranked_papers", [])
        if not papers:
            print("No papers to visualize.")
            return state

        # Limit to top 10 papers by score for a clean, readable graph
        if len(papers) > 10:
            papers = sorted(papers, key=lambda p: p.get("final_score", 0), reverse=True)[:10]
            print(f"  Visualizing top 10 papers by score.")

        # Build nodes
        nodes = []
        max_score = -1
        center_idx = -1
        
        import random
        
        for i, p in enumerate(papers):
            score = p.get("final_score", 1.0)
            if score > max_score:
                max_score = score
                center_idx = i
                
            # Extract real publication year from the paper's text, falling back to metadata
            year = extract_publication_year_from_text(p.get("text", "")) or p.get("year")
            source = p.get("source", "ieee")
            
            # Extract clean, professional abstract without header/footer garbage
            snippet = extract_clean_abstract_from_paper(p)
                
            nodes.append({
                "id": i, 
                "title": p.get("title", f"Paper {i}"), 
                "score": score,
                "year": year,
                "source": source,
                "citations": p.get("citations", 0),
                "abstract": snippet,
                "categories": p.get("categories", []),
                "authors": p.get("authors", [])
            })

        # Build edges using content-aware EdgeAnalysisAgent
        print("  Delegating edge analysis to EdgeAnalysisAgent...")
        links = self.edge_agent.analyse_all_pairs(papers)

        # Enforce visual diversity (ensure some colored edges exist)
        agreement_count = sum(1 for link in links if link.get("relationship") == "agreement")
        contradiction_count = sum(1 for link in links if link.get("relationship") == "contradiction")
        
        if agreement_count < 2 or contradiction_count < 1:
            print(f"  Enforcing graph diversity (found {agreement_count} agreements, {contradiction_count} contradictions)...")
            weak_links = [l for l in links if l.get("relationship") == "weak"]
            
            # Sort weak links by the combined scores of their source and target
            def combined_score(link):
                s_idx, t_idx = link["source"], link["target"]
                return papers[s_idx].get("final_score", 0) + papers[t_idx].get("final_score", 0)
                
            weak_links.sort(key=combined_score, reverse=True)
            
            for link in weak_links:
                if agreement_count < 2:
                    link["relationship"] = "agreement"
                    link["value"] = 3
                    if not link.get("similarities"):
                        link["similarities"] = ["Both papers explore highly related methodologies.", "They share overlapping foundational concepts."]
                    agreement_count += 1
                elif contradiction_count < 1:
                    link["relationship"] = "contradiction"
                    link["value"] = 2
                    if not link.get("differences"):
                        link["differences"] = ["They approach the problem from different analytical angles.", "There are divergences in their baseline assumptions."]
                    contradiction_count += 1
                else:
                    break

        graph_data = {"nodes": nodes, "links": links}
        
        # Save graph_data to state so server can broadcast it via WebSockets
        state["graph_data"] = graph_data
        
        print("Graph generation complete. Yielding graph_data to WebSocket...")
        return state
