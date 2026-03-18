from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_json(analysis_dir: Path, filename: str) -> dict | None:
    p = analysis_dir / filename
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _not_analyzed():
    st.warning("No analysis found. Run `repobrain analyze <repo_path>` first.")


def _build_llm(provider: str, api_key: str, model: str):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if provider == "openai":
        from repobrain.src.llm.openai_client import OpenAIClient
        return OpenAIClient(api_key=api_key, model=model)
    else:
        from repobrain.src.llm.ollama_client import OllamaClient
        return OllamaClient(model=model)


def _build_rag(chroma_dir: str, embedding_model: str):
    from repobrain.src.llm.rag import RAGIndex
    return RAGIndex(persist_dir=chroma_dir, embedding_model=embedding_model)


# ── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RepoBrain",
    page_icon="🧠",
    layout="wide",
)

# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 RepoBrain")
    st.markdown("---")

    repo_path = st.text_input("Repository path", value=".", help="Path to the analyzed repository")
    analysis_dir = Path(repo_path) / "analysis"

    st.markdown("**LLM Provider**")
    provider = st.radio("Provider", ["local (Ollama)", "OpenAI"], label_visibility="collapsed")
    use_openai = provider == "OpenAI"

    if use_openai:
        api_key = st.text_input("OpenAI API Key", type="password")
        llm_model = st.text_input("Model", value="gpt-4o")
    else:
        api_key = ""
        llm_model = st.text_input("Ollama model", value="llama3")

    chroma_dir = st.text_input("Chroma persist dir", value="./.chroma")
    embedding_model = "all-MiniLM-L6-v2"

    st.markdown("---")
    if st.button("Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Running full analysis..."):
            try:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
                from repobrain.src.scanner.scanner import RepoScanner
                from repobrain.src.parser.parser import CodeParser
                from repobrain.src.graph.graph_builder import GraphBuilder
                from repobrain.src.architecture.analyzer import ArchitectureAnalyzer
                from repobrain.src.llm.rag import RAGIndex

                summary = RepoScanner().scan(repo_path)
                parsed = CodeParser().parse(summary["files"], repo_path)
                graph = GraphBuilder().build(parsed, repo_path)
                ArchitectureAnalyzer().analyze(graph, parsed, repo_path)
                rag = RAGIndex(persist_dir=chroma_dir, embedding_model=embedding_model)
                rag.build(parsed, repo_path)

                # Summarize modules using selected LLM
                from repobrain.src.summarizer.summarizer import ModuleSummarizer
                llm = _build_llm(
                    "openai" if use_openai else "local",
                    api_key,
                    llm_model,
                )
                ModuleSummarizer().summarize_all(parsed, rag, llm, repo_path)

                st.success("Analysis complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ── main tabs ─────────────────────────────────────────────────────────────────

tab_overview, tab_arch, tab_deps, tab_modules, tab_ask, tab_impact = st.tabs([
    "Overview", "Architecture", "Dependencies", "Modules", "Ask", "Impact"
])

# ── Overview ──────────────────────────────────────────────────────────────────

with tab_overview:
    st.header("Repository Overview")
    summary = _load_json(analysis_dir, "repository_summary.json")
    if summary is None:
        _not_analyzed()
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Files", summary.get("file_count", 0))
        total_loc = sum(summary.get("loc_by_language", {}).values())
        col2.metric("Lines of Code", f"{total_loc:,}")
        col3.metric("Languages", len(summary.get("languages", [])))

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Languages")
            for lang in summary.get("languages", []):
                loc = summary.get("loc_by_language", {}).get(lang, 0)
                st.markdown(f"- **{lang}** — {loc:,} lines")

        with col_b:
            st.subheader("Frameworks")
            for fw in summary.get("frameworks", []) or ["None detected"]:
                st.markdown(f"- {fw}")

        st.subheader("Entry Points")
        for ep in summary.get("entry_points", []) or ["None detected"]:
            st.code(ep)

# ── Architecture ──────────────────────────────────────────────────────────────

with tab_arch:
    st.header("Architecture")
    arch = _load_json(analysis_dir, "architecture_report.json")
    if arch is None:
        _not_analyzed()
    else:
        st.info(f"**Detected pattern:** {arch.get('pattern', 'Unknown')}")

        view = st.radio("View mode", ["Graph", "List"], horizontal=True)

        if view == "Graph":
            try:
                from streamlit_agraph import agraph, Node, Edge, Config

                LAYER_COLORS = {
                    "API": "#4CAF50",
                    "Service": "#2196F3",
                    "Repository": "#FF9800",
                    "Model": "#9C27B0",
                    "Database": "#F44336",
                    "Utility": "#607D8B",
                    "Config": "#795548",
                    "Test": "#009688",
                    "Unknown": "#9E9E9E",
                }

                layers: dict = arch.get("layers", {})
                nodes = []
                edges = []
                seen_nodes: set[str] = set()

                # Build nodes
                for layer, mods in layers.items():
                    color = LAYER_COLORS.get(layer, "#9E9E9E")
                    for mod in mods:
                        node_id = mod
                        label = Path(mod).stem
                        if node_id not in seen_nodes:
                            nodes.append(Node(
                                id=node_id,
                                label=label,
                                color=color,
                                title=f"[{layer}] {mod}",
                                size=20,
                            ))
                            seen_nodes.add(node_id)

                # Load graph edges from GraphML if available
                graphml = analysis_dir / "dependency_graph.graphml"
                if graphml.exists():
                    import networkx as nx
                    g = nx.read_graphml(str(graphml))
                    for src, dst in g.edges():
                        if src in seen_nodes and dst in seen_nodes:
                            edges.append(Edge(source=src, target=dst))

                config = Config(
                    width="100%",
                    height=600,
                    directed=True,
                    physics=True,
                    hierarchical=False,
                    node={"font": {"color": "#ffffff"}},
                    edge={"color": "#888888"},
                )

                if nodes:
                    agraph(nodes=nodes, edges=edges, config=config)
                    # Legend
                    st.markdown("**Layer colors:**")
                    cols = st.columns(len(LAYER_COLORS))
                    for i, (layer, color) in enumerate(LAYER_COLORS.items()):
                        cols[i].markdown(
                            f"<span style='color:{color}'>■</span> {layer}",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No modules to display.")

            except ImportError:
                st.warning("streamlit-agraph not installed. Showing Mermaid diagram instead.")
                mermaid = arch.get("mermaid_diagram", "")
                if mermaid:
                    st.code(mermaid, language="text")

        else:  # List view
            layers = arch.get("layers", {})
            for layer, mods in layers.items():
                if not mods:
                    continue
                with st.expander(f"{layer} Layer — {len(mods)} module(s)"):
                    for mod in mods:
                        st.markdown(f"- `{mod}`")

# ── Dependencies ──────────────────────────────────────────────────────────────

with tab_deps:
    st.header("Dependency Graph")
    png_path = analysis_dir / "dependency_graph.png"
    graphml = analysis_dir / "dependency_graph.graphml"

    if not graphml.exists() and not png_path.exists():
        _not_analyzed()
    else:
        if png_path.exists():
            st.image(str(png_path), use_container_width=True)
        elif graphml.exists():
            st.info("PNG not available (graphviz not installed). Showing data from GraphML.")

        # Top modules by centrality
        if graphml.exists():
            import networkx as nx
            g = nx.read_graphml(str(graphml))
            rows = []
            for node in g.nodes:
                rows.append({
                    "Module": node,
                    "Betweenness": round(float(g.nodes[node].get("betweenness_centrality", 0)), 4),
                    "Degree": round(float(g.nodes[node].get("degree_centrality", 0)), 4),
                    "In-degree": g.in_degree(node),
                    "Out-degree": g.out_degree(node),
                })
            if rows:
                rows.sort(key=lambda r: r["Betweenness"], reverse=True)
                st.subheader("Top Modules by Centrality")
                import pandas as pd
                st.dataframe(pd.DataFrame(rows[:10]), use_container_width=True)

# ── Modules ───────────────────────────────────────────────────────────────────

with tab_modules:
    st.header("Module Summaries")
    summaries = _load_json(analysis_dir, "module_summaries.json")
    if summaries is None:
        _not_analyzed()
    else:
        search = st.text_input("Filter modules", placeholder="e.g. auth, service...")
        for path, data in summaries.items():
            if search and search.lower() not in path.lower():
                continue
            with st.expander(path):
                st.markdown(data.get("summary", "_No summary available._"))

# ── Ask ───────────────────────────────────────────────────────────────────────

with tab_ask:
    st.header("Ask About the Repository")
    question = st.text_input("Question", placeholder="Where is authentication implemented?")
    if st.button("Ask", key="ask_btn") and question:
        with st.spinner("Querying knowledge index..."):
            try:
                rag = _build_rag(chroma_dir, embedding_model)
                if not rag.is_built():
                    st.warning("Knowledge index not found. Run analysis first.")
                else:
                    chunks = rag.query(question, top_k=5)
                    context = "\n---\n".join(chunks)
                    llm = _build_llm(
                        "openai" if use_openai else "local",
                        api_key,
                        llm_model,
                    )
                    prompt = (
                        f"You are a repository analysis assistant. Answer the following question "
                        f"based ONLY on the repository context provided.\n\n"
                        f"Question: {question}\n\n"
                        f"Context:\n{context}\n\n"
                        f"Rules: Only answer from context. If not in context, say: "
                        f"'Unable to determine from repository analysis.' "
                        f"Never fabricate file names or behaviors."
                    )
                    answer = llm.complete(prompt)
                    st.markdown("**Answer:**")
                    st.write(answer)
                    with st.expander("Sources"):
                        for chunk in chunks:
                            st.code(chunk)
            except Exception as e:
                st.error(f"Error: {e}")

# ── Impact ────────────────────────────────────────────────────────────────────

with tab_impact:
    st.header("Change Impact & Effort Estimation")
    change_desc = st.text_area(
        "Describe the change",
        placeholder="Add Google SSO login",
        height=80,
    )
    if st.button("Analyze Impact", key="impact_btn") and change_desc:
        with st.spinner("Analyzing impact..."):
            try:
                parsed = _load_json(analysis_dir, "parsed_code.json")
                arch = _load_json(analysis_dir, "architecture_report.json") or {"layers": {}}
                if parsed is None:
                    st.warning("Run analysis first.")
                else:
                    graphml = analysis_dir / "dependency_graph.graphml"
                    import networkx as nx
                    graph = nx.read_graphml(str(graphml)) if graphml.exists() else nx.DiGraph()

                    rag = _build_rag(chroma_dir, embedding_model)
                    llm = _build_llm(
                        "openai" if use_openai else "local",
                        api_key,
                        llm_model,
                    )

                    from repobrain.src.impact.impact_analyzer import ImpactAnalyzer
                    from repobrain.src.effort.effort_estimator import EffortEstimator

                    impact_report = ImpactAnalyzer().analyze(
                        change_desc, graph, parsed, rag, llm, repo_path
                    )
                    effort = EffortEstimator().estimate(
                        impact_report, graph, arch, repo_path, llm=llm
                    )

                    # Effort card
                    complexity = effort["complexity"]
                    badge_color = {"Trivial": "#00bcd4", "Low": "green", "Medium": "orange", "High": "red"}.get(complexity, "gray")
                    st.markdown(
                        f"### Complexity: "
                        f"<span style='color:{badge_color}; font-weight:bold'>{complexity}</span>",
                        unsafe_allow_html=True,
                    )

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Engineering Effort", effort["effort_range"])
                    col2.metric("Testing Effort", effort["testing_effort"])
                    col3.metric("Confidence", effort["confidence"])

                    st.markdown("**Layers touched:** " + ", ".join(effort["layers_touched"] or ["unknown"]))

                    # Show LLM reasoning if available
                    llm_reasoning = effort.get("scores", {}).get("llm_reasoning", "")
                    if llm_reasoning:
                        llm_diff = effort.get("scores", {}).get("llm_difficulty", "")
                        st.markdown(f"**AI Assessment** (difficulty {llm_diff}/10): {llm_reasoning}")

                    st.markdown("---")
                    col_seed, col_affected = st.columns(2)
                    with col_seed:
                        st.subheader(f"Seed Modules ({len(impact_report['seed_modules'])})")
                        for m in impact_report["seed_modules"]:
                            st.markdown(f"- `{m}`")
                    with col_affected:
                        st.subheader(f"All Affected Modules ({impact_report['total_affected']})")
                        for m in impact_report["affected_modules"]:
                            st.markdown(f"- `{m}`")

                    # Edit steps with code snippets
                    edit_steps = impact_report.get("edit_steps", [])
                    if edit_steps:
                        st.markdown("---")
                        st.subheader("Suggested Edit Steps")
                        for i, step in enumerate(edit_steps, 1):
                            action = step.get("action", "edit").upper()
                            file = step.get("file", "")
                            desc = step.get("description", "")
                            before = step.get("before", "")
                            after = step.get("after", "")

                            with st.expander(f"Step {i}: `{file}` — **{action}**", expanded=True):
                                st.markdown(desc)

                                if before or after:
                                    col_before, col_after = st.columns(2)
                                    with col_before:
                                        st.markdown("**Before:**")
                                        st.code(before if before else "(no existing code)", language="python")
                                    with col_after:
                                        st.markdown("**After:**")
                                        st.code(after if after else "(removed)", language="python")

            except Exception as e:
                st.error(f"Error: {e}")
