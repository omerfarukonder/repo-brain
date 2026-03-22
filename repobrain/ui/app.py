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
                    from repobrain.src.feedback.feedback_manager import FeedbackManager

                    fm = FeedbackManager(
                        repo_path=repo_path,
                        persist_dir=chroma_dir,
                        embedding_model=embedding_model,
                    )

                    cal = fm.get_calibration_factor()
                    if cal != 1.0:
                        direction_label = "lower" if cal < 1.0 else "higher"
                        st.info(
                            f"Feedback calibration active: **{cal:.2f}x** — "
                            f"estimates will trend **{direction_label}** based on your past feedback."
                        )

                    impact_report = ImpactAnalyzer().analyze(
                        change_desc, graph, parsed, rag, llm, repo_path,
                        feedback_manager=fm, architecture_report=arch,
                    )
                    effort = EffortEstimator().estimate(
                        impact_report, graph, arch, repo_path, llm=llm, feedback_manager=fm
                    )
                    st.session_state["last_effort"] = effort
                    st.session_state["last_impact"] = impact_report

                    import pandas as pd

                    # ── Vagueness warning ─────────────────────────────────
                    if effort.get("insufficient_info"):
                        st.warning(
                            "⚠ Task description is too vague for a reliable estimate. "
                            "Results below are approximate — add more detail for higher confidence."
                        )

                    # ── Clarified intent ──────────────────────────────────
                    clarified = effort.get("clarified_intent", "")
                    if clarified:
                        st.info(f"**What RepoBrain understood:** {clarified}")

                    st.markdown("---")

                    # ══════════════════════════════════════════════════════
                    # SECTION 1 — Verdict
                    # ══════════════════════════════════════════════════════
                    complexity = effort["complexity"]
                    badge_color = {
                        "Trivial": "#00bcd4", "Low": "#4caf50",
                        "Medium": "#ff9800", "High": "#f44336",
                    }.get(complexity, "gray")

                    eta_range   = effort.get("eta_range", "")
                    dt_score    = effort.get("dev_thinking_score", 0)
                    graph_total = effort["scores"].get("graph_total", 0)
                    scores      = effort["scores"]
                    unknown_level = effort.get("unknown_level", "Medium")
                    u_mult      = effort.get("uncertainty_mult", 1.0)
                    c_mult      = effort.get("context_mult", 1.0)

                    st.markdown(
                        f"<h2 style='margin-bottom:0'>Verdict: "
                        f"<span style='color:{badge_color}'>{complexity}</span></h2>",
                        unsafe_allow_html=True,
                    )
                    v1, v2, v3, v4 = st.columns(4)
                    v1.metric("ETA", eta_range or effort["effort_range"],
                              help="Calibrated estimate: sum of atomic tasks × uncertainty × context multipliers")
                    v2.metric("Confidence", effort["confidence"],
                              help="Based on number of affected modules and uncertainty level")
                    v3.metric("Engineering Effort", effort["effort_range"],
                              help="Legacy tier-based range (use ETA for calibrated value)")
                    v4.metric("Testing Effort", effort["testing_effort"])

                    st.markdown("---")

                    # ══════════════════════════════════════════════════════
                    # SECTION 2 — Why this estimate? (Scoring breakdown)
                    # ══════════════════════════════════════════════════════
                    st.subheader("Why this estimate?")
                    st.caption(
                        "Two independent models score the change. "
                        "The Dev-Thinking score drives the complexity tier. "
                        "The Graph score reflects structural coupling in the codebase."
                    )

                    left_score, right_score = st.columns(2)

                    # ── Dev-Thinking dimensions (left) ────────────────────
                    with left_score:
                        st.markdown(f"#### Dev-Thinking Score: **{dt_score}/25**")
                        st.caption("Models uncertainty, structure, and familiarity — like a senior engineer would.")

                        _sa  = scores.get("surface_area", 0)
                        _id  = scores.get("integration_depth", 0)
                        _un  = scores.get("unknowns_score", 0)
                        _ri  = scores.get("risk_score", 0)
                        _cl  = scores.get("cognitive_load", 0)

                        _sa_labels  = {1:"Isolated (1–2 modules)", 2:"Contained (3–5)", 3:"Moderate (6–10)", 4:"Broad (11–20)", 5:"System-wide (20+)"}
                        _id_labels  = {1:"Decoupled", 2:"Lightly coupled", 3:"Moderately integrated", 4:"Deeply integrated", 5:"Core system"}
                        _un_labels  = {1:"Fully specified", 3:"Some ambiguity", 5:"Highly vague"}
                        _ri_labels  = {1:"Safe change", 2:"Low risk", 3:"Moderate risk", 4:"High risk", 5:"Critical risk"}
                        _cl_labels  = {1:"Simple logic", 2:"Straightforward", 3:"Multi-step", 4:"Complex", 5:"Expert-level"}

                        def _label(d, v):
                            return d.get(v, d.get(min(d.keys(), key=lambda k: abs(k-v)), ""))

                        dt_rows = [
                            {"Dimension": "Surface Area",     "Score": f"{_sa}/5", "What it means": _sa_labels.get(_sa, ""), "Why": f"{len(effort.get('layers_touched',[]))} layers, {impact_report['total_affected']} modules affected"},
                            {"Dimension": "Integration Depth","Score": f"{_id}/5", "What it means": _label(_id_labels, _id),  "Why": f"coupling={scores.get('coupling_density',0):.1f}, propagation={scores.get('propagation_risk',0):.1f}"},
                            {"Dimension": "Unknowns",         "Score": f"{_un}/5", "What it means": _label(_un_labels, _un),  "Why": f"LLM assessed uncertainty as '{unknown_level}'"},
                            {"Dimension": "Risk",             "Score": f"{_ri}/5", "What it means": _label(_ri_labels, _ri),  "Why": f"{len(effort.get('risks',[]))} risk(s) identified"},
                            {"Dimension": "Cognitive Load",   "Score": f"{_cl}/5", "What it means": _label(_cl_labels, _cl),  "Why": f"chromatic={scores.get('chromatic_estimate',0):.1f}, domination={scores.get('domination_number',0):.1f}"},
                            {"Dimension": "**Total**",        "Score": f"**{dt_score}/25**", "What it means": f"→ **{complexity}**", "Why": ""},
                        ]
                        st.dataframe(pd.DataFrame(dt_rows), use_container_width=True, hide_index=True)

                    # ── Graph signal breakdown (right) ────────────────────
                    with right_score:
                        st.markdown(f"#### Graph Score: **{graph_total:.1f}**")
                        st.caption("Measures structural properties of the dependency graph using graph algorithms.")

                        graph_rows = [
                            {
                                "Signal": "Coupling Density",
                                "Value": f"{scores.get('coupling_density',0):.2f}",
                                "Weight": "25%",
                                "What it measures": "How interconnected affected modules are (0=isolated, 10=fully coupled)",
                            },
                            {
                                "Signal": "Propagation Risk",
                                "Value": f"{scores.get('propagation_risk',0):.2f}",
                                "Weight": "25%",
                                "What it measures": "Likelihood of change cascading via high-betweenness hub modules",
                            },
                            {
                                "Signal": "Chromatic Estimate",
                                "Value": f"{scores.get('chromatic_estimate',0):.2f}",
                                "Weight": "20%",
                                "What it measures": "Min independent work streams (graph coloring — more = less parallelism)",
                            },
                            {
                                "Signal": "Domination Number",
                                "Value": f"{scores.get('domination_number',0):.2f}",
                                "Weight": "15%",
                                "What it measures": "Control points a developer must understand (min dominating set)",
                            },
                            {
                                "Signal": "Amplification Ratio",
                                "Value": f"{scores.get('amplification_ratio',0):.2f}",
                                "Weight": "15%",
                                "What it measures": "How much the change spreads beyond intended files (affected/seed ratio)",
                            },
                        ]
                        if scores.get("calibration_factor", 1.0) != 1.0:
                            graph_rows.append({
                                "Signal": "Feedback Calibration",
                                "Value": f"×{scores.get('calibration_factor',1.0):.2f}",
                                "Weight": "—",
                                "What it measures": "Learned multiplier from your past feedback",
                            })
                        st.dataframe(pd.DataFrame(graph_rows), use_container_width=True, hide_index=True)

                    st.markdown("---")

                    # ══════════════════════════════════════════════════════
                    # SECTION 3 — ETA Breakdown
                    # ══════════════════════════════════════════════════════
                    task_breakdown = effort.get("task_breakdown", [])
                    if task_breakdown:
                        st.subheader("ETA Breakdown")

                        raw_min = sum(t.get("min_hours", 0) for t in task_breakdown)
                        raw_max = sum(t.get("max_hours", 0) for t in task_breakdown)

                        e1, e2, e3 = st.columns(3)
                        e1.metric("Raw sum", f"{raw_min:.1f}h – {raw_max:.1f}h",
                                  help="Direct sum of individual task estimates before multipliers")
                        e2.metric(f"Uncertainty ×{u_mult}",
                                  f"{unknown_level}",
                                  help="Clear=×1.2 | Medium=×1.5 | Vague=×2.0  — reflects how well-specified the task is")
                        e3.metric(f"Context ×{c_mult:.1f}",
                                  arch.get("pattern", "unknown"),
                                  help="Monolith=×1.0 | Pipeline=×1.2 | Layered=×1.3 | Microservices=×1.5 | +20% if 3+ layers")

                        st.caption(
                            f"Final ETA = {raw_min:.1f}h–{raw_max:.1f}h "
                            f"× {u_mult} (uncertainty) × {c_mult:.1f} (context) "
                            f"= **{eta_range}**"
                        )

                        task_rows = []
                        for t in task_breakdown:
                            task_rows.append({
                                "Task": t["task"],
                                "Layer": t["layer"],
                                "Min (h)": t["min_hours"],
                                "Max (h)": t["max_hours"],
                                "Range": f"{t['min_hours']:.1f}–{t['max_hours']:.1f}h",
                            })
                        st.dataframe(pd.DataFrame(task_rows), use_container_width=True, hide_index=True)
                        st.markdown("---")

                    # ══════════════════════════════════════════════════════
                    # SECTION 4 — Unknowns & Risks
                    # ══════════════════════════════════════════════════════
                    unknowns = effort.get("unknowns", [])
                    risks    = effort.get("risks", [])
                    scope    = effort.get("scope", [])

                    if unknowns or risks or scope:
                        st.subheader("Unknowns & Risks")
                        st.caption(
                            "Identified by the LLM during intent clarification. "
                            "These directly affect the Unknowns and Risk dimensions of the Dev-Thinking score."
                        )
                        u_col, r_col = st.columns(2)

                        with u_col:
                            st.markdown(f"**Unknowns ({len(unknowns)}) — drives Unknowns score {scores.get('unknowns_score',0)}/5**")
                            if unknowns:
                                for u in unknowns:
                                    st.markdown(f"> ❓ {u}")
                            else:
                                st.markdown("_None identified — task is well specified._")
                            if scope:
                                st.markdown(f"**Scope:** {', '.join(scope)}")

                        with r_col:
                            st.markdown(f"**Risks ({len(risks)}) — drives Risk score {scores.get('risk_score',0)}/5**")
                            if risks:
                                for r in risks:
                                    st.markdown(f"> ⚠️ {r}")
                            else:
                                st.markdown("_No risks identified._")

                        st.markdown("---")

                    # ══════════════════════════════════════════════════════
                    # SECTION 5 — Affected Modules
                    # ══════════════════════════════════════════════════════
                    st.subheader("Affected Code")
                    if effort.get("layers_touched"):
                        st.caption("Layers: " + ", ".join(effort["layers_touched"]))

                    col_seed, col_affected = st.columns(2)
                    with col_seed:
                        st.markdown(f"**Seed Modules ({len(impact_report['seed_modules'])})** — files the LLM identified as directly needing changes")
                        for m in impact_report["seed_modules"]:
                            st.markdown(f"- `{m}`")
                    with col_affected:
                        st.markdown(f"**All Affected Modules ({impact_report['total_affected']})** — seeds + 1-hop graph neighbors")
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

# ── Feedback section (always visible after an estimate exists) ────────────────

with tab_impact:
    st.markdown("---")
    st.subheader("Feedback")
    st.caption(
        "Tell RepoBrain in plain language whether this estimate was accurate. "
        "Your feedback is stored and used to calibrate all future estimates."
    )

    feedback_text = st.text_input(
        "Your feedback",
        placeholder='e.g. "this should take much less time" or "estimate was spot on"',
        key="feedback_input",
    )

    if st.button("Submit Feedback", key="feedback_btn") and feedback_text:
        effort_on_disk = analysis_dir / "effort_estimation.json"
        if not effort_on_disk.exists():
            st.warning("No estimate found. Run an impact analysis first.")
        else:
            with st.spinner("Processing feedback..."):
                try:
                    from repobrain.src.feedback.feedback_manager import FeedbackManager
                    fm = FeedbackManager(
                        repo_path=repo_path,
                        persist_dir=chroma_dir,
                        embedding_model=embedding_model,
                    )
                    llm = _build_llm(
                        "openai" if use_openai else "local",
                        api_key,
                        llm_model,
                    )
                    result = fm.add(feedback_text, llm)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        factor = fm.get_calibration_factor()
                        direction = result["parsed_direction"]
                        magnitude = result["parsed_magnitude"]
                        history_count = len(fm.get_history())

                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Correction", direction.capitalize())
                        col_b.metric("Magnitude", f"{magnitude:.1f}")
                        col_c.metric("Calibration factor", f"{factor:.2f}x")

                        st.success(
                            f"Feedback recorded! "
                            f"({history_count} total entries — future estimates will reflect this.)"
                        )
                except Exception as e:
                    st.error(f"Feedback error: {e}")

    # Show feedback history if any exists
    history_file = analysis_dir / "feedback_history.json"
    if history_file.exists():
        try:
            import json as _json
            with history_file.open() as _f:
                _history = _json.load(_f)
            if _history:
                with st.expander(f"Feedback history ({len(_history)} entries)"):
                    for entry in reversed(_history[-10:]):
                        ts = entry.get("timestamp", "")[:19].replace("T", " ")
                        st.markdown(
                            f"**{ts}** — _{entry.get('change_description', '')}_ → "
                            f"`{entry.get('parsed_direction')}` "
                            f"({entry.get('parsed_magnitude', 0):.1f}): "
                            f"*\"{entry.get('raw_feedback', '')}\"*"
                        )
        except Exception:
            pass
