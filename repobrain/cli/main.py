from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from repobrain.config.config import get_config
from repobrain.src.scanner.scanner import RepoScanner
from repobrain.src.parser.parser import CodeParser
from repobrain.src.graph.graph_builder import GraphBuilder, load_graph
from repobrain.src.architecture.analyzer import ArchitectureAnalyzer
from repobrain.src.llm.rag import RAGIndex
from repobrain.src.summarizer.summarizer import ModuleSummarizer
from repobrain.src.summarizer.flow_mapper import FlowMapper
from repobrain.src.impact.impact_analyzer import ImpactAnalyzer
from repobrain.src.effort.effort_estimator import EffortEstimator
from repobrain.src.feedback.feedback_manager import FeedbackManager


def _build_llm_client():
    cfg = get_config()
    provider = cfg.get("llm_provider", "local")
    if provider == "openai":
        from repobrain.src.llm.openai_client import OpenAIClient
        api_key = cfg.get("openai_api_key", "")
        if not api_key:
            click.echo("ERROR: openai_api_key is not set in config.yaml", err=True)
            sys.exit(1)
        return OpenAIClient(api_key=api_key, model=cfg.get("openai_model", "gpt-4o"))
    else:
        from repobrain.src.llm.ollama_client import OllamaClient
        return OllamaClient(
            model=cfg.get("model", "llama3"),
            base_url=cfg.get("ollama_base_url", "http://localhost:11434"),
        )


def _build_rag(cfg: dict) -> RAGIndex:
    return RAGIndex(
        persist_dir=cfg.get("chroma_persist_dir", "./.chroma"),
        embedding_model=cfg.get("embedding_model", "all-MiniLM-L6-v2"),
    )


def _build_feedback_manager(cfg: dict, repo_path: str = ".") -> FeedbackManager:
    return FeedbackManager(
        repo_path=repo_path,
        persist_dir=cfg.get("chroma_persist_dir", "./.chroma"),
        embedding_model=cfg.get("embedding_model", "all-MiniLM-L6-v2"),
    )


def _load_json(repo_path: str, filename: str) -> dict | None:
    p = Path(repo_path).resolve() / "analysis" / filename
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def _sep():
    click.echo("─" * 60)


@click.group()
def cli():
    """RepoBrain — AI-powered repository analysis tool."""


@cli.command()
@click.argument("repo_path")
def analyze(repo_path: str):
    """Full repository analysis: scan → parse → graph → architecture → summarize."""
    cfg = get_config()
    click.echo(f"\nAnalyzing repository: {repo_path}")
    _sep()

    # 1. Scan
    click.echo("Scanning repository...")
    scanner = RepoScanner()
    summary = scanner.scan(repo_path)
    click.echo(f"  Languages detected : {', '.join(summary['languages']) or 'none'}")
    click.echo(f"  Frameworks detected: {', '.join(summary['frameworks']) or 'none'}")
    click.echo(f"  Files analyzed     : {summary['file_count']}")
    click.echo(f"  Entry points       : {', '.join(summary['entry_points']) or 'none'}")

    # 2. Parse
    click.echo("\nParsing source files...")
    parser = CodeParser()
    parsed = parser.parse(summary["files"], repo_path)
    click.echo(f"  Parsed {len(parsed['files'])} files")

    # 3. Build dependency graph
    click.echo("\nBuilding dependency graph...")
    builder = GraphBuilder()
    graph = builder.build(parsed, repo_path)
    click.echo(f"  Nodes: {graph.number_of_nodes()}  Edges: {graph.number_of_edges()}")

    # 4. Analyze architecture
    click.echo("\nAnalyzing architecture...")
    arch_analyzer = ArchitectureAnalyzer()
    arch = arch_analyzer.analyze(graph, parsed, repo_path)
    click.echo(f"  Pattern detected: {arch['pattern']}")
    for layer, mods in arch["layers"].items():
        if mods:
            click.echo(f"    {layer}: {len(mods)} module(s)")

    # 5. Build RAG index
    click.echo("\nBuilding knowledge index...")
    rag = _build_rag(cfg)
    rag.build(parsed, repo_path)
    click.echo("  Knowledge index built")

    # 6. Summarize modules
    click.echo("\nGenerating module summaries (this may take a while)...")
    llm = _build_llm_client()
    summarizer = ModuleSummarizer()
    summaries = summarizer.summarize_all(parsed, rag, llm, repo_path)
    click.echo(f"  Summarized {len(summaries)} modules")

    # 7. Map flows
    click.echo("\nMapping execution flows...")
    flow_mapper = FlowMapper()
    flows = flow_mapper.map_flows(graph, summary["entry_points"], parsed)
    for flow in flows[:3]:
        chain = " → ".join(Path(f).stem for f in flow["flow"][:5])
        click.echo(f"  {chain}{'...' if len(flow['flow']) > 5 else ''}")

    _sep()
    click.echo("Analysis complete. Artifacts saved to ./analysis/")


@cli.command()
@click.argument("repo_path")
def architecture(repo_path: str):
    """Show the architecture of an already-analyzed repository."""
    arch = _load_json(repo_path, "architecture_report.json")
    if arch is None:
        click.echo("No architecture report found. Run `repobrain analyze` first.")
        return

    _sep()
    click.echo(f"Architecture Pattern: {arch['pattern']}")
    _sep()
    for layer, mods in arch["layers"].items():
        if mods:
            click.echo(f"\n{layer} Layer ({len(mods)} modules):")
            for mod in mods:
                click.echo(f"  • {mod}")
    _sep()
    click.echo("\nMermaid diagram source:")
    click.echo(arch.get("mermaid_diagram", ""))


@cli.command()
@click.argument("question")
def ask(question: str):
    """Ask a natural language question about the repository."""
    cfg = get_config()
    rag = _build_rag(cfg)

    if not rag.is_built():
        click.echo("Knowledge index not found. Run `repobrain analyze` first.")
        return

    context_chunks = rag.query(question, top_k=5)
    if not context_chunks:
        click.echo("Unable to determine from repository analysis.")
        return

    context = "\n---\n".join(context_chunks)
    llm = _build_llm_client()

    prompt = (
        f"You are a repository analysis assistant. Answer the following question "
        f"based ONLY on the repository context provided below.\n\n"
        f"Question: {question}\n\n"
        f"Repository context:\n{context}\n\n"
        f"Rules:\n"
        f"1. Only answer based on the repository context provided.\n"
        f"2. If the answer is not in the context, respond with: "
        f"'Unable to determine from repository analysis.'\n"
        f"3. If the question is unrelated to the repository, respond with: "
        f"'This question is not related to the analyzed repository.'\n"
        f"4. Never fabricate module names, file paths, or behaviors."
    )

    answer = llm.complete(prompt)
    _sep()
    click.echo(answer)
    _sep()
    click.echo("\nSources:")
    for chunk in context_chunks:
        first_line = chunk.split("\n")[0]
        click.echo(f"  • {first_line}")


@cli.command()
@click.argument("change_description")
def impact(change_description: str):
    """Analyze the impact and effort of a proposed change."""
    cfg = get_config()

    # Need parsed_code + graph for analysis
    # We use current working directory as repo_path context
    # User should run from the repo root, or we look for ./analysis/
    analysis_dir = Path("./analysis")
    if not (analysis_dir / "parsed_code.json").exists():
        click.echo("No analysis found. Run `repobrain analyze <repo_path>` first.")
        return

    parsed = _load_json(".", "parsed_code.json")
    arch = _load_json(".", "architecture_report.json") or {"layers": {}}

    from repobrain.src.graph.graph_builder import load_graph
    graph = load_graph(".")

    rag = _build_rag(cfg)
    if not rag.is_built():
        click.echo("Knowledge index not found. Run `repobrain analyze` first.")
        return

    llm = _build_llm_client()
    fm = _build_feedback_manager(cfg)

    # Ensure global brain is seeded on first run (idempotent)
    n_global = fm.ensure_global_brain()

    click.echo(f"\nAnalyzing impact of: \"{change_description}\"")
    click.echo(f"  Global brain: {n_global} heuristics | Repo brain: {len(fm.get_history())} feedback entries")
    _sep()

    cal = fm.get_calibration_factor()
    if cal != 1.0:
        direction = "lower" if cal < 1.0 else "higher"
        click.echo(f"  (Repo calibration: {cal:.2f}x — trending {direction} from your past feedback)")

    analyzer = ImpactAnalyzer()
    impact_report = analyzer.analyze(
        change_description, graph, parsed, rag, llm, ".",
        feedback_manager=fm, architecture_report=arch,
    )

    estimator = EffortEstimator()
    effort = estimator.estimate(impact_report, graph, arch, ".", llm=llm, feedback_manager=fm)

    # ── Vagueness guard ───────────────────────────────────────────────────
    if effort.get("insufficient_info"):
        click.echo("\n⚠  INSUFFICIENT INFORMATION — task description is too vague for a")
        click.echo("   reliable estimate. Results below are approximate.")
        _sep()

    # ── Clarified intent ──────────────────────────────────────────────────
    clarified = effort.get("clarified_intent", "")
    if clarified:
        click.echo(f"\nClarified intent: {clarified}")

    # ── Seed / affected modules ───────────────────────────────────────────
    click.echo(f"\nSeed modules identified : {len(impact_report['seed_modules'])}")
    for m in impact_report["seed_modules"]:
        click.echo(f"  • {m}")

    click.echo(f"\nTotal affected modules : {impact_report['total_affected']}")
    for m in impact_report["affected_modules"]:
        click.echo(f"  • {m}")

    _sep()

    # ── Task breakdown ────────────────────────────────────────────────────
    task_breakdown = effort.get("task_breakdown", [])
    if task_breakdown:
        click.echo("Task Breakdown:")
        for t in task_breakdown:
            label = f"{t['task']} ({t['layer']})"
            time_range = f"{t['min_hours']:.1f}–{t['max_hours']:.1f}h"
            click.echo(f"  • {label:<50} {time_range}")
        click.echo("")

    # ── Scores ───────────────────────────────────────────────────────────
    dt_score = effort.get("dev_thinking_score", 0)
    graph_total = effort["scores"].get("graph_total", 0)
    click.echo(f"Dev-Thinking Score  : {dt_score}/25")
    click.echo(f"Graph Score         : {graph_total:.1f}")

    eta = effort.get("eta_range", "")
    if eta:
        u_mult = effort.get("uncertainty_mult", 1.0)
        c_mult = effort.get("context_mult", 1.0)
        unknown_level = effort.get("unknown_level", "Medium")
        click.echo(f"ETA                 : {eta}  [uncertainty: {unknown_level} ×{u_mult}, context: ×{c_mult:.1f}]")

    click.echo("")
    click.echo(f"Complexity          : {effort['complexity']}")
    click.echo(f"Engineering Effort  : {effort['effort_range']}")
    click.echo(f"Testing Effort      : {effort['testing_effort']}")
    click.echo(f"Confidence          : {effort['confidence']}")
    click.echo(f"Layers touched      : {', '.join(effort['layers_touched']) or 'unknown'}")
    if cal != 1.0:
        click.echo(f"Calibration factor  : {effort['scores'].get('calibration_factor', cal):.2f}x (from your feedback)")

    # ── Unknowns ──────────────────────────────────────────────────────────
    unknowns = effort.get("unknowns", [])
    if unknowns:
        click.echo("\nUnknowns:")
        for u in unknowns:
            click.echo(f"  ! {u}")

    # ── Risks ─────────────────────────────────────────────────────────────
    risks = effort.get("risks", [])
    if risks:
        click.echo("\nRisks:")
        for r in risks:
            click.echo(f"  ⚠ {r}")

    # ── Reasoning ────────────────────────────────────────────────────────
    reasoning = effort.get("reasoning", "")
    if reasoning:
        click.echo(f"\nReasoning: {reasoning}")

    _sep()
    click.echo("\nTip: Run `repobrain feedback \"<your thoughts>\"` to improve future estimates.")


@cli.command()
@click.argument("feedback_text")
def feedback(feedback_text: str):
    """Provide natural language feedback on the last effort estimate.

    Example: repobrain feedback "this should take much less time"
    """
    cfg = get_config()
    analysis_dir = Path("./analysis")
    if not (analysis_dir / "effort_estimation.json").exists():
        click.echo("No estimate found. Run `repobrain impact <change>` first.")
        return

    llm = _build_llm_client()
    fm = _build_feedback_manager(cfg)

    click.echo("\nProcessing feedback...")
    result = fm.add(feedback_text, llm)

    if "error" in result:
        click.echo(f"ERROR: {result['error']}")
        return

    direction = result["parsed_direction"]
    magnitude = result["parsed_magnitude"]
    factor = fm.get_calibration_factor()

    _sep()
    click.echo(f"Feedback recorded for: \"{result['change_description']}\"")
    click.echo(f"  Previous estimate : {result['estimated_complexity']} ({result['estimated_effort']})")
    click.echo(f"  Correction        : {direction} (magnitude: {magnitude:.1f})")
    click.echo(f"  Calibration factor: {factor:.2f}x")

    history = fm.get_history()
    click.echo(f"  Total feedback entries: {len(history)}")
    _sep()
    if direction == "lower":
        click.echo("Future estimates will trend lower based on your feedback.")
    elif direction == "higher":
        click.echo("Future estimates will trend higher based on your feedback.")
    else:
        click.echo("Estimate marked as correct. No adjustment applied.")
