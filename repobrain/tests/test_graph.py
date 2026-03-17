import json
from pathlib import Path

import networkx as nx
import pytest

from repobrain.src.graph.graph_builder import GraphBuilder


def _parsed_code(*files: tuple[str, dict]) -> dict:
    return {"files": {path: meta for path, meta in files}}


def _file_meta(imports: list[str] = None, language: str = "Python") -> dict:
    return {
        "language": language,
        "functions": [],
        "classes": [],
        "imports": imports or [],
        "comments": [],
    }


def test_graph_nodes_created(tmp_path):
    parsed = _parsed_code(
        ("auth/service.py", _file_meta()),
        ("auth/routes.py", _file_meta()),
    )
    graph = GraphBuilder().build(parsed, str(tmp_path))
    assert "auth/service.py" in graph.nodes
    assert "auth/routes.py" in graph.nodes


def test_graph_edge_from_import(tmp_path):
    parsed = _parsed_code(
        ("auth/service.py", _file_meta()),
        ("auth/routes.py", _file_meta(imports=["from auth.service import authenticate"])),
    )
    graph = GraphBuilder().build(parsed, str(tmp_path))
    # routes → service
    assert graph.has_edge("auth/routes.py", "auth/service.py")


def test_graph_no_self_loop(tmp_path):
    parsed = _parsed_code(
        ("app.py", _file_meta(imports=["import app"])),
    )
    graph = GraphBuilder().build(parsed, str(tmp_path))
    assert not graph.has_edge("app.py", "app.py")


def test_graphml_written(tmp_path):
    parsed = _parsed_code(("main.py", _file_meta()))
    GraphBuilder().build(parsed, str(tmp_path))
    assert (tmp_path / "analysis" / "dependency_graph.graphml").exists()


def test_centrality_attached(tmp_path):
    parsed = _parsed_code(
        ("a.py", _file_meta(imports=["from b import x"])),
        ("b.py", _file_meta()),
    )
    graph = GraphBuilder().build(parsed, str(tmp_path))
    for node in graph.nodes:
        assert "betweenness_centrality" in graph.nodes[node]
        assert "degree_centrality" in graph.nodes[node]
