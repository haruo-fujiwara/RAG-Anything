#!/usr/bin/env python3
import argparse
from pathlib import Path

import networkx as nx


def _load_graph(path: Path) -> nx.Graph:
    return nx.read_graphml(path)


def _topk_subgraph(graph: nx.Graph, max_nodes: int) -> nx.Graph:
    if max_nodes <= 0 or graph.number_of_nodes() <= max_nodes:
        return graph
    ranked = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    keep = [node for node, _ in ranked[:max_nodes]]
    return graph.subgraph(keep).copy()


def _inject_freeze_after_stabilize(output_path: Path) -> None:
    html = output_path.read_text(encoding="utf-8")
    marker = "var network = new vis.Network(container, data, options);"
    if marker not in html or "stabilizationIterationsDone" in html:
        return
    injection = (
        marker
        + "\n"
        + '    network.once("stabilizationIterationsDone", function () {\n'
        + "      network.setOptions({ physics: false });\n"
        + "    });"
    )
    output_path.write_text(html.replace(marker, injection), encoding="utf-8")


def _render_pyvis(
    graph: nx.Graph, output_path: Path, physics: bool, freeze_after_stabilize: bool
) -> None:
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise SystemExit(
            "pyvis is required for HTML output. Install with: pip install pyvis"
        ) from exc

    net = Network(height="750px", width="100%", directed=graph.is_directed())
    if physics:
        net.barnes_hut()
    else:
        net.toggle_physics(False)

    for node, data in graph.nodes(data=True):
        entity_id = data.get("entity_id") or str(node)
        entity_type = data.get("entity_type")
        label = f"{entity_id} ({entity_type})" if entity_type else entity_id
        title_parts = []
        if entity_type:
            title_parts.append(f"type: {entity_type}")
        if data.get("description"):
            title_parts.append(data["description"])
        title = "\n\n".join(title_parts) if title_parts else label
        net.add_node(node, label=label, title=title)

    for source, target, data in graph.edges(data=True):
        title = data.get("description") or ""
        value = data.get("weight")
        net.add_edge(source, target, title=title, value=value)

    # Avoid notebook rendering; write a standalone HTML file.
    net.write_html(str(output_path), open_browser=False, notebook=False)
    if physics and freeze_after_stabilize:
        _inject_freeze_after_stabilize(output_path)


def _render_dot(graph: nx.Graph, output_path: Path) -> None:
    try:
        from networkx.drawing.nx_pydot import write_dot
    except ImportError as exc:
        raise SystemExit(
            "pydot is required for DOT output. Install with: pip install pydot"
        ) from exc
    write_dot(graph, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the RAG-Anything knowledge graph (GraphML)."
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("rag_storage/graph_chunk_entity_relation.graphml"),
        help="Path to the GraphML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rag_storage/knowledge_graph.html"),
        help="Output file path (.html for PyVis, .dot for Graphviz).",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=500,
        help="Limit to the top N nodes by degree (0 disables).",
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Enable PyVis physics layout (nodes float).",
    )
    parser.add_argument(
        "--freeze-after-stabilize",
        action="store_true",
        help="Freeze the layout after physics stabilization.",
    )
    parser.add_argument(
        "--entity-type",
        action="append",
        default=[],
        help="Only include nodes with this entity_type (repeatable).",
    )
    parser.add_argument(
        "--exclude-entity-type",
        action="append",
        default=[],
        help="Exclude nodes with this entity_type (repeatable).",
    )
    args = parser.parse_args()

    if not args.graph.exists():
        raise SystemExit(f"Graph file not found: {args.graph}")

    graph = _load_graph(args.graph)
    if args.entity_type or args.exclude_entity_type:
        include = set(args.entity_type)
        exclude = set(args.exclude_entity_type)
        keep_nodes = []
        for node, data in graph.nodes(data=True):
            node_type = data.get("entity_type")
            if include and node_type not in include:
                continue
            if exclude and node_type in exclude:
                continue
            keep_nodes.append(node)
        graph = graph.subgraph(keep_nodes).copy()
    graph = _topk_subgraph(graph, args.max_nodes)

    output_suffix = args.output.suffix.lower()
    if output_suffix == ".dot":
        _render_dot(graph, args.output)
    else:
        _render_pyvis(
            graph,
            args.output,
            physics=args.physics,
            freeze_after_stabilize=args.freeze_after_stabilize,
        )

    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
