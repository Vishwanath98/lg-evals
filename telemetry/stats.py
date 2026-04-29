import sqlite3
import json
from collections import defaultdict

DB_PATH = "/home/madhu/lg-evals/data/lg_telemetry.db"

def full_report():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    runs = conn.execute("""
        SELECT * FROM runs ORDER BY timestamp
    """).fetchall()

    edges = conn.execute("""
        SELECT from_node, to_node, COUNT(*) as count
        FROM edge_traces
        GROUP BY from_node, to_node
    """).fetchall()

    nodes = conn.execute("""
        SELECT node_name, COUNT(*) as count, AVG(latency_ms) as avg_latency,
               SUM(CASE WHEN tool_calls != '[]' THEN 1 ELSE 0 END) as tool_call_runs
        FROM node_traces
        GROUP BY node_name
    """).fetchall()

    tool_usage = conn.execute("""
        SELECT tool_calls FROM node_traces WHERE tool_calls != '[]'
    """).fetchall()

    conn.close()

    # Aggregate tool counts
    tool_counts = defaultdict(int)
    for row in tool_usage:
        for tool in json.loads(row["tool_calls"]):
            tool_counts[tool] += 1

    # Edge transition matrix
    edge_matrix = {}
    for e in edges:
        key = f"{e['from_node']}→{e['to_node']}"
        edge_matrix[key] = e["count"]

    print("\n" + "="*60)
    print("LG-EVALS TELEMETRY REPORT")
    print("="*60)

    print(f"\n RUNS: {len(runs)}")
    for r in runs:
        path = json.loads(r["node_path"])
        tools = json.loads(r["tool_calls"])
        print(f"  [{r['id']}] {r['latency_ms']:.0f}ms | path: {' → '.join(path)} | tools: {tools}")

    print(f"\n NODE STATS:")
    for n in nodes:
        print(f"  {n['node_name']:15} calls: {n['count']:3} | avg latency: {n['avg_latency']:.0f}ms | tool runs: {n['tool_call_runs']}")

    print(f"\n EDGE TRANSITIONS (observed):")
    for edge, count in sorted(edge_matrix.items()):
        bar = "█" * count
        print(f"  {edge:35} {bar} ({count}x)")

    print(f"\n TOOL USAGE ACROSS ALL RUNS:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        bar = "█" * count
        print(f"  {tool:20} {bar} ({count}x)")

    # Coverage gap analysis
    all_possible = {
        "planner→executor",
        "planner→synthesizer",
        "executor→planner",
        "executor→synthesizer",
        "synthesizer→__end__"
    }
    observed = set(edge_matrix.keys())
    gaps = all_possible - observed

    print(f"\n COVERAGE GAPS (never observed across {len(runs)} runs):")
    for gap in gaps:
        print(f"  ⚠ {gap} — UNTESTED PATH")

    coverage_pct = round(len(observed & all_possible) / len(all_possible) * 100, 1)
    print(f"\n OVERALL COVERAGE: {coverage_pct}% ({len(observed & all_possible)}/{len(all_possible)} paths)")

    print("\n" + "="*60)

if __name__ == "__main__":
    full_report()
