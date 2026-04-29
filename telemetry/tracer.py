import json
import time
import uuid
import sqlite3
from datetime import datetime

DB_PATH = "/home/madhu/lg-evals/data/lg_telemetry.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            task TEXT,
            node_path TEXT,
            tool_calls TEXT,
            total_nodes INTEGER,
            total_tool_calls INTEGER,
            latency_ms REAL,
            final_answer TEXT,
            success INTEGER,
            source TEXT DEFAULT 'inline'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS node_traces (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            node_name TEXT,
            entry_time TEXT,
            exit_time TEXT,
            latency_ms REAL,
            input_keys TEXT,
            output_keys TEXT,
            tool_calls TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edge_traces (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            from_node TEXT,
            to_node TEXT,
            timestamp TEXT,
            condition TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

class LangGraphTracer:
    def __init__(self, source: str = "inline"):
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.node_traces = []
        self.edge_traces = []
        self.current_node_start = None
        self.current_node = None
        self.source = source

    def on_node_start(self, node_name: str, state: dict):
        self.current_node = node_name
        self.current_node_start = time.time()
        print(f"  → [{node_name}] starting...")

    def on_node_end(self, node_name: str, output: dict):
        latency = (time.time() - (self.current_node_start or time.time())) * 1000
        tool_calls = output.get("tool_calls_made", [])
        trace = {
            "id": str(uuid.uuid4())[:8],
            "run_id": self.run_id,
            "node_name": node_name,
            "entry_time": datetime.utcnow().isoformat(),
            "exit_time": datetime.utcnow().isoformat(),
            "latency_ms": latency,
            "input_keys": "[]",
            "output_keys": json.dumps(list(output.keys())),
            "tool_calls": json.dumps(tool_calls)
        }
        self.node_traces.append(trace)
        print(f"  ✓ [{node_name}] done in {latency:.0f}ms | tools: {tool_calls}")

    def on_edge(self, from_node: str, to_node: str, condition: str = ""):
        trace = {
            "id": str(uuid.uuid4())[:8],
            "run_id": self.run_id,
            "from_node": from_node,
            "to_node": to_node,
            "timestamp": datetime.utcnow().isoformat(),
            "condition": condition
        }
        self.edge_traces.append(trace)
        print(f"  ⟶ {from_node} → {to_node}")

    def save(self, task: str, final_state: dict):
        total_latency = (time.time() - self.start_time) * 1000
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.run_id,
            datetime.utcnow().isoformat(),
            task,
            json.dumps(final_state.get("node_path", [])),
            json.dumps(final_state.get("tool_calls_made", [])),
            len(final_state.get("node_path", [])),
            len(final_state.get("tool_calls_made", [])),
            total_latency,
            final_state.get("final_answer", "")[:500],
            1,
            self.source
        ))
        for trace in self.node_traces:
            conn.execute(
                "INSERT INTO node_traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuple(trace.values())
            )
        for trace in self.edge_traces:
            conn.execute(
                "INSERT INTO edge_traces VALUES (?, ?, ?, ?, ?, ?)",
                tuple(trace.values())
            )
        conn.commit()
        conn.close()
        print(f"\n📊 Run {self.run_id} [{self.source}] saved | {total_latency:.0f}ms total")

    def coverage_report(self, graph):
        static_edges = set()
        try:
            g = graph.get_graph()
            for edge in g.edges:
                static_edges.add((edge.source, edge.target))
        except:
            pass
        observed_edges = {(t["from_node"], t["to_node"]) for t in self.edge_traces}
        gaps = static_edges - observed_edges
        return {
            "static_edges": list(static_edges),
            "observed_edges": list(observed_edges),
            "coverage_gaps": [{"from": s, "to": t} for s, t in gaps],
            "coverage_pct": round(len(observed_edges) / max(len(static_edges), 1) * 100, 1)
        }
