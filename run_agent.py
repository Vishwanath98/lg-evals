import json
import sys
from agent.graph import graph, TOOLS
from telemetry.tracer import LangGraphTracer
from langchain_core.messages import HumanMessage, AIMessage

def run_task(task: str):
    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}")

    tracer = LangGraphTracer()

    initial_state = {
        "messages": [HumanMessage(content=task)],
        "tool_calls_made": [],
        "node_path": [],
        "task": task,
        "final_answer": ""
    }

    final_state = {k: v for k, v in initial_state.items()}

    for step in graph.stream(initial_state, stream_mode="updates"):
        for node_name, output in step.items():
            tracer.on_node_end(node_name, output)

            if output.get("node_path") and final_state.get("node_path"):
                tracer.on_edge(
                    final_state["node_path"][-1] if final_state["node_path"] else "start",
                    output["node_path"][-1]
                )

            for k, v in output.items():
                if isinstance(v, list) and k in final_state and isinstance(final_state[k], list):
                    final_state[k] = final_state[k] + v
                elif v:
                    final_state[k] = v

    if not final_state.get("final_answer"):
        for msg in reversed(final_state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                final_state["final_answer"] = msg.content
                break

    tracer.save(task, final_state)
    coverage = tracer.coverage_report(graph)

    gaps = [g["from"] + "->" + g["to"] for g in coverage["coverage_gaps"]
            if "__" not in g["from"] and "__" not in g["to"]]

    print(f"\n{'='*60}")
    print(f"ANSWER:\n{final_state.get('final_answer', 'none')[:800]}")
    print(f"\nPATH: {' -> '.join(final_state.get('node_path', []))}")
    print(f"TOOLS: {final_state.get('tool_calls_made', [])}")
    print(f"COVERAGE: {coverage['coverage_pct']}%")
    print(f"GAPS: {gaps}")

    return final_state, coverage

if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "list the python files in the current directory and tell me what each one does"
    run_task(task)
