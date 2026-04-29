from agent.graph import graph_v2
from telemetry.tracer import LangGraphTracer
from langchain_core.messages import HumanMessage, AIMessage

tasks = [
    "search for json files in this directory",
    "what is the capital of France",
    "read the file run_agent.py and summarize it",
]

for task in tasks:
    print(f"\nTASK: {task}")
    tracer = LangGraphTracer(source="inline-v2")
    state = {
        "messages": [HumanMessage(content=task)],
        "tool_calls_made": [],
        "node_path": [],
        "task": task,
        "final_answer": ""
    }
    final_state = {k: v for k, v in state.items()}

    for step in graph_v2.stream(state, stream_mode="updates"):
        for node_name, output in step.items():
            tracer.on_node_end(node_name, output)
            if output.get("node_path") and final_state.get("node_path"):
                tracer.on_edge(
                    final_state["node_path"][-1],
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
    coverage = tracer.coverage_report(graph_v2)
    print(f"PATH: {' → '.join(final_state.get('node_path', []))}")
    print(f"TOOLS: {final_state.get('tool_calls_made', [])}")
    print(f"COVERAGE: {coverage['coverage_pct']}%")
