import asyncio
import operator
import sys
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from telemetry.tracer import LangGraphTracer

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_calls_made: Annotated[list, operator.add]
    node_path: Annotated[list, operator.add]
    task: str
    final_answer: str

SYSTEM_PROMPT = """You are a precise task execution agent with MCP filesystem tools.
Use the most specific tool for each job:
- list_directory or directory_tree for exploring structure
- read_text_file for reading files
- search_files for finding files by pattern
- write_file for creating files
- get_file_info for metadata
Never improvise. Use the right tool for the right job."""

async def run_mcp_task(task: str):
    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"MCP Mode: filesystem server")
    print(f"{'='*60}")

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/home/madhu/lg-evals"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            tool_names = [t.name for t in tools]
            print(f"MCP tools available ({len(tools)}): {tool_names}\n")

            llm = ChatOpenAI(
                base_url="http://127.0.0.1:1235/v1",
                api_key="lmstudio",
                model="qwen3.5-35b-a3b",
                temperature=0.1,
            ).bind_tools(tools)

            llm_plain = ChatOpenAI(
                base_url="http://127.0.0.1:1235/v1",
                api_key="lmstudio",
                model="qwen3.5-35b-a3b",
                temperature=0.1,
            )

            tool_node = ToolNode(tools)

            # All nodes must be async for MCP tools
            async def planner_node(state: AgentState) -> AgentState:
                response = await llm.ainvoke(
                    [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
                )
                return {
                    "messages": [response],
                    "node_path": ["planner"],
                    "tool_calls_made": [],
                    "task": state["task"],
                    "final_answer": ""
                }

            async def executor_node(state: AgentState) -> AgentState:
                last = state["messages"][-1]
                if not hasattr(last, 'tool_calls') or not last.tool_calls:
                    return {
                        "messages": [],
                        "node_path": ["executor_skip"],
                        "tool_calls_made": [],
                        "task": state["task"],
                        "final_answer": ""
                    }
                tool_calls_made = [tc["name"] for tc in last.tool_calls]
                result = await tool_node.ainvoke(state)
                return {
                    "messages": result["messages"],
                    "node_path": ["executor"],
                    "tool_calls_made": tool_calls_made,
                    "task": state["task"],
                    "final_answer": ""
                }

            async def synthesizer_node(state: AgentState) -> AgentState:
                response = await llm_plain.ainvoke(
                    [SystemMessage(content="Give a clear, complete final answer based on all results.")] +
                    state["messages"]
                )
                return {
                    "messages": [response],
                    "node_path": ["synthesizer"],
                    "tool_calls_made": [],
                    "task": state["task"],
                    "final_answer": response.content
                }

            def route_after_planner(state: AgentState) -> str:
                last = state["messages"][-1]
                if hasattr(last, 'tool_calls') and last.tool_calls:
                    return "executor"
                return "synthesizer"

            def route_after_executor(state: AgentState) -> str:
                if len(state["messages"]) > 14 or len(state["tool_calls_made"]) > 6:
                    return "synthesizer"
                if isinstance(state["messages"][-1], ToolMessage):
                    return "planner"
                return "synthesizer"

            builder = StateGraph(AgentState)
            builder.add_node("planner", planner_node)
            builder.add_node("executor", executor_node)
            builder.add_node("synthesizer", synthesizer_node)
            builder.set_entry_point("planner")
            builder.add_conditional_edges("planner", route_after_planner, {
                "executor": "executor",
                "synthesizer": "synthesizer"
            })
            builder.add_conditional_edges("executor", route_after_executor, {
                "planner": "planner",
                "synthesizer": "synthesizer"
            })
            builder.add_edge("synthesizer", END)
            graph = builder.compile()

            tracer = LangGraphTracer()

            initial_state = {
                "messages": [HumanMessage(content=task)],
                "tool_calls_made": [],
                "node_path": [],
                "task": task,
                "final_answer": ""
            }

            final_state = {k: v for k, v in initial_state.items()}

            # Use async streaming for MCP tools
            async for step in graph.astream(initial_state, stream_mode="updates"):
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
            coverage = tracer.coverage_report(graph)
            gaps = [g["from"] + "->" + g["to"] for g in coverage["coverage_gaps"]
                    if "__" not in g["from"] and "__" not in g["to"]]

            print(f"\n{'='*60}")
            print(f"ANSWER:\n{final_state.get('final_answer', 'none')[:800]}")
            print(f"\nPATH: {' -> '.join(final_state.get('node_path', []))}")
            print(f"TOOLS USED: {final_state.get('tool_calls_made', [])}")
            print(f"COVERAGE: {coverage['coverage_pct']}%")
            print(f"GAPS: {gaps}")

            return final_state, coverage, tool_names

if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "list all files in this directory and show the directory tree"
    asyncio.run(run_mcp_task(task))
