import operator
import json
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ── State ────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_calls_made: Annotated[list, operator.add]
    node_path: Annotated[list, operator.add]  # track execution path
    task: str
    final_answer: str

# ── LLM ──────────────────────────────────────────────────────────────
def get_llm():
    return ChatOpenAI(
        base_url="http://127.0.0.1:1235/v1",  # through observatory proxy
        api_key="lmstudio",
        model="qwen3.5-35b-a3b",
        temperature=0.1,
    )

# ── Tools ─────────────────────────────────────────────────────────────
from langchain_core.tools import tool

@tool
def read_file(path: str) -> str:
    """PREFERRED tool for reading file contents. Always use this instead of terminal cat/head commands."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading {path}: {e}"

@tool
def search_files(directory: str, pattern: str = "*") -> str:
    """PREFERRED tool for listing and finding files. Always use this instead of terminal ls/find commands."""
    import glob, os
    try:
        matches = glob.glob(f"{directory}/{pattern}", recursive=True)
        return "\n".join(matches[:50])
    except Exception as e:
        return f"Error searching {directory}: {e}"

@tool
def write_file(path: str, content: str) -> str:
    """PREFERRED tool for writing files. Always use this instead of terminal echo/redirect commands."""
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"Written to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"

@tool
def run_command(command: str) -> str:
    """Use ONLY for: running scripts, installing packages, git operations.
    DO NOT use for reading files (use read_file), listing files (use search_files)."""
    import subprocess
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {e}"

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"[Web search for '{query}' — connect Firecrawl API for real results]"

@tool
def summarize(content: str, focus: str = "") -> str:
    """Summarize content, optionally focusing on a specific aspect."""
    return f"[Summary of {len(content)} chars focusing on: {focus or 'general'}]"

TOOLS = [read_file, search_files, write_file, run_command, web_search, summarize]

# ── Nodes ─────────────────────────────────────────────────────────────
def planner_node(state: AgentState) -> AgentState:
    """Plans the approach for the given task."""
    llm = get_llm().bind_tools(TOOLS)
    messages = state["messages"]
    
    if not any(isinstance(m, HumanMessage) for m in messages):
        messages = [HumanMessage(content=state["task"])] + messages

    system = """You are a planning agent. Given a task:
1. Break it into clear steps
2. Identify which tools you'll need
3. Start executing the first step

Available tools: read_file, search_files, write_file, run_command, web_search, summarize
Prefer specialized tools (read_file, search_files) over run_command for file operations."""

    from langchain_core.messages import SystemMessage
    response = llm.invoke([SystemMessage(content=system)] + messages)
    
    return {
        "messages": [response],
        "node_path": ["planner"],
        "tool_calls_made": [],
        "task": state["task"],
        "final_answer": ""
    }

def executor_node(state: AgentState) -> AgentState:
    """Executes tool calls from the planner."""
    tool_node = ToolNode(TOOLS)
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {
            "messages": [],
            "node_path": ["executor_skip"],
            "tool_calls_made": [],
            "task": state["task"],
            "final_answer": ""
        }
    
    # Log tool calls
    tool_calls_made = [tc["name"] for tc in last_message.tool_calls]
    
    result = tool_node.invoke(state)
    
    return {
        "messages": result["messages"],
        "node_path": ["executor"],
        "tool_calls_made": tool_calls_made,
        "task": state["task"],
        "final_answer": ""
    }

def synthesizer_node(state: AgentState) -> AgentState:
    """Synthesizes all tool results into a final answer."""
    llm = get_llm()
    
    from langchain_core.messages import SystemMessage
    system = """You are a synthesis agent. Review all the tool results and conversation,
then provide a clear, complete final answer to the original task.
Be concise and direct."""
    
    response = llm.invoke([SystemMessage(content=system)] + state["messages"])
    
    return {
        "messages": [response],
        "node_path": ["synthesizer"],
        "tool_calls_made": [],
        "task": state["task"],
        "final_answer": response.content
    }

# ── Routing ───────────────────────────────────────────────────────────
def route_after_planner(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, 'tool_calls') and last.tool_calls:
        return "executor"
    return "synthesizer"

def route_after_executor(state: AgentState) -> str:
    # Check if we need more planning or can synthesize
    msg_count = len(state["messages"])
    tool_calls_total = len(state["tool_calls_made"])
    
    if msg_count > 12 or tool_calls_total > 6:
        return "synthesizer"  # force synthesis to avoid infinite loops
    
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        return "planner"  # go back to planner with tool results
    return "synthesizer"

# ── Build Graph ───────────────────────────────────────────────────────
def build_graph():
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
    
    return builder.compile()

graph = build_graph()

def build_graph_v2():
    """V2: fixes executor→synthesizer coverage gap"""
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.set_entry_point("planner")

    builder.add_conditional_edges("planner", route_after_planner, {
        "executor": "executor",
        "synthesizer": "synthesizer"
    })

    def route_after_executor_v2(state: AgentState) -> str:
        tool_count = len(state["tool_calls_made"])
        msg_count = len(state["messages"])
        # Direct to synthesizer if single tool call completed cleanly
        if tool_count == 1 and msg_count <= 4:
            return "synthesizer"  # closes the coverage gap
        if msg_count > 12 or tool_count > 5:
            return "synthesizer"
        from langchain_core.messages import ToolMessage
        if isinstance(state["messages"][-1], ToolMessage):
            return "planner"
        return "synthesizer"

    builder.add_conditional_edges("executor", route_after_executor_v2, {
        "planner": "planner",
        "synthesizer": "synthesizer"
    })

    builder.add_edge("synthesizer", END)
    return builder.compile()

graph_v2 = build_graph_v2()

def build_graph_v2():
    """V2: fixes executor→synthesizer coverage gap"""
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.set_entry_point("planner")

    builder.add_conditional_edges("planner", route_after_planner, {
        "executor": "executor",
        "synthesizer": "synthesizer"
    })

    def route_after_executor_v2(state: AgentState) -> str:
        tool_count = len(state["tool_calls_made"])
        msg_count = len(state["messages"])
        # Direct to synthesizer if single tool call completed cleanly
        if tool_count == 1 and msg_count <= 4:
            return "synthesizer"  # closes the coverage gap
        if msg_count > 12 or tool_count > 5:
            return "synthesizer"
        from langchain_core.messages import ToolMessage
        if isinstance(state["messages"][-1], ToolMessage):
            return "planner"
        return "synthesizer"

    builder.add_conditional_edges("executor", route_after_executor_v2, {
        "planner": "planner",
        "synthesizer": "synthesizer"
    })

    builder.add_edge("synthesizer", END)
    return builder.compile()

graph_v2 = build_graph_v2()
