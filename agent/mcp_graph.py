import operator
import asyncio
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_calls_made: Annotated[list, operator.add]
    node_path: Annotated[list, operator.add]
    task: str
    final_answer: str

def get_llm(tools):
    return ChatOpenAI(
        base_url="http://127.0.0.1:1235/v1",
        api_key="lmstudio",
        model="qwen3.5-35b-a3b",
        temperature=0.1,
    ).bind_tools(tools)

SYSTEM_PROMPT = """You are a precise task execution agent.
You have access to filesystem tools via MCP protocol.
These are APPLICATION tools — not shell commands.
Always use the specific tool for the job:
- Use filesystem tools for reading/writing files
- Use search tools for finding files
- Never improvise with terminal commands
Think step by step and use the minimum tools needed."""

def make_nodes(tools):
    llm = get_llm(tools)
    tool_node = ToolNode(tools)

    def planner_node(state: AgentState) -> AgentState:
        response = llm.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        )
        return {
            "messages": [response],
            "node_path": ["planner"],
            "tool_calls_made": [],
            "task": state["task"],
            "final_answer": ""
        }

    def executor_node(state: AgentState) -> AgentState:
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
        result = tool_node.invoke(state)
        return {
            "messages": result["messages"],
            "node_path": ["executor"],
            "tool_calls_made": tool_calls_made,
            "task": state["task"],
            "final_answer": ""
        }

    def synthesizer_node(state: AgentState) -> AgentState:
        llm_plain = ChatOpenAI(
            base_url="http://127.0.0.1:1235/v1",
            api_key="lmstudio",
            model="qwen3.5-35b-a3b",
            temperature=0.1,
        )
        response = llm_plain.invoke(
            [SystemMessage(content="Summarize the results and give a clear final answer.")] +
            state["messages"]
        )
        return {
            "messages": [response],
            "node_path": ["synthesizer"],
            "tool_calls_made": [],
            "task": state["task"],
            "final_answer": response.content
        }

    return planner_node, executor_node, synthesizer_node, tool_node

def route_after_planner(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, 'tool_calls') and last.tool_calls:
        return "executor"
    return "synthesizer"

def route_after_executor(state: AgentState) -> str:
    if len(state["messages"]) > 14 or len(state["tool_calls_made"]) > 6:
        return "synthesizer"
    from langchain_core.messages import ToolMessage
    if isinstance(state["messages"][-1], ToolMessage):
        return "planner"
    return "synthesizer"

async def build_mcp_graph():
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/home/madhu/lg-evals"],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)

            print(f"MCP tools loaded: {[t.name for t in tools]}")

            planner, executor, synthesizer, tool_node = make_nodes(tools)

            builder = StateGraph(AgentState)
            builder.add_node("planner", planner)
            builder.add_node("executor", executor)
            builder.add_node("synthesizer", synthesizer)
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
            return graph, tools
