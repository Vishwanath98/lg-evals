cat << 'EOF' > ~/lg-evals/README.md
# lg-evals

> Measuring and improving AI agent determinism through telemetry.

Built on an RTX 3090 running local LLMs via LM Studio. Uses LangGraph for agent orchestration and MCP for standardized tool access.

---

## The Problem

AI agents are probabilistic. The same task can take different paths, use different tools, and produce inconsistent results. Without observability, you can't know which execution paths exist, which ones get used, and which ones are broken.

This project instruments a LangGraph agent end-to-end — capturing every node transition, tool call, and routing decision — then uses that data to find and fix gaps.

---

## Key Result

| | V1 Graph | V2 Graph |
|---|---|---|
| Graph coverage | 60% | 80% |
| Avg latency | 5051ms | 2421ms |
| executor→synthesizer path | never triggered | working |

One routing condition change = **52% faster, 20% more coverage.**

The `executor→synthesizer` path existed in the graph topology but was structurally unreachable. Telemetry found it across 7 runs. One line of code fixed it.

---

## How It Works

The agent runs as a 3-node LangGraph graph:
Task input
↓
[Planner]      decides what tools are needed
↓
[Executor]     runs the tools
↓
[Synthesizer]  produces the final answer

A telemetry layer wraps each run and captures:
- Which nodes executed and in what order
- Which graph edges fired vs never triggered
- Which tools were called and how many times
- Latency per node and total per run

All data is stored in SQLite and queryable for cross-run analysis.

---

## Two Tool Backends Compared

| | Inline Python tools | MCP filesystem server |
|---|---|---|
| Tools available | 6 | 14 |
| Avg latency | ~5s | ~10s |
| Setup | simple | subprocess + async |
| Tool specificity | generic | specific (directory_tree, get_file_info) |

The same coverage gap appeared in both backends — confirming it was a graph structure issue, not a tool or model issue.

---

## Stack

- **LangGraph** — agent graph orchestration
- **MCP** — Model Context Protocol filesystem server
- **LM Studio** — local inference server (qwen3.5-35b-a3b on RTX 3090 24GB)
- **SQLite** — telemetry storage across runs
- **FastAPI proxy** — intercepts all LLM calls for token/cost/tool tracking

---

## Project Structure
lg-evals/
agent/
graph.py            V1 and V2 LangGraph agents
telemetry/
tracer.py           Captures node and edge transitions per run
stats.py            Full report across all runs
compare.py          V1 vs V2 vs MCP comparison
run_agent.py          Run V1 inline tool agent
run_mcp_agent.py      Run MCP filesystem agent
run_v2_test.py        Run V2 graph tests
data/
lg_telemetry.db     SQLite database with all run history

---

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install langgraph langchain-openai langchain-core mcp langchain-mcp-adapters httpx rich

~/.lmstudio/bin/lms server start --port 1234

python run_v2_test.py
python telemetry/stats.py

npm install -g @modelcontextprotocol/server-filesystem
python run_mcp_agent.py "list all files in this directory"
```

---

## Sample Telemetry Output
RUNS: 10
[04d0faf3] 7505ms  planner → executor → planner → synthesizer  tools: [search_files, read_file]
[9e4cdaf6] 1907ms  planner → executor → synthesizer             tools: [search_files]
[17675850] 3429ms  planner → synthesizer                        tools: []
EDGE TRANSITIONS (observed):
executor→planner        ████████  8x
executor→synthesizer    ██        2x
planner→executor        ██████████ 10x
planner→synthesizer     ████████  8x
TOOL USAGE:
search_files    ████  4x
read_file       ████  4x
write_file      █     1x
list_directory  █     1x
directory_tree  █     1x
COVERAGE GAPS:
executor→synthesizer  UNTESTED  ← found in V1, fixed in V2

---

## What I Learned

- **Coverage gaps are structural, not probabilistic.** The missing path was a routing condition that made it unreachable by design — not a model behavior issue.
- **MCP adds tool richness at a latency cost.** The model picked more specific tools but subprocess communication adds ~2x overhead.
- **Proxy-level instrumentation is framework-agnostic.** A FastAPI proxy sitting between the agent and the LLM captures everything without touching agent code.

---

## Related

**Hermes Observatory** — A proxy-based telemetry layer for the Hermes autonomous agent. Detected 76.9% tool misrouting. Fixed by rewriting tool descriptions at the proxy layer — zero changes to agent code.
EOF
