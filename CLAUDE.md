# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Galaxy is a multi-agent prediction framework that decomposes complex prediction tasks (particularly financial forecasting) into structured execution plans using a factor-based approach. The system uses LLMs to generate decomposition plans that are then executed by specialized sub-agents.

**Core Architecture Flow:**
1. `PredictionTask` - Parses raw prediction questions into structured task representations
2. `BaseKnowHow` - Domain knowledge definitions that specify factor decomposition strategies
3. `DecomposeAgent` (Planner) - Compiles Task + KnowHow → `DecompositionPlan`
4. `DecompositionPlan` - Machine-executable plan with factor execution steps
5. Sub-Agent Executors (planned) - Execute the plan's tasks
6. Synthesis Agent (planned) - Aggregates factor outputs into final prediction

## Project Structure

```
core/
  schemas/         # Core data structures (plan, knowhow_base, PredictionTask)
  llm/            # LLM client abstractions (base, openrouter_mini_client, gpt5_client)
  tools/          # Tool interface layer (not yet implemented)

planner/
  decompose_agent.py     # Main planner: compiles PredictionTask + KnowHow → DecompositionPlan
  decompose_prompter.py  # Prompt building for decomposition with JSON schema
  smart_matcher.py       # Task-to-KnowHow matching logic

knowhow_store/
  finance/
    stock_price.py  # Stock price prediction knowledge: factors, tools, aggregation rules

excutors/         # Sub-agent executors (planned, not implemented)
synthesis/        # Synthesis agent for aggregating factor outputs (planned)
log/              # JSON output logs for decomposition plans
temp/             # Temporary files
```

## Development Commands

### Running Tests

```bash
# Run main test using OpenRouter mini client (cost-effective)
uv run python test_mini.py

# Run basic test
uv run python test.py
```

The test outputs:
- Console: System/user prompts and plan summary
- `log/` directory: Detailed JSON decomposition plans with timestamps

### Environment Setup

```bash
# Install dependencies with uv
uv sync

# Required environment variables (.env):
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

### Python Environment

Uses Python 3.12+ with uv package manager. Virtual environment is in `.venv/`.

## Key Architectural Concepts

### 1. KnowHow System

`BaseKnowHow` encodes domain expertise as structured knowledge:
- **Factor decomposition**: Breaking predictions into sub-problems (micro_trend, macro_sector_trend, valuation_correction, event_news_impact)
- **Tool catalogs**: Allowed tools per factor (financial_data, web_search, reading, reasoning)
- **Output schemas**: Strict contracts for what each factor must produce
- **Task hints**: Guidance for the planner on computation steps
- **Aggregation specs**: How to synthesize factor outputs

Example: `StockPriceKnowHow` in `knowhow_store/finance/stock_price.py`

### 2. Decomposition Plans

A `DecompositionPlan` consists of:
- **Global assumptions**: Overall context/constraints
- **Placeholders**: Variables like `${ticker}`, `${target_date}`
- **Factors**: List of `FactorExecutionPlan` objects

Each `FactorExecutionPlan` contains:
- **Tasks**: Ordered sequence of `TaskItem` objects
- **Decision logic**: How to interpret factor outputs
- **Output fields**: Must match the factor's output schema exactly

### 3. Task Items (TaskItem)

Executable steps with explicit semantics:
- **kind**: `fetch|compute|extract|judge|write` - determines executor behavior
- **goal**: Human-readable description
- **tool**: For fetch/extract tasks (e.g., "financial_data", "web_search")
- **params**: Tool parameters (may contain placeholders)
- **compute**: Formula/pseudocode for compute/judge tasks
- **writes**: Output fields this task produces (critical for validation)

### 4. Validation Rules

`validate_plan()` in `core/schemas/plan.py` enforces:
- Factor names must exist in KnowHow
- Output fields must exactly match factor's output schema
- Tools must be in factor's whitelist
- Every output field must be written by at least one task
- Tasks must not be empty

### 5. LLM Client Architecture

Base class: `BaseLLMClient` in `core/llm/base.py`
- Abstract interface with `chat()` and `stream_chat()` methods
- Returns `LLMResponse` dataclass with content, tool_calls, usage stats
- Tracks cumulative token usage via `get_usage_stats()`

Implementations:
- `OpenRouterMiniClient`: Uses OpenAI-compatible API, supports async/sync modes
- `GPT5Client`: (Implementation in `gpt5_client.py`)

The `DecomposeAgent._call_llm()` handles both sync and async clients transparently.

## Important Design Patterns

### Prompt Engineering

The decompose prompter (`planner/decompose_prompter.py`) uses:
- Structured JSON schemas for LLM outputs (reduces hallucination)
- Explicit instructions: "Output = ONE JSON object", "NO Markdown"
- Task-level granularity: Every step must specify what it writes
- Placeholder documentation for runtime variable substitution

### Factor-Based Decomposition

Predictions are broken into independent factors (micro trends, macro context, valuation, events) each with:
- Specific analytical role
- Dedicated tools
- Output contracts (expected_return, confidence, reasoning, etc.)
- Aggregation weights

This enables:
- Parallel execution of factors
- Confidence-weighted synthesis
- Interpretable predictions
- Factor-level debugging

### Separation of Concerns

1. **Planning vs Execution**: Planner only generates plans, doesn't execute tools
2. **Knowledge vs Logic**: KnowHow defines what's possible, DecomposeAgent decides how
3. **Schema-driven validation**: Plans validated against KnowHow contracts before execution
4. **Placeholder system**: Plans are templates; executors substitute runtime values

## Common Patterns When Modifying

### Adding a New Factor

1. Update the KnowHow class (e.g., `StockPriceKnowHow`)
2. Define `FactorSpec` with: name, description, agent_role, tools, analysis_steps, output_schema, task_hints
3. Add to decomposition strategy
4. Update aggregation weights if needed
5. The planner will automatically incorporate it

### Adding a New KnowHow Domain

1. Create new file in `knowhow_store/{domain}/{subdomain}.py`
2. Inherit from `BaseKnowHow`
3. Define factors, tools, evaluation criteria, aggregation
4. Update smart_matcher if needed for task routing

### Modifying Decomposition Prompts

Edit `planner/decompose_prompter.py`:
- `system_prompt`: General instructions for the planner
- `user_payload`: Task context, knowhow seed, JSON schema, generation notes
- `DECOMPOSE_OUTPUT_JSON_SCHEMA`: Output structure contract

### Testing Decomposition

The main test pattern (`test_mini.py`):
```python
llm = OpenRouterMiniClient(model="openai/gpt-4.1-mini", temperature=0)
knowhow = StockPriceKnowHow()
task = PredictionTask(task_id="...", task_question="...", metadata={...})
agent = DecomposeAgent(llm)

# Preview prompts before execution
print(agent.preview_prompt(task, knowhow))

# Generate plan (validates automatically)
plan = agent.plan(task, knowhow)

# Plans are saved to log/ directory as timestamped JSON
```

## Notes

- **Tool layer is not implemented**: allowed_tools in KnowHow is interface-reserved for future MCP/tool registry integration
- **Executors are not implemented**: Sub-agent execution layer is planned
- **Synthesis is not implemented**: Final aggregation logic is planned
- **Async support**: LLM clients support both async and sync modes; DecomposeAgent handles both transparently
- **Logging**: Decomposition plans are auto-saved to `log/` directory with timestamps for debugging
- **Temperature=0 for determinism**: Tests use temperature=0 for reproducible plan generation

## File Encoding

All Python files use UTF-8 encoding. Chinese comments are present in test files and README.
