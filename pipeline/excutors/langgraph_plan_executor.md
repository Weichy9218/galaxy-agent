# LangGraphPlanExecutor 执行流程说明

本文档解释 `excutors/langgraph_plan_executor.py` 中 SubAgent 执行器的核心流程，便于后续扩展或替换真实工具。

## 架构概览

```
DecompositionPlan  ──>  LangGraphPlanExecutor  ──>  FactorExecutionResult
                           │
                           ├─ ToolRegistry（mock 工具 & reasoning fallback）
                           └─ 可选 DecomposeAgent（用于因子级重新规划）
```

- **Plan 输入**：来自 `DecomposeAgent.plan()` 的 `DecompositionPlan`。每个因子都包含一组有序 `TaskItem`。
- **LangGraph 状态机**：每个因子被编译成一个 `StateGraph`，节点结构为 `manager → execute_task → manager`，遇到异常时会跳转到 `handle_error` 节点。
- **ToolRegistry**：当前默认使用 mock 数据；`enable_reasoning_llm=False` 时只返回启发式决策，避免依赖外部 LLM。后续可替换为真实行情接口或 GPT 推理。
- **Planner 回路**：当 reasoning 判定需要重规划时，执行器会调用 `DecomposeAgent.plan` 重新生成计划，并仅替换当前因子。

## 状态字段

`FactorState` 是图上的共享状态，核心字段如下：

| 字段             | 说明                                 |
|------------------|--------------------------------------|
| `context`        | 执行上下文，包含占位符和任务写入值       |
| `task_index`     | 当前执行到的任务下标                     |
| `current_task`   | 当前准备执行的 `TaskItem`              |
| `task_results`   | 历史执行记录（含 `confidence`）         |
| `retries`        | 每个任务已重试次数                      |
| `last_error`     | 最近一次执行失败的错误描述               |
| `status`         | `manager` / `execute` / `error` / `done` |
| `failed_task`    | 发生错误的任务对象                      |
| `replan_attempted` | 避免重复 replan                      |

## Manager 循环

1. **manager 节点**：选取下一个任务，或在 `status=error` 时等待错误处理结果；当任务全部完成时返回 `done`，终止图执行。
2. **execute_task 节点**：
   - 解析占位符 → 调用工具 → 生成写入结果和粗略 `confidence`；
   - 若 `confidence < 阈值`（默认 0.6），将该任务视为链路异常，转到 `handle_error`。
3. **handle_error 节点**：
   - 调用 reasoning（默认为启发式 fallback），得到 `decision` ∈ {`retry`,`replan`,`skip`}；
   - `retry`：在 `max_retries` 范围内重新执行；
   - `replan`：调用 Planner 重新生成当前因子的任务序列（保留新的 placeholders & global assumptions）；
   - `skip`：将输出标记为错误并继续下一个任务。

所有 reasoning、重试、跳过等结果都会记录在 `task_results` 中，方便后续审计。

## Replan 处理

当 reasoning 返回 `replan` 且尚未尝试重新规划时：

1. 在后台线程里调用 `DecomposeAgent.plan` 获得新的 `DecompositionPlan`；
2. 取出同名因子，更新 `FactorState.factor`、`context`、`placeholders`、`global_assumptions`；
3. 重置 `task_index` 和 `retries`，并追加一条 `TaskExecutionResult(kind="replan")`。

若重规划失败，则回退为 `skip` 策略，同时记录 warning。

## Confidence 体系

- `TaskExecutionResult.confidence` 表示该任务输出的可信度（简单启发式）；
- reasoning 结果也会附带 `confidence`；
- 当某个任务输出置信度过低时，Manager 会自动触发 reasoning 判断是否需要重试或重新规划。

## 如何接入真实工具

1. **ToolRegistry**：将 `_mock_financial_data` / `_mock_generic_tool` 替换为真实数据调用；需要 LLM reasoning 时，将 `enable_reasoning_llm=True` 并提供 `ReasoningTool` 实现。
2. **Confidence 计算**：在 `_execute_task` 中使用真实的校验或指标来生成 `confidence`，而不是固定值。
3. **Replan 参数**：如果有更细粒度的局部 replan 机制，可改写 `DecomposeAgent.replan_factor` 或在 `ReasoningTool` 中返回额外指令。

## CLI 使用

```bash
python excutors/langgraph_plan_executor.py path/to/plan.json
```

带选项示例（关闭 LLM）：

```python
from excutors.langgraph_plan_executor import LangGraphPlanExecutor, ToolRegistry, load_plan_from_json
plan = load_plan_from_json(".../plan.json")
executor = LangGraphPlanExecutor(tool_registry=ToolRegistry(enable_reasoning_llm=False))
result = asyncio.run(executor.run_plan(plan))
```

默认输出包含每个因子所有任务的写入、细节、confidence 和 reasoning 轨迹，便于上层综合器进一步处理。
