import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from langgraph.graph import StateGraph, END

if TYPE_CHECKING:
    from pipeline.planner.decompose_agent import DecomposeAgent
    from core.schemas.knowhow_base import BaseKnowHow

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.schemas.PredictionTask import PredictionTask
from core.schemas.plan import (
    DecompositionPlan,
    FactorExecutionPlan,
    TaskItem,
)
from core.tools import ReasoningTool


@dataclass
class TaskExecutionResult:
    task_id: str
    kind: str
    writes: Dict[str, Any]
    detail: str
    confidence: float


@dataclass
class FactorExecutionResult:
    factor_name: str
    outputs: Dict[str, Any]
    summary: str
    decision_logic: str
    tasks: List[TaskExecutionResult] = field(default_factory=list)


@dataclass
class FactorState:
    plan: DecompositionPlan
    factor: FactorExecutionPlan
    context: Dict[str, Any] = field(default_factory=dict)
    task_results: List[TaskExecutionResult] = field(default_factory=list)
    task_index: int = 0
    retries: Dict[str, int] = field(default_factory=dict)
    max_retries: int = 1
    last_error: Optional[str] = None
    status: str = "manager"
    failed_task: Optional[TaskItem] = None
    current_task: Optional[TaskItem] = None
    replan_attempted: bool = False


class ToolRegistry:
    def __init__(
        self,
        reasoning_tool: Optional[ReasoningTool] = None,
        enable_reasoning_llm: bool = False,
    ):
        self._tools = {
            "financial_data": self._mock_financial_data,
            "web_search": self._mock_generic_tool,
            "reading": self._mock_generic_tool,
            "reasoning": self._reasoning_dispatch,
            "structured_data": self._mock_generic_tool,
        }
        if enable_reasoning_llm:
            try:
                self._reasoning_tool = reasoning_tool or ReasoningTool()
                self._reasoning_available = True
            except Exception as exc:  # noqa: BLE001
                self._reasoning_tool = None
                self._reasoning_available = False
                self._reasoning_init_error = str(exc)
        else:
            self._reasoning_tool = None
            self._reasoning_available = False
            self._reasoning_init_error = "LLM reasoning disabled"

    async def call(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        handler = self._tools.get(name, self._mock_generic_tool)
        return await handler(params)

    async def _mock_generic_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": "mock_result", "meta": params}

    async def _mock_financial_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        ticker = params.get("ticker", "UNKNOWN")
        fields = params.get("fields") or []
        size = _estimate_series_length(params.get("range"), params.get("interval"))
        base_value = 100 + (abs(hash(ticker)) % 50)
        series = [round(base_value * (1 + 0.002 * i), 2) for i in range(size)]
        data: Dict[str, Any] = {}
        for field in fields:
            data[field] = list(series)
        if "sector" in fields:
            data["sector"] = f"MockSector#{abs(hash(ticker)) % 5}"
        if "beta" in fields:
            data["beta"] = round(1 + (abs(hash(ticker)) % 30) / 100, 2)
        return data

    async def _reasoning_dispatch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._reasoning_available and self._reasoning_tool is not None:
            return await self._reasoning_tool.run(params)

        error = params.get("error", "")
        retries = params.get("retries", 0)
        decision = "retry" if retries < 1 else "skip"
        explanation = (
            f"Fallback reasoning - LLM unavailable ({getattr(self, '_reasoning_init_error', 'unknown')}), "
            f"error='{error}', decision={decision}."
        )
        return {"decision": decision, "confidence": 0.3, "analysis": explanation}


class LangGraphPlanExecutor:
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        planner: Optional["DecomposeAgent"] = None,
        knowhow: Optional["BaseKnowHow"] = None,
        confidence_threshold: float = 0.6,
        enable_reasoning_llm: bool = False,
    ):
        self.tools = tool_registry or ToolRegistry(enable_reasoning_llm=enable_reasoning_llm)
        self.planner = planner
        self.knowhow = knowhow
        self.confidence_threshold = confidence_threshold
        self.enable_reasoning_llm = enable_reasoning_llm

    async def run_plan(self, plan: DecompositionPlan) -> Dict[str, Any]:
        factor_tasks = [
            self._run_factor_graph(plan, factor)
            for factor in plan.factors
        ]
        factor_results = await asyncio.gather(*factor_tasks)
        summary = self._build_global_summary(plan, factor_results)
        return {
            "task_id": plan.task.task_id,
            "factor_results": [self._factor_result_to_dict(res) for res in factor_results],
            "global_summary": summary,
        }

    async def _run_factor_graph(
        self,
        plan: DecompositionPlan,
        factor: FactorExecutionPlan,
    ) -> FactorExecutionResult:
        workflow = StateGraph(FactorState)
        workflow.add_node("manager", self._make_manager_node(factor))
        workflow.add_node("execute_task", self._make_execute_node())
        workflow.add_node("handle_error", self._make_error_node())
        workflow.set_entry_point("manager")

        workflow.add_conditional_edges(
            "manager",
            lambda state: _route_from_manager(state),
            {
                "done": END,
                "execute": "execute_task",
                "error": "handle_error",
            },
        )
        workflow.add_edge("execute_task", "manager")
        workflow.add_edge("handle_error", "manager")

        app = workflow.compile()
        initial_context = dict(plan.placeholders)
        state = FactorState(plan=plan, factor=factor, context=initial_context)
        raw_state = await app.ainvoke(state)
        final_state = _ensure_factor_state(raw_state)

        outputs = {
            field: final_state.context.get(field, "<unfilled>")
            for field in factor.output_fields
        }
        summary = self._build_factor_summary(factor, outputs)
        return FactorExecutionResult(
            factor_name=factor.factor_name,
            outputs=outputs,
            summary=summary,
            decision_logic=factor.decision_logic,
            tasks=final_state.task_results,
        )

    def _make_manager_node(self, factor: FactorExecutionPlan):
        def node(state: FactorState) -> FactorState:
            state.factor = factor
            if state.task_index >= len(factor.tasks):
                state.status = "done"
                return state
            if state.status == "error":
                return state
            state.current_task = factor.tasks[state.task_index]
            state.status = "execute"
            return state

        return node

    def _make_execute_node(self):
        async def node(state: FactorState) -> FactorState:
            if not state.current_task:
                state.status = "done"
                return state
            task = state.current_task
            try:
                writes, detail, confidence = await self._execute_task(task, state.context)
                state.context.update(writes)
                state.task_results.append(
                    TaskExecutionResult(
                        task_id=task.id,
                        kind=task.kind,
                        writes=writes,
                        detail=detail,
                        confidence=confidence,
                    )
                )
                if confidence < self.confidence_threshold:
                    state.last_error = f"Confidence {confidence:.2f} below threshold"
                    state.status = "error"
                    state.failed_task = task
                    state.retries[task.id] = state.retries.get(task.id, 0) + 1
                else:
                    state.task_index += 1
                    state.status = "manager"
                    state.last_error = None
                    state.failed_task = None
            except Exception as exc:  # noqa: BLE001
                state.last_error = str(exc)
                state.status = "error"
                state.failed_task = task
                state.retries[task.id] = state.retries.get(task.id, 0) + 1
            return state

        return node

    def _make_error_node(self):
        async def node(state: FactorState) -> FactorState:
            task = state.failed_task
            if task is None:
                state.status = "manager"
                return state

            decision, explanation, reason_conf = await self._reason_about_failure(state, task)
            state.task_results.append(
                TaskExecutionResult(
                    task_id=f"{task.id}_reasoning",
                    kind="reasoning",
                    writes={},
                    detail=explanation,
                    confidence=reason_conf,
                )
            )

            if decision == "retry" and state.retries.get(task.id, 0) <= state.max_retries:
                state.status = "execute"
            elif decision == "replan" and not state.replan_attempted:
                replanned = await self._attempt_replan(state)
                if replanned:
                    state.replan_attempted = True
                    state.status = "manager"
                    state.failed_task = None
                    state.last_error = None
                else:
                    state.task_results.append(
                        TaskExecutionResult(
                            task_id=f"{task.id}_replan_failed",
                            kind="warning",
                            writes={},
                            detail="Replan attempt failed; fallback to skip.",
                            confidence=0.4,
                        )
                    )
                    state.task_index += 1
                    state.status = "manager"
                    state.failed_task = None
                    state.last_error = None
            else:
                writes = {field: f"error({state.last_error})" for field in task.writes}
                state.context.update(writes)
                state.task_results.append(
                    TaskExecutionResult(
                        task_id=f"{task.id}_skipped",
                        kind="error",
                        writes=writes,
                        detail=f"任务失败，原因: {state.last_error}; decision={decision}",
                        confidence=0.1,
                    )
                )
                state.task_index += 1
                state.status = "manager"
                state.failed_task = None
                state.last_error = None
            return state

        return node

    async def _execute_task(
        self,
        task: TaskItem,
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str, float]:
        writes: Dict[str, Any] = {}
        detail: str
        confidence: float

        if task.kind == "fetch" and task.tool:
            params = _resolve_params(task.params, context)
            result = await self.tools.call(task.tool, params)
            for name in task.writes:
                writes[name] = result.get(name, result)
            detail = f"调用工具 {task.tool} 获得数据"
            confidence = 0.9
        elif task.kind == "compute":
            detail = f"按照规则计算: {task.compute or '无'}"
            for name in task.writes:
                writes[name] = f"computed({name})"
            confidence = 0.75
        elif task.kind == "judge":
            detail = f"判断逻辑: {task.compute or '无'}"
            for name in task.writes:
                writes[name] = f"judged({name})"
            confidence = 0.7
        elif task.kind == "write":
            detail = "写入最终输出字段"
            for name in task.writes:
                writes[name] = f"written({name})"
            confidence = 0.8
        else:
            detail = "未识别的任务类型"
            for name in task.writes:
                writes[name] = f"unhandled({name})"
            confidence = 0.3

        return writes, detail, confidence

    async def _reason_about_failure(
        self,
        state: FactorState,
        task: TaskItem,
    ) -> Tuple[str, str, float]:
        params = {
            "error": state.last_error,
            "task_id": task.id,
            "goal": task.goal,
            "retries": state.retries.get(task.id, 0),
            "history": [result.detail for result in state.task_results[-3:]],
        }
        result = await self.tools.call("reasoning", params)
        decision = result.get("decision", "skip")
        explanation = (
            result.get("analysis")
            or result.get("explanation")
            or "Reasoning completed without analysis."
        )
        reason_confidence = float(result.get("confidence", 0.5))
        return decision, explanation, reason_confidence

    @staticmethod
    def _factor_result_to_dict(result: FactorExecutionResult) -> Dict[str, Any]:
        return {
            "factor_name": result.factor_name,
            "outputs": result.outputs,
            "summary": result.summary,
            "decision_logic": result.decision_logic,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "kind": task.kind,
                    "writes": task.writes,
                    "detail": task.detail,
                    "confidence": task.confidence,
                }
                for task in result.tasks
            ],
        }

    def _build_factor_summary(
        self,
        factor: FactorExecutionPlan,
        outputs: Dict[str, Any],
    ) -> str:
        lines = [f"因子 {factor.factor_name} 输出:"]
        for key, value in outputs.items():
            lines.append(f"- {key}: {value}")
        lines.append(f"决策逻辑: {factor.decision_logic}")
        return "\n".join(lines)

    async def _attempt_replan(self, state: FactorState) -> bool:
        if self.planner is None or self.knowhow is None:
            return False

        factor_name = state.factor.factor_name

        def _sync_replan() -> Tuple[Optional[FactorExecutionPlan], Dict[str, Any], str]:
            new_plan = self.planner.plan(state.plan.task, self.knowhow)
            matched = None
            for fac in new_plan.factors:
                if fac.factor_name == factor_name:
                    matched = fac
                    break
            return matched, new_plan.placeholders, new_plan.global_assumptions

        try:
            loop = asyncio.get_running_loop()
            new_factor, placeholders, global_assumptions = await loop.run_in_executor(
                None, _sync_replan
            )
        except Exception:  # noqa: BLE001
            return False

        if not new_factor:
            return False

        state.factor = new_factor
        state.task_index = 0
        state.current_task = None
        state.failed_task = None
        state.context = dict(placeholders)
        state.plan.placeholders = placeholders
        state.plan.global_assumptions = global_assumptions
        state.retries = {}
        state.last_error = None
        state.task_results.append(
            TaskExecutionResult(
                task_id=f"{factor_name}_replan",
                kind="replan",
                writes={},
                detail="因子任务重新规划，开始新一轮执行。",
                confidence=0.85,
            )
        )
        return True

    def _build_global_summary(
        self,
        plan: DecompositionPlan,
        results: List[FactorExecutionResult],
    ) -> str:
        lines = [
            f"任务 {plan.task.task_id} 完成，假设: {plan.global_assumptions or '无'}",
            "因子概览:",
        ]
        for res in results:
            key_val = res.outputs.get("expected_return") or res.outputs.get("recommended_option")
            lines.append(f"- {res.factor_name}: 关键输出 {key_val}")
        return "\n".join(lines)


def load_plan_from_json(path: str) -> DecompositionPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    task_info = payload["task"]
    metadata = {}
    if task_info.get("resolved_time"):
        metadata["end_time"] = task_info["resolved_time"]
    task = PredictionTask(
        task_id=task_info["task_id"],
        task_question=task_info.get("event_description", "Plan-derived task"),
        metadata=metadata,
        event_description=task_info.get("event_description"),
        resolved_time=task_info.get("resolved_time"),
        task_type=task_info.get("task_type"),
        target_entity=task_info.get("target_entity"),
        target_metric=task_info.get("target_metric"),
        answer_instructions=task_info.get("answer_instructions"),
    )

    factors: List[FactorExecutionPlan] = []
    for factor_data in payload.get("factors", []):
        tasks = [
            TaskItem(
                id=task_data["id"],
                kind=task_data["kind"],
                goal=task_data["goal"],
                tool=task_data.get("tool"),
                params=task_data.get("params", {}) or {},
                compute=task_data.get("compute"),
                writes=task_data.get("writes", []) or [],
            )
            for task_data in factor_data.get("tasks", [])
        ]
        factors.append(
            FactorExecutionPlan(
                factor_name=factor_data["factor_name"],
                tasks=tasks,
                decision_logic=factor_data.get("decision_logic", ""),
                output_fields=factor_data.get("output_fields", []) or [],
                variables=factor_data.get("variables", {}) or {},
                notes=factor_data.get("notes", "") or "",
            )
        )

    return DecompositionPlan(
        task=task,
        factors=factors,
        global_assumptions=payload.get("global_assumptions", ""),
        placeholders=payload.get("placeholders", {}) or {},
    )


def _resolve_params(raw_params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}

    def _resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith("${") and value.endswith("}"):
                return context.get(value, value)
            if "${" in value:
                return _substitute_placeholders(value, context)
        if isinstance(value, list):
            return [_resolve_value(v) for v in value]
        if isinstance(value, dict):
            return _resolve_params(value, context)
        return value

    for key, value in raw_params.items():
        resolved[key] = _resolve_value(value)
    return resolved


PLACEHOLDER_PATTERN = re.compile(r"\$\{[^}]+\}")


def _substitute_placeholders(text: str, context: Dict[str, Any]) -> str:
    def replace(match: re.Match) -> str:
        placeholder = match.group(0)
        return str(context.get(placeholder, placeholder))

    return PLACEHOLDER_PATTERN.sub(replace, text)


def _estimate_series_length(range_value: Optional[str], interval: Optional[str]) -> int:
    if not isinstance(range_value, str):
        return 10
    digits = "".join(ch for ch in range_value if ch.isdigit())
    if digits.isdigit():
        return max(int(digits), 5)
    if "y" in range_value.lower():
        return 252 if interval == "1d" else 52
    return 10


def _ensure_factor_state(value: Any) -> FactorState:
    if isinstance(value, FactorState):
        return value
    if isinstance(value, dict):
        task_entries: List[TaskExecutionResult] = []
        for item in value.get("task_results", []):
            if isinstance(item, TaskExecutionResult):
                task_entries.append(item)
            elif isinstance(item, dict):
                task_entries.append(
                    TaskExecutionResult(
                        task_id=item.get("task_id", ""),
                        kind=item.get("kind", ""),
                        writes=dict(item.get("writes", {})),
                        detail=item.get("detail", ""),
                        confidence=float(item.get("confidence", 0.0)),
                    )
                )
        if not task_entries:
            task_entries = []
        failed_task_raw = value.get("failed_task")
        failed_task = _restore_task_item(failed_task_raw)
        current_task_raw = value.get("current_task")
        current_task = _restore_task_item(current_task_raw)
        return FactorState(
            plan=value["plan"],
            factor=value["factor"],
            context=dict(value.get("context", {})),
            task_results=task_entries,
            task_index=value.get("task_index", 0),
            retries=dict(value.get("retries", {})),
            max_retries=value.get("max_retries", 1),
            last_error=value.get("last_error"),
            status=value.get("status", "manager"),
            failed_task=failed_task,
            current_task=current_task,
            replan_attempted=value.get("replan_attempted", False),
        )
    raise TypeError(f"Unexpected state type: {type(value)}")


def _route_from_manager(state: FactorState) -> str:
    if state.status == "done":
        return "done"
    if state.status == "error":
        return "error"
    return "execute"


def _restore_task_item(data: Any) -> Optional[TaskItem]:
    if data is None or isinstance(data, TaskItem):
        return data
    if isinstance(data, dict):
        return TaskItem(
            id=data["id"],
            kind=data["kind"],
            goal=data.get("goal", ""),
            tool=data.get("tool"),
            params=data.get("params", {}) or {},
            compute=data.get("compute"),
            writes=data.get("writes", []) or [],
        )
    return None


async def run_plan_from_file(path: str) -> Dict[str, Any]:
    plan = load_plan_from_json(path)
    executor = LangGraphPlanExecutor()
    return await executor.run_plan(plan)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run decomposition plan with LangGraph.")
    parser.add_argument("plan_path", help="Path to plan JSON file")
    parser.add_argument("--output", help="Where to save execution results")
    args = parser.parse_args()

    result = asyncio.run(run_plan_from_file(args.plan_path))
    output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Execution result saved to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
