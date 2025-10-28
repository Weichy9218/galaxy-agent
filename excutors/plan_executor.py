# excutors/plan_executor.py
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.schemas.PredictionTask import PredictionTask
from core.schemas.plan import (
    DecompositionPlan,
    FactorExecutionPlan,
    TaskItem,
)


@dataclass
class TaskExecutionResult:
    task_id: str
    kind: str
    writes: Dict[str, Any]
    detail: str


@dataclass
class FactorExecutionResult:
    factor_name: str
    outputs: Dict[str, Any]
    summary: str
    decision_logic: str
    tasks: List[TaskExecutionResult] = field(default_factory=list)


class ToolRegistry:
    """
    极简工具注册表，为当前 demo 提供 mock 能力。
    真正落地时可替换成真实的数据拉取/搜索实现。
    """

    def __init__(self):
        self._tools = {
            "financial_data": self._mock_financial_data,
            "web_search": self._mock_generic_tool,
            "reading": self._mock_generic_tool,
            "reasoning": self._mock_generic_tool,
            "structured_data": self._mock_generic_tool,
        }

    async def call(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        handler = self._tools.get(name, self._mock_generic_tool)
        return await handler(params)

    async def _mock_generic_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": f"mock_result", "meta": params}

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


class PlanExecutor:
    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.tools = tool_registry or ToolRegistry()

    async def run_plan(self, plan: DecompositionPlan) -> Dict[str, Any]:
        factor_tasks = [
            self._execute_factor(plan, factor)
            for factor in plan.factors
        ]
        factor_results = await asyncio.gather(*factor_tasks)
        global_summary = self._build_global_summary(plan, factor_results)
        return {
            "task_id": plan.task.task_id,
            "factor_results": [self._factor_result_to_dict(result) for result in factor_results],
            "global_summary": global_summary,
        }

    async def _execute_factor(
        self,
        plan: DecompositionPlan,
        factor: FactorExecutionPlan,
    ) -> FactorExecutionResult:
        context: Dict[str, Any] = dict(plan.placeholders)
        task_results: List[TaskExecutionResult] = []

        for task in factor.tasks:
            writes, detail = await self._execute_task(task, context)
            context.update(writes)
            task_results.append(
                TaskExecutionResult(task_id=task.id, kind=task.kind, writes=writes, detail=detail)
            )

        outputs: Dict[str, Any] = {}
        for field in factor.output_fields:
            outputs[field] = context.get(field, "<unfilled>")

        summary = self._build_factor_summary(factor, outputs)
        return FactorExecutionResult(
            factor_name=factor.factor_name,
            outputs=outputs,
            summary=summary,
            decision_logic=factor.decision_logic,
            tasks=task_results,
        )

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
                }
                for task in result.tasks
            ],
        }

    async def _execute_task(
        self,
        task: TaskItem,
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        writes: Dict[str, Any] = {}
        detail: str

        if task.kind == "fetch" and task.tool:
            params = _resolve_params(task.params, context)
            result = await self.tools.call(task.tool, params)
            for name in task.writes:
                writes[name] = result.get(name, result)
            detail = f"Fetched data via {task.tool}"
        elif task.kind == "compute":
            detail = f"Computed: {task.compute or 'manual rules'}"
            for name in task.writes:
                writes[name] = f"computed({name})"
        elif task.kind == "judge":
            detail = f"Judged with rule: {task.compute or 'N/A'}"
            for name in task.writes:
                writes[name] = f"judged({name})"
        elif task.kind == "write":
            detail = "Recorded final outputs"
            for name in task.writes:
                writes[name] = f"written({name})"
        else:
            detail = "Task type not recognized, copied instructions."
            for name in task.writes:
                writes[name] = f"unhandled({name})"

        return writes, detail

    def _build_factor_summary(
        self,
        factor: FactorExecutionPlan,
        outputs: Dict[str, Any],
    ) -> str:
        lines = [f"因子 {factor.factor_name} 输出："]
        for key, value in outputs.items():
            lines.append(f"- {key}: {value}")
        lines.append(f"决策逻辑: {factor.decision_logic}")
        return "\n".join(lines)

    def _build_global_summary(
        self,
        plan: DecompositionPlan,
        results: List[FactorExecutionResult],
    ) -> str:
        lines = [
            f"任务 {plan.task.task_id} 的因子执行完成，基于假设：{plan.global_assumptions or '无'}。",
            "各因子结果概览：",
        ]
        for res in results:
            highlight = res.outputs.get("expected_return") or res.outputs.get("recommended_option")
            lines.append(f"- {res.factor_name}: 关键输出 {highlight}")
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


async def run_plan_from_file(path: str) -> Dict[str, Any]:
    plan = load_plan_from_json(path)
    executor = PlanExecutor()
    return await executor.run_plan(plan)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run decomposition plan.")
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
