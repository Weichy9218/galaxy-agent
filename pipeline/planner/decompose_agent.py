# planner/decompose_agent.py
import json
import re
import asyncio
from typing import List, Dict, Any, Tuple, Optional

from core.schemas.PredictionTask import PredictionTask
from core.schemas.knowhow_base import BaseKnowHow
from core.schemas.plan import (
    DecompositionPlan,
    FactorExecutionPlan,
    TaskItem,
    validate_plan,
)
from pipeline.planner.decompose_prompter import (
    build_decompose_prompt,
    DECOMPOSE_OUTPUT_JSON_SCHEMA,
)

class DecomposeAgent:
    """
    Compile (PredictionTask + KnowHow) -> DecompositionPlan
    约定 llm_client.chat(messages=[...], **kwargs) -> str
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    # 便于你先审阅 prompt
    def preview_prompt(self, task: PredictionTask, knowhow: BaseKnowHow) -> str:
        system, user = build_decompose_prompt(task, knowhow, DECOMPOSE_OUTPUT_JSON_SCHEMA)
        return f"--- SYSTEM ---\n{system}\n\n--- USER ---\n{user}"

    # 让 llm 来根据system 和 user prompt 输出新的 decompose plan
    def plan(self, task: PredictionTask, knowhow: BaseKnowHow) -> DecompositionPlan:
        system, user = build_decompose_prompt(task, knowhow, DECOMPOSE_OUTPUT_JSON_SCHEMA)
        raw = self._call_llm(system, user)
        
        # print(raw)

        obj = self._extract_json(raw)

        # print(obj)

        # 映射 JSON -> dataclass
        factors: List[FactorExecutionPlan] = []
        for f in obj.get("factors", []):
            tasks = [self._task_from_json(t) for t in f.get("tasks", [])]

            fep = FactorExecutionPlan(
                factor_name=f["factor_name"],
                tasks=tasks,
                decision_logic=f.get("decision_logic", ""),
                output_fields=f.get("output_fields", []),
                variables=f.get("variables", {}) or {},
                notes=f.get("notes", "") or "",
            )
            factors.append(fep)

        self._ensure_output_coverage(factors, knowhow)

        plan = DecompositionPlan(
            task=task,
            factors=factors,
            global_assumptions=obj.get("global_assumptions", "") or "",
            placeholders=obj.get("placeholders", {}) or {},
        )

        # 轻量校验（只抛一次错误信息，后续你可加重试/自愈模块）
        errors = validate_plan(plan, knowhow)
        if errors:
            raise ValueError("DecompositionPlan validation failed: " + " | ".join(errors))
        return plan

    # ---------------- internal helpers ----------------

    def _call_llm(self, system: str, user: str) -> str:
        """
        统一 LLM 接口调用
        支持同步和异步 LLM 客户端
        返回 LLM 响应的文本内容
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        
        # 调用 LLM
        result = self.llm.chat(messages=messages, temperature=0)

        if asyncio.iscoroutine(result):
            result = asyncio.run(result)

        if hasattr(result, "content"):
            return result.content

        return str(result)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        提取首个 JSON 对象（容忍 ```json ... ``` 包裹）
        """
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError("LLM did not return a JSON object.")
        return json.loads(m.group(0))

    def _task_from_json(self, t: Dict[str, Any]) -> TaskItem:
        """
        宽松映射；不做复杂异常处理，保持紧凑。
        必填: id/kind/goal/writes；可选: tool/params/compute
        """
        return TaskItem(
            id=t["id"],
            kind=t["kind"],
            goal=t["goal"],
            tool=t.get("tool"),
            params=t.get("params", {}) or {},
            compute=t.get("compute"),
            writes=t.get("writes", []) or [],
        )

    def _ensure_output_coverage(
        self,
        factors: List[FactorExecutionPlan],
        knowhow: BaseKnowHow,
    ) -> None:
        """
        若 LLM 漏写某些 output_contract 字段，自动补充占位 write 任务，
        以避免计划验证阶段直接失败。后续执行器可视情况替换这些占位任务。
        """
        for factor in factors:
            spec = knowhow.factor_by_name(factor.factor_name)
            if not spec:
                continue

            spec_keys = list(spec.output_schema.keys())
            factor.output_fields = spec_keys

            existing_ids = {task.id for task in factor.tasks}
            written = {field for task in factor.tasks for field in task.writes}
            missing = [key for key in spec_keys if key not in written]

            for key in missing:
                task_id = self._generate_unique_task_id(existing_ids, prefix=f"auto_{key}")
                existing_ids.add(task_id)
                factor.tasks.append(
                    TaskItem(
                        id=task_id,
                        kind="write",
                        goal=f"补充写入 {key}（自动生成）",
                        tool=None,
                        params={},
                        compute=f"总结前序步骤结果，填充 {key} 字段。",
                        writes=[key],
                    )
                )

    @staticmethod
    def _generate_unique_task_id(existing_ids: set, prefix: str) -> str:
        base = prefix.replace(" ", "_")
        if base not in existing_ids:
            return base
        idx = 1
        candidate = f"{base}_{idx}"
        while candidate in existing_ids:
            idx += 1
            candidate = f"{base}_{idx}"
        return candidate

    def replan_factor(
        self,
        task: PredictionTask,
        knowhow: BaseKnowHow,
        factor_name: str,
    ) -> FactorExecutionPlan:
        """重新规划单个因子。当前实现为整体重跑 plan 后取对应因子。"""
        plan = self.plan(task, knowhow)
        for factor in plan.factors:
            if factor.factor_name == factor_name:
                return factor
        raise ValueError(f"Factor {factor_name} not found in new plan")
