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
from planner.decompose_prompter import (
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

    def plan(self, task: PredictionTask, knowhow: BaseKnowHow) -> DecompositionPlan:
        system, user = build_decompose_prompt(task, knowhow, DECOMPOSE_OUTPUT_JSON_SCHEMA)
        raw = self._call_llm(system, user)
        obj = self._extract_json(raw)

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
        
        # 如果是协程（异步），则同步执行
        if asyncio.iscoroutine(result):
            result = asyncio.run(result)
        
        # 提取内容（处理 LLMResponse 对象或字符串）
        if hasattr(result, 'content'):
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
