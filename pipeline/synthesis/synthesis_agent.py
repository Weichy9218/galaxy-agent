from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.schemas.plan import DecompositionPlan, FactorExecutionPlan, TaskItem
from core.schemas.knowhow_base import BaseKnowHow


OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["narrative", "final_answer", "blended_expected_return", "blended_confidence"],
    "properties": {
        "narrative": {"type": "string"},
        "blended_expected_return": {"type": ["number", "null"]},
        "blended_confidence": {"type": ["number", "null"]},
        "final_answer": {"type": "string"},
        "supporting_points": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "additionalProperties": False,
}


@dataclass
class FactorContribution:
    name: str
    weight: float
    outputs: Dict[str, Any]
    decision_logic: str
    summary: str
    raw_tasks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SynthesisResult:
    task_id: str
    blended_expected_return: Optional[float]
    blended_confidence: Optional[float]
    final_answer: str
    narrative: str
    supporting_points: List[str]
    factor_contributions: List[FactorContribution]
    raw_response: str


class SynthesisAgent:
    """
    LLM-based synthesis agent.

    输入：DecompositionPlan、执行结果（来自 SubAgent）、KnowHow。
    输出：最终 narrative + prediction（遵循 answer_instructions）。
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    async def async_synthesize(
        self,
        plan: DecompositionPlan,
        execution: Dict[str, Any],
        knowhow: BaseKnowHow,
    ) -> SynthesisResult:
        system, user = build_synthesis_prompt(plan, execution, knowhow, OUTPUT_SCHEMA)
        raw = await self._call_llm_async(system, user)
        obj = parse_json(raw)
        contributions = build_contributions(plan, execution, knowhow)
        return SynthesisResult(
            task_id=plan.task.task_id,
            blended_expected_return=_json_number(obj.get("blended_expected_return")),
            blended_confidence=_json_number(obj.get("blended_confidence")),
            final_answer=obj.get("final_answer", "").strip(),
            narrative=obj.get("narrative", "").strip(),
            supporting_points=[p.strip() for p in obj.get("supporting_points", [])],
            factor_contributions=contributions,
            raw_response=raw,
        )

    def synthesize(
        self,
        plan: DecompositionPlan,
        execution: Dict[str, Any],
        knowhow: BaseKnowHow,
    ) -> SynthesisResult:
        system, user = build_synthesis_prompt(plan, execution, knowhow, OUTPUT_SCHEMA)
        raw = self._call_llm(system, user)
        obj = parse_json(raw)
        contributions = build_contributions(plan, execution, knowhow)
        return SynthesisResult(
            task_id=plan.task.task_id,
            blended_expected_return=_json_number(obj.get("blended_expected_return")),
            blended_confidence=_json_number(obj.get("blended_confidence")),
            final_answer=obj.get("final_answer", "").strip(),
            narrative=obj.get("narrative", "").strip(),
            supporting_points=[p.strip() for p in obj.get("supporting_points", [])],
            factor_contributions=contributions,
            raw_response=raw,
        )

    def _call_llm(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = self.llm.chat(messages=messages, temperature=0)
        if hasattr(result, "content"):
            return result.content
        return str(result)

    async def _call_llm_async(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = await self.llm.chat(messages=messages, temperature=0)
        if hasattr(result, "content"):
            return result.content
        return str(result)


def build_synthesis_prompt(
    plan: DecompositionPlan,
    execution: Dict[str, Any],
    knowhow: BaseKnowHow,
    output_schema: Dict[str, Any],
) -> Tuple[str, str]:
    aggregation = knowhow.decomposition.aggregation
    synthesis_contract = knowhow.synthesis_contract()
    plan_summary = summarise_plan(plan)

    payload = {
        "task": {
            "task_id": plan.task.task_id,
            "event_description": plan.task.event_description,
            "resolved_time": plan.task.resolved_time,
            "target_entity": plan.task.target_entity,
            "target_metric": plan.task.target_metric,
            "task_type": plan.task.task_type,
            "answer_instructions": plan.task.answer_instructions,
        },
        "global_assumptions": plan.global_assumptions,
        "aggregation": {
            "initial_weights": aggregation.initial_weights if aggregation else {},
            "approach": aggregation.approach if aggregation else "",
            "notes": aggregation.formula_note if aggregation else "",
        },
        "synthesis_contract": synthesis_contract,
        "plan_summary": plan_summary,
        "execution": execution.get("factor_results", []),
        "raw_execution": execution,
        "output_schema": output_schema,
    }

    system_prompt = (
        "You are the Synthesis Agent. Combine factor-level analysis into a final forecast.\n"
        "Follow the output JSON schema exactly. Respect answer instructions (e.g. boxed format).\n"
        "Fill numerical fields with null when unsupported rather than guessing.\n"
        "Explain key drivers in the narrative and supporting_points."
    )

    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    return system_prompt, user_prompt


def summarise_plan(plan: DecompositionPlan) -> List[Dict[str, Any]]:
    summaries = []
    for factor in plan.factors:
        summaries.append(
            {
                "factor_name": factor.factor_name,
                "decision_logic": factor.decision_logic,
                "output_fields": factor.output_fields,
                "notes": factor.notes,
            }
        )
    return summaries


def build_contributions(
    plan: DecompositionPlan,
    execution: Dict[str, Any],
    knowhow: BaseKnowHow,
) -> List[FactorContribution]:
    factor_map = {f["factor_name"]: f for f in execution.get("factor_results", [])}
    aggregation = knowhow.decomposition.aggregation
    weights = aggregation.initial_weights if aggregation else None

    contributions: List[FactorContribution] = []
    for factor in plan.factors:
        weight = (weights or {}).get(factor.factor_name)
        if weight is None:
            weight = 1.0
        result = factor_map.get(factor.factor_name, {})
        contributions.append(
            FactorContribution(
                name=factor.factor_name,
                weight=weight,
                outputs=result.get("outputs", {}),
                decision_logic=factor.decision_logic,
                summary=result.get("summary", ""),
                raw_tasks=result.get("tasks", []),
            )
        )
    return contributions


def parse_json(text: str) -> Dict[str, Any]:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        snippet = text[start:end]
        return json.loads(snippet)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Synthesis Agent output not JSON: {text}") from exc


def _json_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        return _to_float(value)
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        match = re.match(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$", cleaned)
        if match:
            return float(cleaned)
        if cleaned.endswith("%"):
            try:
                return float(cleaned[:-1]) / 100.0
            except ValueError:
                return None
        inner = cleaned
        if "(" in cleaned and cleaned.endswith(")"):
            inner = cleaned.split("(", 1)[-1][:-1]
            return _to_float(inner)
    return None
