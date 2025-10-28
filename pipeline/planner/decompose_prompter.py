# planner/decompose_prompter.py
import json
from typing import Tuple, Dict, Any
from core.schemas.PredictionTask import PredictionTask
from core.schemas.knowhow_base import BaseKnowHow


DECOMPOSE_OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["global_assumptions", "placeholders", "factors"],
    "properties": {
        "global_assumptions": {"type": "string"},
        "placeholders": {
            "type": "object",
            "description": "Document variables like ${ticker}, ${target_date}, ${sector_etf}."
        },
        "factors": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["factor_name", "tasks", "decision_logic", "output_fields"],
                "properties": {
                    "factor_name": {"type": "string"},
                    "variables": {"type": "object"},
                    "notes": {"type": "string"},
                    "decision_logic": {"type": "string"},
                    "output_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1
                    },
                    "tasks": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["id", "kind", "goal", "writes"],
                            "properties": {
                                "id": {"type": "string"},
                                "kind": {"type": "string", "enum": ["fetch","compute","extract","judge","write"]},
                                "goal": {"type": "string"},
                                "tool": {"type": "string"},
                                "params": {"type": "object"},
                                "compute": {"type": "string"},
                                "writes": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "additionalProperties": False
                        }
                    }
                },
                "additionalProperties": False
            }
        }
    },
    "additionalProperties": False
}


def build_decompose_prompt(task: PredictionTask, knowhow: BaseKnowHow,
                           json_schema: Dict[str, Any] = DECOMPOSE_OUTPUT_JSON_SCHEMA) -> Tuple[str, str]:
    """
    返回 (system_prompt, user_payload_json)
    - 去掉 aggregation
    - allowed_tools 只作为接口位（可为 {}）
    - 强化 tasks 规范：每步要么拉数据，要么计算/判断/写结果，且明确写哪些输出字段
    """
    seed = knowhow.plan_seed_for_planner(include_aggregation=False)  # 新增参数，见 knowhow_base.py 改动

    system_prompt = (
        "You are the Decompose Agent (Planner). "
        "Compile a machine-executable plan from the task and know-how. "
        "Do NOT execute tools. Do NOT summarize.\n\n"
        "STRICT RULES:\n"
        "• Output = ONE JSON object following the provided JSON Schema (no Markdown).\n"
        "• For each factor, produce a sequence of 'tasks' (fetch/compute/extract/judge/write).\n"
        "• Every output field listed in the factor's output_contract MUST be written by at least one task via 'writes'.\n"
        "• Be explicit in computations: define formulas, transformations, thresholds in 'compute'.\n"
        "• When fetching/reading, specify 'tool' and 'params' (queries, ranges, intervals, selectors, etc.).\n"
        "• Use placeholders like ${ticker}, ${target_date} and document them in 'placeholders'.\n"
        "• Keep steps minimal but sufficient for a SubAgent to follow deterministically."
    )

    user_payload = {
        "task": {
            "task_id": getattr(task, "task_id", ""),
            "task_type": getattr(task, "task_type", knowhow.task_type),
            "target_entity": getattr(task, "target_entity", ""),
            "horizon": getattr(task, "horizon", ""),
            "question": getattr(task, "question", ""),
            "constraints": getattr(task, "constraints", {})
        },
        "knowhow_seed": seed, # 全部的factos
        "allowed_tools": knowhow.allowed_tools,  # 仅作为接口位；尚未建设也可传 {}
        "json_schema": json_schema,
        "generation_notes": {
            "tie_tasks_to_output_contract": "Every output_contract key must appear in some task.writes.",
            "examples_style": "Prefer explicit formulas like r_5d = close_0/close_-5 - 1; RSI_14_norm = (RSI14-50)/50 clipped to [-1,1].",
            "web_search_goals": "State query intent, topic keywords and desired sources (official, filings, reputable media).",
            "valuation_rules": "Z-scores vs 52w mean/std; fallback rules when coverage is insufficient."
        }
    }
    return system_prompt, json.dumps(user_payload, ensure_ascii=False, indent=2)
