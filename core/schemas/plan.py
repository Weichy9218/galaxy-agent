# core/schemas/plan.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from core.schemas.PredictionTask import PredictionTask
from core.schemas.knowhow_base import BaseKnowHow

@dataclass
class DataCall:
    tool: str
    params: Dict[str, Any]

@dataclass
class TaskItem:
    """
    结构化的“可执行任务步”：
    - kind: fetch/compute/extract/judge/write 之一（便于执行器做分支）
    - goal: 这一步要完成什么（面向人/Agent）
    - tool: 可选；当 kind=fetch 或 extract 时通常需要
    - params: 工具/计算所需参数（允许出现占位符 ${ticker}）
    - compute: 可选；当 kind=compute/judge/write 时给出公式/规则或伪代码
    - writes: 这一步负责落地哪些 output_contract 字段（强制覆盖机制的关键）
    """
    id: str
    kind: str
    goal: str
    tool: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    compute: Optional[str] = None
    writes: List[str] = field(default_factory=list)

@dataclass
class FactorExecutionPlan:
    factor_name: str
    tasks: List[TaskItem]                      # 取代 step_by_step：更易于执行器逐条 follow
    decision_logic: str                        # 判定强弱/方向/阈值等
    output_fields: List[str]                   # 必须完全覆盖 factor.output_schema.keys()
    variables: Dict[str, Any] = field(default_factory=dict)  # 例如 {"sector_etf":"${sector_etf}"}
    notes: str = ""                            # 可选：补充说明

@dataclass
class DecompositionPlan:
    task: PredictionTask
    factors: List[FactorExecutionPlan]
    global_assumptions: str = ""
    placeholders: Dict[str, str] = field(default_factory=dict)  # 统一占位符说明（如 "${ticker}","${target_date}"）

def validate_plan(plan: DecompositionPlan, knowhow: BaseKnowHow) -> List[str]:
    errors: List[str] = []
    allowed_factor_names = set(knowhow.factor_names())
    tool_catalog = set(knowhow.allowed_tools.keys())  # 接口位：未来可由 MCP/ToolRegistry 注入替换

    if not plan.factors:
        return ["No factors in plan"]

    for f in plan.factors:
        if f.factor_name not in allowed_factor_names:
            errors.append(f"Unknown factor: {f.factor_name}")
            continue

        spec = knowhow.factor_by_name(f.factor_name)
        if not spec:
            errors.append(f"Missing spec for {f.factor_name}")
            continue

        # 输出字段应与 output_schema 完全一致（顺序不敏感）
        spec_keys = set(spec.output_schema.keys())
        got_keys = set(f.output_fields)
        if got_keys != spec_keys:
            errors.append(f"{f.factor_name}: output_fields mismatch, expect {sorted(spec_keys)}, got {sorted(got_keys)}")

        if not f.tasks:
            errors.append(f"{f.factor_name}: tasks is empty")
            continue

        # 工具白名单（因子级 + 全局 catalog）
        factor_whitelist = set(spec.tools)
        writes_covered = set()
        for t in f.tasks:
            if t.kind in ("fetch", "extract") and t.tool:
                if t.tool not in factor_whitelist:
                    errors.append(f"{f.factor_name}.{t.id}: tool '{t.tool}' not allowed for this factor (whitelist={list(factor_whitelist)})")
                if t.tool not in tool_catalog:
                    # 仅提示，不作为硬错误（工具层待定）；按你要求“保留接口位”
                    pass
            # 写入字段计数
            for k in t.writes:
                writes_covered.add(k)

        # 每个输出字段必须至少被一个 task 写入
        uncovered = spec_keys - writes_covered
        if uncovered:
            errors.append(f"{f.factor_name}: outputs not written by tasks -> {sorted(uncovered)}")

    return errors
