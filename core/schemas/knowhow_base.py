# core/schemas/knowhow_base.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class FactorSpec:
    name: str
    description: str
    agent_role: str
    tools: List[str]
    analysis_steps: List[str]                 # 仍保留：给 Planner 的“知识要点/提示”
    output_schema: Dict[str, Any]             # 契约：SubAgent 最终要交付的字段
    task_hints: List[str] = field(default_factory=list)  # 新增：更细的任务提示（如“写入expected_return”）
    special_instructions: List[str] = field(default_factory=list)

@dataclass
class AggregationSpec:
    approach: str
    initial_weights: Dict[str, float]
    horizon_rules: Dict[str, str] = field(default_factory=dict)
    conflict_rules: Dict[str, str] = field(default_factory=dict)
    formula_note: str = ""

@dataclass
class DecompositionStrategy:
    description: str
    factors: List[FactorSpec]
    aggregation: Optional[AggregationSpec] = None

@dataclass
class BaseKnowHow:
    # ---- 元信息 ----
    domain: str
    sub_domain: str
    description: str

    # ---- 自描述 ----
    applicable_scenarios: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    # ---- 评估与工具 ----
    evaluation_criteria: Dict[str, str] = field(default_factory=dict)
    allowed_tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {tool_id: {param_hints...}}

    # ---- 分解策略 ----
    decomposition: DecompositionStrategy = None  # type: ignore

    # ====== 常用便捷方法 ======
    @property
    def task_type(self) -> str:
        return f"{self.domain}.{self.sub_domain}"

    def factor_names(self) -> List[str]:
        return [f.name for f in self.decomposition.factors]

    def factor_by_name(self, name: str) -> Optional[FactorSpec]:
        return next((f for f in self.decomposition.factors if f.name == name), None)

    def plan_seed_for_planner(self, include_aggregation: bool = True) -> Dict[str, Any]:
        """
        提供给 Decompose Agent 的“结构化种子”。aggregation 可选（默认 True）。
        """
        seed: Dict[str, Any] = {
            "task_type": self.task_type,
            "evaluation_criteria": self.evaluation_criteria,
            "factors": [
                {
                    "name": f.name,
                    "role": f.agent_role,
                    "tools": f.tools,
                    "analysis_steps": f.analysis_steps,
                    "output_contract": list(f.output_schema.keys()),
                    "task_hints": f.task_hints,
                }
                for f in self.decomposition.factors
            ],
        }
        if include_aggregation and self.decomposition.aggregation:
            seed["aggregation"] = {
                "approach": self.decomposition.aggregation.approach,
                "initial_weights": self.decomposition.aggregation.initial_weights
            }
        return seed

    def synthesis_contract(self) -> Dict[str, Any]:
        return {
            "factor_required_fields": ["expected_return", "confidence"],
            "initial_weights": (self.decomposition.aggregation.initial_weights
                                if self.decomposition.aggregation else {}),
            "approach": (self.decomposition.aggregation.approach
                         if self.decomposition.aggregation else "")
        }