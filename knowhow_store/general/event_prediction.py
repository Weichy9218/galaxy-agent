# knowhow_store/general/event_prediction.py
from typing import Dict, Any, Optional
from core.schemas.knowhow_base import (
    BaseKnowHow,
    FactorSpec,
    AggregationSpec,
    DecompositionStrategy,
)


class GeneralEventKnowHow(BaseKnowHow):
    """
    面向通用事件预测任务的 Know-How。
    适配 data/standardized_data.jsonl 中涵盖的体育赛果、政策事件、公司公告等多类型问题。
    """

    DOMAIN = "general"
    SUB_DOMAIN = "event_prediction"
    DESCRIPTION = "Universal framework for discrete event outcome forecasting with multiple choice or yes/no answers."
    TASK_TYPE_ALIASES = [
        "event_prediction",
        "general.event_prediction",
        "general/event_prediction",
    ]

    APPLICABLE_SCENARIOS = [
        "体育或电竞比赛的胜负/赔率预测",
        "重大政策、法规或宏观事件的发生概率",
        "企业财报、产品发布或并购是否达成",
        "多选题形式的离散结果预测任务",
        "需要综合历史、现势、专家观点的事件判断",
    ]

    KEY_CONCEPTS = [
        "baseline probability",
        "rough order-of-magnitude reasoning",
        "recent momentum vs structural pattern",
        "expert/market consensus",
        "risk and tail scenarios",
        "option canonicalization",
    ]

    EVALUATION_CRITERIA = {
        "recency": "近期(<=14天)信息优先；超出90天仅作为背景。",
        "source_quality": "权威机构 > 主流媒体 > 专家评论 > 社交媒体传闻。",
        "signal_strength": "量化信号要给出样本规模或指标强度。",
        "consensus_alignment": "与主流赔率/投票一致但须说明差异来源。",
        "risk_noting": "识别黑天鹅或关键不确定性并评估影响方向。",
    }

    ALLOWED_TOOLS: Dict[str, Dict[str, Any]] = {
        "web_search": {
            "hints": ["query", "recency_days", "limit", "domains"],
            "examples": [
                {"query": "teamA vs teamB injury report", "recency_days": 7, "limit": 5},
                {"query": "bill passage likelihood analysis", "recency_days": 14, "limit": 5},
            ],
        },
        "structured_data": {
            "hints": ["entity", "metric", "lookback", "filter"],
            "examples": [
                {"entity": "teamA", "metric": "match_results", "lookback": "2y"},
                {"entity": "companyX", "metric": "earnings_surprise", "lookback": "8q"},
            ],
        },
        "reading": {
            "hints": ["url", "selectors"],
        },
        "reasoning": {
            "hints": ["inputs", "formula", "assumptions"],
        },
    }

    def _factor_spec(self, name: str) -> Optional[FactorSpec]:
        cfgs: Dict[str, Dict[str, Any]] = {
            "event_scoping": {
                "role": "scenario architect",
                "tools": ["reasoning", "reading"],
                "analysis_steps": [
                    "概括事件核心要素（主语、时间、判定格式）。",
                    "整理可选答案并标注需要进一步确认的关键信息。",
                ],
                "task_hints": [
                    "使用任务字段提取事件描述、resolved_time、options。",
                    "记录需要的占位符，如 ${event_date}、${primary_actor}。",
                ],
                "output_schema": {
                    "event_summary": "string",
                    "decision_window": "string",
                    "options_checked": ["string"],
                    "placeholders": {"label": "value"},
                    "recommended_option": "string",
                    "probability_delta": "float",
                    "confidence": "float",
                    "reasoning": "string",
                },
                "desc": "把原始问题转成标准化事件表述并列出关键占位符。",
            },
            "historical_baseline": {
                "role": "historical analyst",
                "tools": ["structured_data", "reasoning", "web_search"],
                "analysis_steps": [
                    "回顾主要参与方的近期表现或历史发生率。",
                    "得出一个基础倾向并说明可信度。",
                ],
                "task_hints": [
                    "列出最近的几场比赛/事件及结果。",
                    "简单说明样本规模是否足够代表当前情境。",
                ],
                "output_schema": {
                    "history_snapshot": [
                        {"note": "string", "outcome": "string"}
                    ],
                    "baseline_view": "string",
                    "recommended_option": "string",
                    "probability_delta": "float",
                    "confidence": "float",
                    "reasoning": "string",
                },
                "desc": "给出历史或先验角度的初步倾向。",
            },
            "current_signals": {
                "role": "situational analyst",
                "tools": ["web_search", "reading", "reasoning"],
                "analysis_steps": [
                    "关注最近发生的动态或消息。",
                    "判断这些动态对结果的方向性影响。",
                ],
                "task_hints": [
                    "列出 1-3 条最相关的新闻或状态更新。",
                    "标注每条信息的影响方向（正/负/不确定）。",
                ],
                "output_schema": {
                    "recent_events": [{"title": "string", "impact": "positive|negative|neutral"}],
                    "signal_direction": "positive|negative|neutral",
                    "recommended_option": "string",
                    "probability_delta": "float",
                    "confidence": "float",
                    "reasoning": "string",
                },
                "desc": "整理即时信号并判断对结果的拉动方向。",
            },
            "consensus_and_risk": {
                "role": "consensus analyst",
                "tools": ["web_search", "reading", "reasoning", "structured_data"],
                "analysis_steps": [
                    "查看外部共识或赔率信息。",
                    "标记潜在风险点或竞争结果。",
                ],
                "task_hints": [
                    "引用 1-2 个共识来源（赔率/民调/专家）。",
                    "列出需要关注的高风险情景。",
                ],
                "output_schema": {
                    "consensus_notes": ["string"],
                    "risk_flags": ["string"],
                    "recommended_option": "string",
                    "probability_delta": "float",
                    "confidence": "float",
                    "reasoning": "string",
                },
                "desc": "对照外部观点并列出关键风险，辅助最终判断。",
            },
        }
        cfg = cfgs.get(name)
        if not cfg:
            return None
        return FactorSpec(
            name=name,
            description=cfg["desc"],
            agent_role=cfg["role"],
            tools=cfg["tools"],
            analysis_steps=cfg["analysis_steps"],
            output_schema=cfg["output_schema"],
            task_hints=cfg["task_hints"],
            special_instructions=[],
        )

    @classmethod
    def initial_weights(cls) -> Dict[str, float]:
        return {
            "event_scoping": 0.15,
            "historical_baseline": 0.35,
            "current_signals": 0.25,
            "consensus_and_risk": 0.25,
        }

    def __init__(self):
        factor_names = [
            "event_scoping",
            "historical_baseline",
            "current_signals",
            "consensus_and_risk",
        ]
        factors = [self._factor_spec(name) for name in factor_names]
        factors = [f for f in factors if f is not None]

        aggregation = AggregationSpec(
            approach="weighted_probability_blending",
            initial_weights=self.initial_weights(),
            horizon_rules={
                "short_term(<7d)": "weight_up(current_signals,+0.15); weight_down(historical_baseline,-0.10)",
                "long_term(>30d)": "weight_up(historical_baseline,+0.15); weight_down(current_signals,-0.10)",
            },
            conflict_rules={
                "consensus_override": "if consensus_and_risk.confidence>0.65 and disagrees with others, cap probability shift to ±0.2",
                "risk_alert": "if consensus_and_risk.risk_flags is not empty, reduce overall confidence by 0.1",
            },
            formula_note="Final probability = Σ (w_i * probability_delta_i) + baseline; ensure normalization before option selection.",
        )

        strategy = DecompositionStrategy(
            description="General-purpose decomposition for discrete event forecasting across domains.",
            factors=factors,
            aggregation=aggregation,
        )

        super().__init__(
            domain=self.DOMAIN,
            sub_domain=self.SUB_DOMAIN,
            description=self.DESCRIPTION,
            applicable_scenarios=self.APPLICABLE_SCENARIOS,
            key_concepts=self.KEY_CONCEPTS,
            examples=[],
            evaluation_criteria=self.EVALUATION_CRITERIA,
            allowed_tools=self.ALLOWED_TOOLS,
            decomposition=strategy,
        )

    def synthesis_contract(self) -> Dict[str, Any]:
        """
        覆盖默认合成契约，使综合层关注通用概率字段。
        """
        aggregation = self.decomposition.aggregation
        return {
            "factor_required_fields": ["recommended_option", "probability_delta", "confidence"],
            "initial_weights": aggregation.initial_weights if aggregation else {},
            "approach": aggregation.approach if aggregation else "",
        }
