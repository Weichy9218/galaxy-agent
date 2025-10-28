# knowhow_store/finance/stock_price.py
from typing import List, Dict, Any, Optional
from core.schemas.knowhow_base import (
    BaseKnowHow, FactorSpec, AggregationSpec, DecompositionStrategy
)

class StockPriceKnowHow(BaseKnowHow):
    DOMAIN = "finance"
    SUB_DOMAIN = "stock_price"
    DESCRIPTION = "Predict stock closing price or price movement at a specific future date"

    APPLICABLE_SCENARIOS = [
        "Predict stock closing price at a future date",
        "Predict whether stock price will rise or fall",
        "Predict if stock will reach a certain price level",
        "Estimate stock price change percentage",
        "Forecast stock valuation changes"
    ]

    KEY_CONCEPTS = [
        "technical indicators", "valuation metrics", "macro trends",
        "sector performance", "earnings reports", "market sentiment",
        "price momentum", "moving averages"
    ]

    EVALUATION_CRITERIA = {
        "timeliness": "Recent (<=30d) gets highest weight; >90d down-weighted",
        "source_reliability": "official > mainstream_media > analyst > social",
        "technical_applicability": "Better on liquid large-caps; small-caps rely more on fundamentals",
        "market_regime": "Bull markets favor trend; bear markets favor valuation",
        "event_impact": "Structural > routine"
    }

    ALLOWED_TOOLS: Dict[str, Dict[str, Any]] = {
        # 工具层暂未实现：只保留接口位和参数提示
        "financial_data": {
            "hints": ["ticker", "range", "interval", "fields"],
            "examples": [
                {"ticker":"NVDA", "range":"60d", "interval":"1d",
                 "fields":["open","high","low","close","volume"]},
                {"ticker":"NVDA", "range":"1y", "interval":"1d",
                 "fields":["pe","pb","market_cap"]}
            ]
        },
        "web_search": {
            "hints": ["query", "recency_days", "limit"]
        },
        "reading": {
            "hints": ["url", "selectors"]
        },
        "reasoning": {
            "hints": ["inputs", "formula"]
        }
    }

    def _factor_spec(self, name: str) -> Optional[FactorSpec]:
        cfgs: Dict[str, Dict[str, Any]] = {
            "micro_trend": {
                "role": "quantitative analyst",
                "tools": ["financial_data", "reasoning"],
                "analysis_steps": [
                    "Technical micro trend from OHLCV (60d) with r_5d, MA20 gap, RSI14 normalization",
                    "Clip extreme signals; align sign consistency; compute strength score"
                ],
                "task_hints": [
                    "fetch: 60d OHLCV with 1d interval",
                    "compute: r_5d, MA20, gap_MA20=(close_0-MA20_0)/MA20_0, RSI14, RSI_14_norm=(RSI14-50)/50 clipped [-1,1]",
                    "compute: Micro=0.5*r_5d + 0.3*gap_MA20 + 0.2*RSI_14_norm",
                    "judge: trend_direction by Micro thresholds (+1%/-1%) and strength tiers",
                    "write: expected_return=clip(Micro,±3%), confidence based on indicator alignment"
                ],
                "output_schema": {
                    "trend_direction": "upward|downward|neutral",
                    "strength_score": "float(-1..1)",
                    "expected_return": "float",
                    "confidence": "float",
                    "key_indicators": {"MA_signal":"golden_cross|death_cross|neutral","RSI":"int(0..100)"},
                    "reasoning": "string"
                },
                "desc": "Micro trend via price/volume technicals"
            },
            "macro_sector_trend": {
                "role": "macro analyst",
                "tools": ["financial_data", "reasoning"],
                "analysis_steps": [
                    "Market/sector context and relative strength vs SPY; map sector to SPDR ETF"
                ],
                "task_hints": [
                    "fetch: SPY 20d close; sector ETF 20d close; stock 20d close; stock metadata for sector/beta",
                    "compute: r_5d_index, r_5d_sector, r_20d_stock, RS_20d=r_20d_stock - r_20d_index",
                    "compute: Macro=0.5*r_5d_index + 0.3*r_5d_sector + 0.2*RS_20d",
                    "judge: sector_trend tiers by |r_5d_sector|; market_environment risk-on/off",
                    "write: expected_return=clip(Macro,±2%), confidence adjustments by alignment"
                ],
                "output_schema": {
                    "sector": "string",
                    "sector_trend": "strong_upward|moderate_upward|neutral|downward",
                    "sector_return_1m": "float",
                    "stock_beta": "float",
                    "expected_return": "float",
                    "confidence": "float",
                    "market_environment": {"overall_market":"bullish|bearish|neutral"},
                    "reasoning": "string"
                },
                "desc": "Sector & market backdrop and relative strength"
            },
            "valuation_correction": {
                "role": "fundamental analyst",
                "tools": ["financial_data", "reasoning"],
                "analysis_steps": [
                    "Valuation mean-reversion signal using PE/PB 52w z-scores; coverage-based confidence"
                ],
                "task_hints": [
                    "fetch: 1y daily PE/PB",
                    "compute: current_PE, historical_avg_PE, z_PE_52w; 同理 PB",
                    "compute: Val = -0.5*z_PE_52w - 0.5*z_PB_52w",
                    "judge: correction_direction from Val sign and |Val| tiers",
                    "write: expected_return=clip(Val*0.005,±1%), confidence by coverage and |z|"
                ],
                "output_schema": {
                    "valuation_status": "overvalued|fairly_valued|undervalued",
                    "current_PE": "float",
                    "historical_avg_PE": "float",
                    "current_PB": "float",
                    "historical_avg_PB": "float",
                    "correction_direction": "downward|neutral|upward",
                    "expected_return": "float",
                    "confidence": "float",
                    "reasoning": "string"
                },
                "desc": "Mean-reversion pressure from valuation deviation"
            },
            "event_news_impact": {
                "role": "news analyst",
                "tools": ["web_search", "reading", "reasoning"],
                "analysis_steps": [
                    "Time-decayed sentiment + earnings surprise proxy; prefer official filings/PR"
                ],
                "task_hints": [
                    "fetch: web_search last 14d; queries include earnings/results/guidance/M&A/litigation/SEC",
                    "extract: parse article timestamp/source/event type; numeric surprises if present",
                    "compute: news_shock∈[-1,1] with source weights and exp(-age/7) decay",
                    "compute: fund_shock from EPS/Rev surprise if available else 0",
                    "compute: epsilon_event=clip(0.5*fund_shock+0.5*news_shock,[-1,1])",
                    "judge: overall_sentiment/impact_duration by epsilon_event and event type",
                    "write: expected_return=clip(epsilon_event*0.02,±4%), confidence by source & corroboration"
                ],
                "output_schema": {
                    "events": [{"event":"string","date":"YYYY-MM-DD","category":"enum","sentiment":"enum"}],
                    "overall_sentiment": "positive|neutral|negative",
                    "expected_return": "float",
                    "confidence": "float",
                    "impact_duration": "short_term|medium_term|long_term",
                    "reasoning": "string"
                },
                "desc": "Recent events & news shock normalization"
            }
        }
        c = cfgs.get(name)
        if not c: 
            return None
        return FactorSpec(
            name=name,
            description=c["desc"],
            agent_role=c["role"],
            tools=c["tools"],
            analysis_steps=c["analysis_steps"],
            output_schema=c["output_schema"],
            task_hints=c["task_hints"],
            special_instructions=[]
        )

    @classmethod
    def initial_weights(cls) -> Dict[str, float]:
        return {
            "micro_trend": 0.55,
            "macro_sector_trend": 0.25,
            "valuation_correction": 0.12,
            "event_news_impact": 0.08,
        }

    def __init__(self):
        factors = [
            self._factor_spec("micro_trend"),
            self._factor_spec("macro_sector_trend"),
            self._factor_spec("valuation_correction"),
            self._factor_spec("event_news_impact"),
        ]
        factors = [f for f in factors if f is not None]

        agg = AggregationSpec(
            approach="multi_factor_return_decomposition",
            initial_weights=self.initial_weights(),
            horizon_rules={
                "<=1w": "weight_up(micro_trend,+0.2); weight_down(valuation_correction,-0.1)",
                ">=1m": "weight_up(valuation_correction,+0.15); weight_down(micro_trend,-0.15)"
            },
            conflict_rules={
                "micro_vs_valuation_strong_conflict": "if both conf>0.7 and opposite, apply 0.2 penalty to overall confidence"
            },
            formula_note="E(r)=Σ w_i * r_i; confidence=Σ w_i * c_i * (1 - conflict_penalty)"
        )

        decomp = DecompositionStrategy(
            description="Decompose expected return into micro/macro/valuation/event sub-problems",
            factors=factors,
            aggregation=agg  # Planner 侧已支持 include_aggregation=False，不会下发
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
            decomposition=decomp
        )
