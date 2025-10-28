# planner/smart_matcher.py
from typing import Optional
from knowhow_store.finance.stock_price import StockPriceKnowHow
from core.schemas.PredictionTask import PredictionTask 
class SmartMatcher:
    """最简匹配器：后续可升级为语义检索 + 规则裁决。"""
    def match(self, task: PredictionTask):
        if task.task_type in ("finance.stock_price", "stock_price", "finance/stock_price"):
            return StockPriceKnowHow()
        # TODO: 其他task_type
        raise ValueError(f"Unsupported task_type: {task.task_type}")
