# knowhow_store/general/__init__.py
"""
General Know-How package.

当前仅暴露通用事件预测的 Know-How，用于在没有更具体模板时兜底。
"""

from .event_prediction import GeneralEventKnowHow

__all__ = ["GeneralEventKnowHow"]
