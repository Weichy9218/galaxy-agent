# core/utils/smart_matcher.py
"""
SmartMatcher

结合 KnowHowRegistry，基于任务信息自动选择最合适的 Know-How。
优先级：
1. task.task_type 直接匹配（含别名）
2. 关键字/特征打分
3. 兜底通用 Know-How
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from core.schemas.PredictionTask import PredictionTask
from core.schemas.knowhow_base import BaseKnowHow
from knowhow_store.registry import KnowHowMetadata, KnowHowRegistry


@dataclass
class MatchResult:
    metadata: KnowHowMetadata
    confidence: float
    matched_by: str
    reasoning: str


class SmartMatcher:
    def __init__(
        self,
        registry: Optional[KnowHowRegistry] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.registry = registry or KnowHowRegistry()
        self.llm_client = llm_client

    def match(self, task: PredictionTask) -> Tuple[BaseKnowHow, MatchResult]:
        meta = self._match_by_task_type(task)
        if meta:
            return self._instantiate(meta, matched_by="task_type", confidence=0.95)

        meta, score, explanation = self._match_by_keywords(task)
        if meta:
            confidence = min(0.9, 0.3 + score * 0.1)
            return self._instantiate(
                meta,
                matched_by="keywords",
                confidence=confidence,
                reasoning=explanation,
            )

        fallback = self.registry.fallback() or self._default_fallback()
        return self._instantiate(
            fallback,
            matched_by="fallback",
            confidence=0.5,
            reasoning="未找到更具体的 Know-How，使用通用模板。",
        )

    # --------- internal helpers ---------

    def _instantiate(
        self,
        meta: KnowHowMetadata,
        matched_by: str,
        confidence: float,
        reasoning: Optional[str] = None,
    ) -> Tuple[BaseKnowHow, MatchResult]:
        instance = self.registry.instantiate(meta)
        result = MatchResult(
            metadata=meta,
            confidence=confidence,
            matched_by=matched_by,
            reasoning=reasoning or "",
        )
        return instance, result

    def _match_by_task_type(self, task: PredictionTask) -> Optional[KnowHowMetadata]:
        task_type = getattr(task, "task_type", None)
        if not task_type:
            return None

        candidate = self.registry.get(task_type)
        if candidate:
            return candidate

        normalized = task_type.replace("_", ".").replace("/", ".")
        candidate = self.registry.get(normalized)
        if candidate:
            return candidate

        for meta in self.registry.all():
            if normalized in meta.task_type_aliases:
                return meta
        return None

    def _match_by_keywords(
        self,
        task: PredictionTask,
    ) -> Tuple[Optional[KnowHowMetadata], float, str]:
        combined_text = " ".join(
            filter(
                None,
                [
                    getattr(task, "task_question", ""),
                    getattr(task, "event_description", ""),
                    getattr(task, "target_entity", ""),
                    getattr(task, "metadata", {}).get("dataset_name", ""),
                ],
            )
        )
        tokens = _tokenize(combined_text)

        best_meta: Optional[KnowHowMetadata] = None
        best_score = 0.0
        rationale = ""

        for meta in self.registry.all():
            score, reason = self._score_metadata(meta, tokens, task)
            if score > best_score:
                best_meta = meta
                best_score = score
                rationale = reason

        if best_meta and (best_score >= 2 or best_meta.domain != "general"):
            return best_meta, best_score, rationale

        return None, 0.0, ""

    def _score_metadata(
        self,
        meta: KnowHowMetadata,
        tokens: set,
        task: PredictionTask,
    ) -> Tuple[float, str]:
        score = 0.0
        hits = []

        token_hits = tokens & meta.keywords
        if token_hits:
            score += len(token_hits)
            hits.extend(sorted(token_hits))

        domain_token = meta.domain.lower()
        if domain_token and domain_token in tokens:
            score += 2
            hits.append(domain_token)

        sub_tokens = _tokenize(meta.sub_domain)
        if sub_tokens and sub_tokens.issubset(tokens):
            score += 2
            hits.extend(sorted(sub_tokens))

        entity = getattr(task, "target_entity", "") or ""
        if meta.domain == "finance" and _looks_like_ticker(entity):
            score += 3
            hits.append("ticker")

        if meta.domain == "finance" and "stock" in tokens:
            score += 2
            hits.append("stock")

        reason = ", ".join(dict.fromkeys(hits))
        return score, reason

    def _default_fallback(self) -> KnowHowMetadata:
        # 当注册表中没有明确 fallback 时，选择任意一个（通常不会发生）
        meta_list = self.registry.all()
        if not meta_list:
            raise RuntimeError("No Know-How available in registry.")
        return meta_list[0]


# --------- helpers ---------

def _tokenize(text: str) -> set:
    tokens = set()
    for chunk in str(text).replace("/", " ").replace("-", " ").replace("_", " ").split():
        chunk = "".join(ch for ch in chunk.lower() if ch.isalnum())
        if len(chunk) >= 3:
            tokens.add(chunk)
    return tokens


def _looks_like_ticker(text: str) -> bool:
    if not text:
        return False
    stripped = text.replace(".", "").replace(":", "")
    return stripped.isupper() and 1 <= len(stripped) <= 6
