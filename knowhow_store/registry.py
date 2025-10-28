from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.schemas.knowhow_base import (
    BaseKnowHow,
    FactorSpec,
    AggregationSpec,
    DecompositionStrategy,
)


@dataclass
class KnowHowMetadata:
    domain: str
    sub_domain: str
    description: str
    applicable_scenarios: List[str]
    key_concepts: List[str]
    examples: List[str]
    task_type_aliases: List[str]
    source_path: Path

    @property
    def task_type(self) -> str:
        return f"{self.domain}.{self.sub_domain}"

    @property
    def keywords(self) -> set[str]:
        tokens: set[str] = set()
        fields = [
            self.domain,
            self.sub_domain,
            self.description,
            *self.applicable_scenarios,
            *self.key_concepts,
            *self.examples,
            *self.task_type_aliases,
        ]
        for field in fields:
            for chunk in str(field).replace("/", " ").replace("_", " ").split():
                cleaned = "".join(c for c in chunk.lower() if c.isalnum())
                if len(cleaned) >= 3:
                    tokens.add(cleaned)
        return tokens


class KnowHowRegistry:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).resolve().parent
        self._knowhows: Dict[str, Tuple[BaseKnowHow, KnowHowMetadata]] = {}
        self._discover()

    # public API -----------------------------------------------------

    def all(self) -> List[KnowHowMetadata]:
        return [meta for _, meta in self._knowhows.values()]

    def get(self, task_type: str) -> Optional[KnowHowMetadata]:
        key = task_type.replace("/", ".")
        entry = self._knowhows.get(key)
        if entry:
            return entry[1]

        for _, meta in self._knowhows.values():
            aliases = {alias.replace("/", ".") for alias in meta.task_type_aliases}
            if key in aliases:
                return meta
        return None

    def lookup_instance(self, task_type: str) -> Optional[BaseKnowHow]:
        key = task_type.replace("/", ".")
        entry = self._knowhows.get(key)
        if entry:
            return entry[0]
        for inst, meta in self._knowhows.values():
            aliases = {alias.replace("/", ".") for alias in meta.task_type_aliases}
            if key in aliases:
                return inst
        return None

    def fallback(self) -> Optional[KnowHowMetadata]:
        for _, meta in self._knowhows.values():
            if meta.domain == "general":
                return meta
        return next(iter(self.all()), None)

    # discovery -----------------------------------------------------

    def _discover(self) -> None:
        for json_path in self.base_dir.rglob("*.json"):
            if "factors" in json_path.parts:
                continue
            data = self._load_json(json_path)
            instance = self._build_knowhow(json_path, data)
            meta = KnowHowMetadata(
                domain=data["domain"],
                sub_domain=data["sub_domain"],
                description=data.get("description", ""),
                applicable_scenarios=data.get("applicable_scenarios", []),
                key_concepts=data.get("key_concepts", []),
                examples=data.get("examples", []),
                task_type_aliases=data.get("task_type_aliases", []),
                source_path=json_path,
            )
            key = meta.task_type
            self._knowhows[key] = (instance, meta)

    def _build_knowhow(self, base_path: Path, data: Dict[str, any]) -> BaseKnowHow:
        aggregation = self._build_aggregation(data.get("aggregation_spec"))
        factors = [
            self._load_factor(base_path.parent / factor_ref["$ref"])
            if "$ref" in factor_ref
            else self._factor_from_dict(factor_ref)
            for factor_ref in data.get("decomposition_strategy", {}).get("factors", [])
        ]
        decomp = DecompositionStrategy(
            description=data.get("decomposition_strategy", {}).get("description", ""),
            factors=factors,
            aggregation=aggregation,
        )

        return BaseKnowHow(
            domain=data["domain"],
            sub_domain=data["sub_domain"],
            description=data.get("description", ""),
            applicable_scenarios=data.get("applicable_scenarios", []),
            key_concepts=data.get("key_concepts", []),
            examples=data.get("examples", []),
            evaluation_criteria=data.get("evaluation_criteria", {}),
            allowed_tools=data.get("allowed_tools", {}),
            decomposition=decomp,
        )

    def _build_aggregation(self, payload: Optional[Dict[str, Any]]) -> Optional[AggregationSpec]:
        if not payload:
            return None
        return AggregationSpec(
            approach=payload.get("approach", ""),
            initial_weights=payload.get("initial_weights", {}),
            horizon_rules=payload.get("horizon_rules", {}),
            conflict_rules=payload.get("conflict_rules", {}),
            formula_note=payload.get("formula_note", ""),
        )

    def _load_factor(self, path: Path) -> FactorSpec:
        data = self._load_json(path)
        return self._factor_from_dict(data)

    def _factor_from_dict(self, data: Dict[str, Any]) -> FactorSpec:
        return FactorSpec(
            name=data["name"],
            description=data.get("description", ""),
            agent_role=data.get("agent_role", ""),
            tools=data.get("tools", []),
            analysis_steps=data.get("analysis_steps", []),
            output_schema=data.get("output_schema", {}),
            task_hints=data.get("task_hints", []),
            special_instructions=data.get("special_instructions", []),
        )

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def instantiate(self, metadata: KnowHowMetadata) -> BaseKnowHow:
        data = self._load_json(metadata.source_path)
        return self._build_knowhow(metadata.source_path, data)


__all__ = ["KnowHowRegistry", "KnowHowMetadata"]
