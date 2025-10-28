# knowhow_store/registry.py
"""
Know-How 注册表

根据当前项目结构（knowhow_store/*）动态发现所有继承自 BaseKnowHow 的类，
并提供按 task_type / domain / 关键字 的检索能力。
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Type

from core.schemas.knowhow_base import BaseKnowHow


@dataclass
class KnowHowMetadata:
    """描述单个 Know-How 的静态信息。"""

    cls: Type[BaseKnowHow]
    module: str
    name: str
    domain: str
    sub_domain: str
    task_type: str
    description: str
    applicable_scenarios: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    task_type_aliases: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.domain}.{self.sub_domain}"

    @property
    def keywords(self) -> Set[str]:
        tokens: Set[str] = set()
        fields = [
            self.domain,
            self.sub_domain,
            self.description,
            *self.applicable_scenarios,
            *self.key_concepts,
            *self.task_type_aliases,
        ]
        for field in fields:
            tokens.update(_tokenize(field))
        return tokens


class KnowHowRegistry:
    """
    动态加载 knowhow_store 下的所有 BaseKnowHow 子类。

    使用方式：
        registry = KnowHowRegistry()
        meta = registry.get("finance.stock_price")
        knowhow_cls = meta.cls
    """

    def __init__(
        self,
        search_root: Optional[Path] = None,
        package_prefix: str = "knowhow_store",
    ) -> None:
        self._package_prefix = package_prefix
        self._root = search_root or Path(__file__).resolve().parent
        self._loaded: Dict[str, KnowHowMetadata] = {}
        self._fallback: Optional[KnowHowMetadata] = None
        self._discover()

    # --------- 公共接口 ---------

    def all(self) -> List[KnowHowMetadata]:
        return list(self._loaded.values())

    def get(self, task_type: str) -> Optional[KnowHowMetadata]:
        key = task_type.replace("/", ".")
        if key in self._loaded:
            return self._loaded[key]

        # 允许只写 sub_domain
        for meta in self._loaded.values():
            if key == meta.sub_domain:
                return meta
        return None

    def get_by_domain(self, domain: str) -> List[KnowHowMetadata]:
        return [m for m in self._loaded.values() if m.domain == domain]

    def fallback(self) -> Optional[KnowHowMetadata]:
        return self._fallback

    # --------- 内部实现 ---------

    def _discover(self) -> None:
        for py_file in self._iter_python_files(self._root):
            module_name = self._module_name_for(py_file)
            if not module_name:
                continue

            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if not issubclass(obj, BaseKnowHow) or obj is BaseKnowHow:
                    continue
                if obj.__module__ != module_name:
                    continue
                self._register(obj, module_name)

    def _register(self, cls: Type[BaseKnowHow], module_name: str) -> None:
        instance = cls()
        aliases = set(getattr(instance, "task_type_aliases", []))
        aliases.update(getattr(cls, "TASK_TYPE_ALIASES", []))

        meta = KnowHowMetadata(
            cls=cls,
            module=module_name,
            name=cls.__name__,
            domain=instance.domain,
            sub_domain=instance.sub_domain,
            task_type=instance.task_type,
            description=instance.description,
            applicable_scenarios=list(instance.applicable_scenarios),
            key_concepts=list(instance.key_concepts),
            task_type_aliases=sorted(aliases),
        )
        self._loaded[meta.task_type] = meta

        if meta.domain == "general" and self._fallback is None:
            self._fallback = meta

    def _module_name_for(self, path: Path) -> Optional[str]:
        if path.name == "__init__.py":
            return None
        try:
            relative = path.relative_to(self._root).with_suffix("")
        except ValueError:
            return None

        parts = [self._package_prefix, *relative.parts]
        return ".".join(parts)

    @staticmethod
    def _iter_python_files(root: Path) -> Iterable[Path]:
        for py in root.rglob("*.py"):
            if "__pycache__" in py.parts:
                continue
            if py == Path(__file__):
                continue
            yield py


# --------- 辅助函数 ---------

def _tokenize(text: str) -> Set[str]:
    tokens: Set[str] = set()
    for chunk in text.replace("/", " ").replace("-", " ").replace("_", " ").split():
        chunk = "".join(ch for ch in chunk.lower() if ch.isalnum())
        if len(chunk) >= 3:
            tokens.add(chunk)
    return tokens
