"""
TaskLoader - 任务加载器

从JSONL文件加载预测任务
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径中，以便可以直接运行此文件
# 当前文件在 core/utils/task_loader.py，所以项目根目录是 parent.parent.parent
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import logging
from typing import List, Dict, Any, Optional

# task 模块在同一包内
from core.schemas.PredictionTask import PredictionTask

# 配置日志记录器
logger = logging.getLogger(__name__)


class TaskLoader:
    """
    任务加载器 - 从JSONL文件加载任务

    简化设计：
    - 只负责读取JSONL文件
    - 转换为PredictionTask对象
    - 提供基本的过滤功能
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        初始化任务加载器

        Args:
            data_path: JSONL文件路径，如果不提供则使用默认路径
        """
        if data_path is None:
            # 默认路径: 项目根目录/data/standardized_data.jsonl
            # 注意：这里假设 __file__ 是在 core/utils/task_loader.py
            # 如果结构不同，可能需要调整 .parent 的数量
            project_root = Path(__file__).parent.parent.parent
            data_path = project_root / "data" / "standardized_data.jsonl"

        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def load_all_tasks(self) -> List[PredictionTask]:
        """
        加载所有任务

        Returns:
            PredictionTask对象列表
        """
        tasks = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    task = self._parse_task(data)
                    tasks.append(task)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON on line {line_num}: {e}. Line content: {line[:100]}...")
                except KeyError as e:
                    logger.warning(f"Missing required key in data on line {line_num}: {e}. Data keys: {list(data.keys()) if 'data' in locals() else 'Unknown'}. Line content: {line[:100]}...")
                except Exception as e:
                    logger.warning(f"Unexpected error processing line {line_num}: {e}. Line content: {line[:100]}...")

        logger.info(f"Successfully loaded {len(tasks)} tasks from {self.data_path}")
        return tasks

    def load_task_by_id(self, task_id: str) -> Optional[PredictionTask]:
        """
        根据task_id加载单个任务

        Args:
            task_id: 任务ID

        Returns:
            PredictionTask对象，如果未找到则返回None
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if data.get("task_id") == task_id:
                        logger.info(f"Found task with ID {task_id} on line {line_num}.")
                        return self._parse_task(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON on line {line_num} while searching for ID {task_id}: {e}. Line content: {line[:100]}...")
                except Exception as e:
                    logger.warning(f"Unexpected error processing line {line_num} while searching for ID {task_id}: {e}. Line content: {line[:100]}...")

        logger.info(f"Task with ID {task_id} not found in {self.data_path}.")
        return None

    def load_tasks(self, limit: Optional[int] = None) -> List[PredictionTask]:
        """
        加载指定数量的任务

        Args:
            limit: 最多加载多少个任务

        Returns:
            PredictionTask列表
        """
        tasks = []
        count = 0

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if limit is not None and count >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    task = self._parse_task(data)
                    tasks.append(task)
                    count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON on line {line_num}: {e}. Line content: {line[:100]}...")
                except KeyError as e:
                    logger.warning(f"Missing required key in data on line {line_num}: {e}. Data keys: {list(data.keys()) if 'data' in locals() else 'Unknown'}. Line content: {line[:100]}...")
                except Exception as e:
                    logger.warning(f"Unexpected error processing line {line_num}: {e}. Line content: {line[:100]}...")

        logger.info(f"Successfully loaded {len(tasks)} tasks (limit was {limit}) from {self.data_path}")
        return tasks

    def _parse_task(self, data: Dict[str, Any]) -> PredictionTask:
        """
        解析原始JSON数据为PredictionTask对象

        Args:
            data: 从JSONL读取的字典

        Returns:
            PredictionTask对象（自动解析结构化字段）
        """
        # 支持多种数据格式
        # 格式1: 标准格式 (task_id, task_question, metadata)
        # 格式2: 财经数据格式 (id, prompt, title, en_title, end_time, level)
        
        task_id = data.get("task_id") or data.get("id", "")
        task_question = data.get("task_question") or data.get("prompt", "")
        
        # 构建metadata
        metadata = data.get("metadata", {})
        
        # 如果没有metadata，从其他字段构建
        if not metadata and "id" in data:
            metadata = {
                "title": data.get("title", ""),
                "en_title": data.get("en_title", ""),
                "end_time": data.get("end_time", ""),
                "level": data.get("level", 0),
                "slug": data.get("slug")
            }

        if not task_id:
            logger.warning(f"Data missing task_id/id field. Data keys: {list(data.keys())}")
        if not task_question:
            logger.warning(f"Data missing task_question/prompt field. Data keys: {list(data.keys())}")

        return PredictionTask(
            task_id=task_id or "",
            task_question=task_question or "",
            metadata=metadata,
            ground_truth=data.get("ground_truth"),
            file_path=data.get("file_path")
            # event_description, resolved_time, options, answer_instructions
            # 会在__post_init__中自动解析
        )


# 使用示例
if __name__ == "__main__":
    # 初始化加载器
    loader = TaskLoader()

    # 加载前3个任务
    tasks = loader.load_tasks(limit=3)
    print(f"Loaded {len(tasks)} tasks")

    for i, task in enumerate(tasks):
        print(f"\n--- Task {i+1} ---")
        print(f"{task}")
        print(f"Event Description: {task.event_description}")
        print(f"Resolved Time: {task.resolved_time}")
        print(f"Options: {task.options}")
        print(f"Answer Instructions: {task.answer_instructions}")