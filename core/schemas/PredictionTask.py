import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class PredictionTask:
    """
    预测任务（标准化后给整个预测流水线用）

    设计目标：
    - 从原始task_question中提取结构化信息
    - 为 SmartMatcher / Planner 提供明确上下文
    - 不做过度工程，保持最小可用字段
    """

    # ---- 原始必需输入 ----
    task_id: str
    task_question: str                  # 原始完整问题
    metadata: Dict[str, Any]            # 训练元信息，如 end_time, dataset_name...

    # ---- 标签/真值类信息（可选）----
    ground_truth: Optional[str] = None  # 实际答案（如果有）
    file_path: Optional[str] = None     # 来源文件（如果有）

    # ---- 解析后字段（自动填充）----
    event_description: Optional[str] = None     # 事件描述: "2025-10-24..., What will be the low of Apple stock"
    resolved_time: Optional[str] = None         # 判定时间 / 截止时间
    options: Optional[List[Dict[str, str]]] = None  # [{"label":"A","content":"Yes"},...]
    answer_instructions: Optional[str] = None   # "IMPORTANT: ... \boxed{YOUR_PREDICTION} ..."

    # ---- 额外结构化信息（便于后续KnowHow匹配/Planner执行）----
    task_type: Optional[str] = None      # e.g. "stock_intraday_low", "index_intraday_high", "book_rank_range"
    target_entity: Optional[str] = None  # e.g. "AAPL", "INDEXSP:.INX", "Amazon Charts Most Read Fiction"
    target_metric: Optional[str] = None  # e.g. "day_low_price", "day_high_price", "rank_3_to_5_titles"

    def __post_init__(self):
        """
        初始化后自动进行字段提取。
        解析策略是启发式的，不做强鲁棒（后续可加validator）。
        """

        # 1. 基础解析：事件、答题格式
        if self.event_description is None:
            self.event_description = self._extract_event_description()

        if self.answer_instructions is None:
            self.answer_instructions = self._extract_answer_instructions()

        # 2. 时间、选项
        if self.resolved_time is None:
            self.resolved_time = self._extract_resolved_time()

        if self.options is None:
            self.options = self._extract_options()

        # 3. 推断任务类型 / 目标实体 / 目标指标
        # 这些是给 SmartMatcher / Planner 用的
        if self.task_type is None:
            self.task_type = self._infer_task_type()

        if self.target_entity is None:
            self.target_entity = self._infer_target_entity()

        if self.target_metric is None:
            self.target_metric = self._infer_target_metric()

    # -----------------------
    # 各类解析函数
    # -----------------------

    def _extract_event_description(self) -> str:
        """
        尝试从 task_question 中抓出 "The event to be predicted: ..."
        返回人类可读的事件描述，不包含IMPORTANT部分。
        """
        match_quoted = re.search(
            r'The event to be predicted:\s*"([^"]+?)"',
            self.task_question,
            re.DOTALL
        )
        if match_quoted:
            return match_quoted.group(1).strip()

        match_unquoted = re.search(
            r'The event to be predicted:\s*(.+?)(?=\n|IMPORTANT|$)',
            self.task_question,
            re.DOTALL | re.IGNORECASE
        )
        if match_unquoted:
            desc = match_unquoted.group(1).strip()
            return desc.rstrip(' .')

        # fallback: IMPORTANT 之前的部分
        fallback = self.task_question.split("IMPORTANT:")[0].strip()
        fallback = re.sub(
            r'^You are an agent that can predict future events\.\s*',
            '',
            fallback,
            flags=re.IGNORECASE
        )
        return fallback[:500]

    def _extract_resolved_time(self) -> Optional[str]:
        """
        判定时间/预测区间的截止点。
        优先metadata.end_time，其次文本中的 YYYY-MM-DD，最后尝试 'resolved around ...'
        """
        end_time = self.metadata.get("end_time")
        if end_time:
            return f"{end_time} (GMT+8)"

        text = self.event_description or self.task_question
        date_match = re.search(r'\b(20\d{2}-\d{2}-\d{2})\b', text)
        if date_match:
            return f"{date_match.group(1)} (GMT+8)"

        match = re.search(r'resolved\s+around\s+([^\n.]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip('.')

        return None

    def _extract_answer_instructions(self) -> str:
        """
        IMPORTANT 后面的整块指令。
        这是后续 Synthesis Agent 格式化输出时的硬约束。
        """
        match = re.search(
            r'IMPORTANT:(.+?)$',
            self.task_question,
            re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()

        return "Your final answer MUST end with \\boxed{YOUR_PREDICTION} format."

    def _extract_options(self) -> List[Dict[str, str]]:
        """
        尝试识别多选型问题的选项。
        优先 A./B./C. ... 格式，否则回退到 Yes/No。
        """
        text_for_options = self.task_question or ""
        text_for_options = re.split(r'\bIMPORTANT\b', text_for_options, flags=re.IGNORECASE)[0]
        text_for_options = text_for_options.replace('\r\n', '\n').replace('\r', '\n')

        label_pat  = r'(?:[A-Za-z]|[0-9]{1,2})'
        bullet_pat = r'[\.\)\:]'
        pattern = re.compile(
            rf'(?:(?<=\n)|^)\s*({label_pat})\s*{bullet_pat}\s*(.+?)'
            rf'(?=(?:\n\s*{label_pat}\s*{bullet_pat}\s*)|\Z)',
            re.DOTALL
        )

        options = []
        for m in pattern.finditer(text_for_options):
            label, content = m.group(1), m.group(2)
            if re.fullmatch(r'[A-Za-z]', label):
                label = label.upper()
            options.append({
                "label": label.strip(),
                "content": content.strip()
            })

        if not options:
            answer_instr = self.answer_instructions or ""
            yes_no_pattern = re.compile(
                r'\\boxed\{Yes\}\s+or\s+\\boxed\{No\}|\\boxed\{No\}\s+or\s+\\boxed\{Yes\}',
                re.IGNORECASE
            )
            if yes_no_pattern.search(answer_instr):
                options = [
                    {"label": "A", "content": "Yes"},
                    {"label": "B", "content": "No"}
                ]

        return options

    # -----------------------
    # 辅助推断：为 SmartMatcher / Planner 服务
    # -----------------------

    def _infer_task_type(self) -> Optional[str]:
        """
        用启发式规则（轻量，不追求完美）识别任务范式，
        之后SmartMatcher可以直接用它来选用哪套KnowHow。
        """
        text = self.event_description or ""
        t = text.lower()

        # 股票日内最低价
        # e.g. "What will be the low of Apple stock (AAPL) be for the day?"
        if "stock" in t and "low" in t and "for the day" in t:
            return "stock_intraday_low"

        # 指数当日最高点
        if ("index" in t or "s&p 500" in t or "sp 500" in t or "s&p 500 index" in t) and "day's high" in t:
            return "index_intraday_high"

        # 排名预测 (books rank 3 to 5...)
        if "rank 3 to 5" in t and "amazon charts" in t:
            return "book_rank_range"

        # fallback: None
        return None

    def _infer_target_entity(self) -> Optional[str]:
        """
        提取目标实体，比如股票ticker/指数代号/榜单名。
        简单regex+heuristic。
        """
        text = self.event_description or ""

        # ticker in parentheses like (AAPL)
        m = re.search(r'\(([A-Z\.:\-]{1,15})\)', text)
        if m:
            return m.group(1).strip()

        # S&P 500 Index
        if "s&p 500" in text.lower():
            return "S&P 500"

        # Amazon Charts Most Read Fiction
        if "amazon charts" in text.lower():
            return "Amazon Charts Most Read Fiction"

        return None

    def _infer_target_metric(self) -> Optional[str]:
        """
        提取要预测的“指标/输出单位”。
        这个字段让 Planner 知道是什么数值/信息要被最终合成。
        """
        text = self.event_description or ""
        t = text.lower()

        if "low" in t and "for the day" in t:
            return "day_low_price"

        if "day's high" in t or "day high" in t:
            return "day_high_price"

        if "rank 3 to 5" in t and "book" in t:
            return "rank_3_to_5_titles"

        return None

    # -----------------------
    # 下游接口
    # -----------------------

    def get_formatted_prompt(self) -> str:
        """
        返回一个人类可读/LLM可读的完整上下文。
        这个可以直接给 Decompose Agent 作为“任务描述”部分。
        """
        parts = []

        parts.append(f"Event to predict: {self.event_description}")

        if self.resolved_time:
            parts.append(f"Result will be determined around: {self.resolved_time}")

        if self.options:
            parts.append("\nOptions:")
            for opt in self.options:
                parts.append(f"{opt['label']}. {opt['content']}")

        parts.append(f"\n{self.answer_instructions}")
        return "\n".join(parts)

    def to_planner_input(self) -> Dict[str, Any]:
        """
        给 Decompose Agent / Planner 的结构化输入。
        Planner 理论上应优先使用这个，而不是重新解析自然语言。
        """
        return {
            "task_id": self.task_id,
            "event_description": self.event_description,
            "resolved_time": self.resolved_time,
            "target_entity": self.target_entity,
            "target_metric": self.target_metric,
            "task_type": self.task_type,
            "answer_instructions": self.answer_instructions,
            "options": self.options,
            "metadata": self.metadata,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        通用序列化（比 planner_input 更全，用于存档/audit）
        """
        return {
            "task_id": self.task_id,
            "task_question": self.task_question,
            "event_description": self.event_description,
            "resolved_time": self.resolved_time,
            "options": self.options,
            "answer_instructions": self.answer_instructions,
            "metadata": self.metadata,
            "ground_truth": self.ground_truth,
            "file_path": self.file_path,
            "task_type": self.task_type,
            "target_entity": self.target_entity,
            "target_metric": self.target_metric,
        }

    def __str__(self) -> str:
        opt_info = f", {len(self.options)} options" if self.options else ""
        return (
            f"PredictionTask(id={self.task_id}{opt_info}, "
            f"task_type={self.task_type}, "
            f"entity={self.target_entity}, "
            f"event={self.event_description[:50] if self.event_description else 'N/A'}...)"
        )
