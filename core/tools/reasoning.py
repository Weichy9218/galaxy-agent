import json
from typing import Any, Dict, Optional

from core.llm.gpt5_client import GPT5Client
from core.llm.base import LLMResponse

SYSTEM_PROMPT = (
    "You are the SubAgent manager. Given a task execution context, "
    "decide whether to retry the task, skip it, or request a replan. "
    "Always respond with ONE JSON object matching this schema:\n"
    '{"decision":"retry|skip|replan","confidence":float,"analysis":"string"}'
)


class ReasoningTool:
    """
    Thin wrapper over GPT5Client to provide structured reasoning decisions.

    The tool expects a context dict with keys like error, task_id, goal, retries, history.
    """

    def __init__(self, llm_client: Optional[GPT5Client] = None):
        self.client = llm_client or GPT5Client(async_mode=True, temperature=0)

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the LLM and return a structured decision dict.
        Fallback to heuristic defaults when parsing fails.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(context, ensure_ascii=False, indent=2),
            },
        ]

        try:
            response: LLMResponse = await self.client.chat(messages=messages, temperature=0)
            raw = response.content.strip()
            data = self._extract_json(raw)
        except Exception as exc:  # noqa: BLE001
            return {
                "decision": "retry" if context.get("retries", 0) < 1 else "skip",
                "confidence": 0.3,
                "analysis": f"ReasoningTool fallback due to error: {exc}",
            }

        decision = data.get("decision") or "retry"
        confidence = float(data.get("confidence", 0.5))
        analysis = data.get("analysis") or data.get("explanation") or ""
        return {
            "decision": decision,
            "confidence": confidence,
            "analysis": analysis,
        }

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Parse the first JSON object found in text.
        """
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            snippet = text[start:end]
            return json.loads(snippet)
        except Exception:
            return {}
