import json
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from core.utils.task_loader import TaskLoader
from core.utils.smart_matcher import SmartMatcher
from pipeline.planner.decompose_agent import DecomposeAgent
from core.llm.gpt5_client import GPT5Client
from pipeline.excutors.langgraph_plan_executor import LangGraphPlanExecutor
from pipeline.synthesis.synthesis_agent import SynthesisAgent


class PredictionPipeline:
    """
    PredictionTask -> KnowHow -> DecomposeAgent -> Executor -> SynthesisAgent
    将重要中间结果写入 log/{task_id}_*.json
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        log_dir: str = "log",
        enable_reasoning_llm: bool = False,
    ):
        self.loader = TaskLoader(data_path=data_path)
        self.matcher = SmartMatcher()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.llm = GPT5Client(async_mode=False, temperature=0)
        self.decompose_agent = DecomposeAgent(self.llm)
        self.synthesis_agent = SynthesisAgent(self.llm)
        self.enable_reasoning_llm = enable_reasoning_llm

    def run_single(self, task_id: str) -> Dict[str, Any]:
        task = self.loader.load_task_by_id(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found.")

        knowhow, match_result = self.matcher.match(task)
        plan = self.decompose_agent.plan(task, knowhow)
        plan_path = self._log_json(task_id, "plan", plan_to_dict(plan))

        executor = LangGraphPlanExecutor(
            planner=self.decompose_agent,
            knowhow=knowhow,
            enable_reasoning_llm=self.enable_reasoning_llm,
        )

        execution_result = asyncio.run(executor.run_plan(plan))
        execution_path = self._log_json(task_id, "execution", execution_result)

        synthesis_result = self.synthesis_agent.synthesize(plan, execution_result, knowhow)
        synthesis_path = self._log_json(task_id, "synthesis", synthesis_result_to_dict(synthesis_result))

        return {
            "task_id": task_id,
            "plan_log": str(plan_path),
            "execution_log": str(execution_path),
            "synthesis_log": str(synthesis_path),
            "final_answer": synthesis_result.final_answer,
            "match_info": {
                "matched_by": match_result.matched_by,
                "confidence": match_result.confidence,
                "domain": match_result.metadata.domain,
                "sub_domain": match_result.metadata.sub_domain,
            },
        }

    def _log_json(self, task_id: str, suffix: str, data: Dict[str, Any]) -> Path:
        path = self.log_dir / f"{task_id}_{suffix}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def plan_to_dict(plan) -> Dict[str, Any]:
    return {
        "task": {
            "task_id": plan.task.task_id,
            "task_type": plan.task.task_type,
            "target_entity": plan.task.target_entity,
            "target_metric": plan.task.target_metric,
            "event_description": plan.task.event_description,
            "resolved_time": plan.task.resolved_time,
            "answer_instructions": plan.task.answer_instructions,
        },
        "global_assumptions": plan.global_assumptions,
        "placeholders": plan.placeholders,
        "factors": [
            {
                "factor_name": factor.factor_name,
                "decision_logic": factor.decision_logic,
                "output_fields": factor.output_fields,
                "variables": factor.variables,
                "notes": factor.notes,
                "tasks": [
                    {
                        "id": task.id,
                        "kind": task.kind,
                        "goal": task.goal,
                        "tool": task.tool,
                        "params": task.params,
                        "compute": task.compute,
                        "writes": task.writes,
                    }
                    for task in factor.tasks
                ],
            }
            for factor in plan.factors
        ],
    }


def synthesis_result_to_dict(result) -> Dict[str, Any]:
    return {
        "task_id": result.task_id,
        "blended_expected_return": result.blended_expected_return,
        "blended_confidence": result.blended_confidence,
        "final_answer": result.final_answer,
        "narrative": result.narrative,
        "supporting_points": result.supporting_points,
        "factor_contributions": [
            {
                "name": contrib.name,
                "weight": contrib.weight,
                "outputs": contrib.outputs,
                "decision_logic": contrib.decision_logic,
                "summary": contrib.summary,
                "raw_tasks": contrib.raw_tasks,
            }
            for contrib in result.factor_contributions
        ],
        "raw_response": result.raw_response,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prediction Pipeline Orchestrator")
    parser.add_argument("--task-id", required=True, help="Task ID from standardized_data.jsonl")
    parser.add_argument("--data-path", help="Path to JSONL dataset")
    parser.add_argument("--log-dir", default="log", help="Directory to store logs")
    parser.add_argument("--enable-reasoning-llm", action="store_true", help="Enable LLM for reasoning fallback")
    args = parser.parse_args()

    pipeline = PredictionPipeline(
        data_path=args.data_path,
        log_dir=args.log_dir,
        enable_reasoning_llm=args.enable_reasoning_llm,
    )
    result = pipeline.run_single(args.task_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
