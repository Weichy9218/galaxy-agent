"""
使用 OpenRouterMiniClient 进行测试
使用 gpt-4.1-mini 模型，成本更低
"""

import json
import os
from datetime import datetime
from pathlib import Path

from core.llm.openrouter_mini_client import OpenRouterMiniClient
from knowhow_store.finance.stock_price import StockPriceKnowHow
from planner.decompose_agent import DecomposeAgent
from core.schemas.PredictionTask import PredictionTask

def save_plan_to_log(plan, task_id: str, log_dir: str = "log"):
    """
    将 DecompositionPlan 保存到 log 目录，格式化为可读的 JSON
    
    Args:
        plan: DecompositionPlan 对象
        task_id: 任务 ID
        log_dir: 日志目录路径
    """
    # 创建 log 目录
    Path(log_dir).mkdir(exist_ok=True)
    
    # 生成文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_id}_{timestamp}_plan.json"
    filepath = Path(log_dir) / filename
    
    # 将 DecompositionPlan 转换为可序列化的字典
    plan_dict = {
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
        "factors": []
    }
    
    # 转换每个 factor
    for factor in plan.factors:
        factor_dict = {
            "factor_name": factor.factor_name,
            "decision_logic": factor.decision_logic,
            "output_fields": factor.output_fields,
            "variables": factor.variables,
            "notes": factor.notes,
            "tasks": []
        }
        
        # 转换每个 task
        for task in factor.tasks:
            task_dict = {
                "id": task.id,
                "kind": task.kind,
                "goal": task.goal,
                "tool": task.tool,
                "params": task.params,
                "compute": task.compute,
                "writes": task.writes,
            }
            factor_dict["tasks"].append(task_dict)
        
        plan_dict["factors"].append(factor_dict)
    
    # 保存为格式化的 JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(plan_dict, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 计划已保存到: {filepath}")
    return filepath


# 使用 mini 客户端进行测试
llm = OpenRouterMiniClient(
    model="openai/gpt-4.1-mini",  # 使用 mini 版本降低成本
    max_tokens=16000,
    temperature=0  # 测试时使用 0 温度以获得确定性结果
)

knowhow = StockPriceKnowHow()

# 构建符合 PredictionTask 要求的参数
task = PredictionTask(
    task_id="t001",
    task_question="""The event to be predicted: "What will be the low of NVDA stock be for the day on 2025-10-30?"
    
IMPORTANT: Your final answer MUST end with \\boxed{YOUR_PREDICTION} format, where YOUR_PREDICTION is a numerical value representing the predicted stock price.""",
    metadata={
        "end_time": "2025-10-30",
        "dataset_name": "test_dataset"
    }
)

agent = DecomposeAgent(llm)

# 先看 prompt
print("=" * 80)
print("PROMPT 预览:")
print("=" * 80)
print(agent.preview_prompt(task, knowhow))
print("=" * 80)
print()

# 真正生成 Plan（返回 dataclass，可直接喂 SubAgent Runner）
print("正在生成计划...")
plan = agent.plan(task, knowhow)

print("=" * 80)
print("✅ 计划生成成功!")
print("=" * 80)

# 保存计划到 log 目录
log_file = save_plan_to_log(plan, task.task_id)
print()

# 打印计划摘要
print("=" * 80)
print("计划摘要:")
print("=" * 80)
print(f"任务 ID: {plan.task.task_id}")
print(f"任务类型: {plan.task.task_type}")
print(f"目标实体: {plan.task.target_entity}")
print(f"因子数量: {len(plan.factors)}")
print()

for i, factor in enumerate(plan.factors, 1):
    print(f"  {i}. {factor.factor_name}")
    print(f"     - 任务数: {len(factor.tasks)}")
    print(f"     - 输出字段: {', '.join(factor.output_fields)}")
print()

# 打印使用统计
print("=" * 80)
print("Token 使用统计:")
print("=" * 80)
stats = llm.get_usage_stats()
for key, value in stats.items():
    print(f"  {key}: {value}")
print()

print(f"📁 详细计划已保存至: {log_file}")

