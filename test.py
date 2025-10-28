from core.llm.gpt5_client import GPT5Client  # 你的实现
from knowhow_store.finance.stock_price import StockPriceKnowHow
from planner.decompose_agent import DecomposeAgent
from core.schemas.PredictionTask import PredictionTask

llm = GPT5Client(max_tokens=16000)  # 增加 max_tokens 以允许完整的 JSON 响应
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
print(agent.preview_prompt(task, knowhow))

# 真正生成 Plan（返回 dataclass，可直接喂 SubAgent Runner）
plan = agent.plan(task, knowhow)
print(plan)
