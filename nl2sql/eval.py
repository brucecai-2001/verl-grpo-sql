import re

from openai import OpenAI
from datasets import load_dataset
from utils import query_database
from tqdm import tqdm  # 导入tqdm库用于进度条显示

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
deepseek_api_base = "https://api.deepseek.com"
deepseek_api_key = ""

CLIENT = OpenAI(
    api_key=deepseek_api_key,
    base_url=deepseek_api_base,
)

LOCAL_MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
DEEPSEEK_MODEL = "deepseek-chat"
EVAL_DATESET = {"eval": "/root/autodl-tmp/BIRD/eval.parquet"}

def invoke(prompt: str, model_pth: str) -> str:
    """
        Start a LLM call
    Returns:
        str: response
    """
    chat_response = CLIENT.chat.completions.create(
        model=model_pth,
        messages=[
            {"role": "user", "content": prompt},
        ],
        #max_tokens=65536,
        temperature=0.6,
        #top_p=0.95,
        #extra_body={
        #    "top_k": 20,
        #},
        stream=False
    )
    return chat_response.choices[0].message.content

if __name__ == '__main__':
    eval_set = load_dataset("parquet", data_files=EVAL_DATESET, split="eval")
    num = len(eval_set)
    correct = 0
    
    # 使用tqdm包装迭代器，创建进度条
    for data in tqdm(eval_set, desc="Evaluating", total=num):
        prompt = data["prompt"][0]["content"]
        db = data["reward_model"]["ground_truth"]["db_path"]
        gt_sql = data["reward_model"]["ground_truth"]["ground_truth_sql"]
        gt_res = query_database(db_path=db, sql=gt_sql)

        # rollout
        generated_str = invoke(prompt, DEEPSEEK_MODEL)

        # parse
        pattern = re.compile(r"```sql\s*([\s\S]+?)\s*```", re.DOTALL)
        match_result = re.search(pattern, generated_str)
        if match_result is not None:
            generated_sql = match_result.group(1)
            res = query_database(db_path=db, sql=generated_sql)
            if res == gt_res:
                correct += 1
        else:
            continue
    
    # 计算准确率并显示最终结果
    accuracy = correct / num if num > 0 else 0
    print(f"\nEvaluation completed! Accuracy: {accuracy:.2%} ({correct}/{num})")
