import re
from openai import OpenAI
from datasets import load_dataset
from utils import query_database

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
CLIENT = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
EVAL_DATESET = "/root/autodl-tmp/BIRD/eval.parquet"

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
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        extra_body={
            "top_k": 20,
        },
    )
    return chat_response.choices[0].message.content

if __name__ == '__main__':
    eval_set = load_dataset(path=EVAL_DATESET)
    num = len(eval_set)
    correct = 0
    for data in eval_set:
        prompt = data["prompt"][0]["content"]
        db = data["reward_model"]["ground_truth"]["db_path"]
        gt_sql = data["reward_model"]["ground_truth"]["ground_truth_sql"]
        gt_res = query_database(db_path=db, sql=gt_sql)

        # rollout
        generated_str = invoke(prompt, MODEL)

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
            