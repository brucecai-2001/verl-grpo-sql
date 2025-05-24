import os
import json
import csv
import random
import time

from datasets import Dataset
from chardet.universaldetector import UniversalDetector
from tqdm import tqdm

USER_PROMPT = """
You are a data analyst, and you need to generate an SQLite3 query statement based on the schema provided below.

## Schemas
The schemas are the following:
{schemas}

## Constraint
1.  require sql in sqllite3

User query: {query}
External knowledge: {evidence}
"""

TRAIN_SIZE = 1300

def detect_encoding(file_path):
    """检测文件编码"""
    detector = UniversalDetector()
    with open(file_path, 'rb') as f:
        for line in f:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result['encoding']

def get_schema_description_prompt(description_path: str):
    try:
        table_prompt = ""
        for root, dirs, files in os.walk(description_path):
            for file in files:
                if file.endswith('.csv'):
                    try:
                        detected_encoding = detect_encoding(os.path.join(root, file))

                        with open(os.path.join(root, file), 'r', encoding=detected_encoding) as csvfile:
                            reader = csv.DictReader(csvfile)
                            table_prompt += f"## Table: {file[:-4]}\n "
                            for row in reader:
                                original_column_name = row['original_column_name']
                                column_description = row['column_description']
                                data_format = row['data_format']
                                value_description = row['value_description']
                                table_prompt += f"column_name: {original_column_name}\n column_description: {column_description}\n data_format: {data_format}\n value_description: {value_description}\n\n"
                                
                    except FileNotFoundError:
                        print("错误：文件未找到，请检查文件路径。")
                    except Exception as e:
                        print(f"发生未知错误：{e}")
        return table_prompt
        
    except FileNotFoundError:
        print(f"错误：文件夹 {description_path} 未找到。")
    except Exception as e:
        print(f"错误：发生未知错误 {e}。")


if __name__ == '__main__':
    base_path = '/root/autodl-tmp/BIRD'
    database_path = base_path + '/' + 'database'
    question_path = base_path + '/dev.json'

    # get the schema description
    schema_descriptions = {}
    for root, dirs, files in os.walk(database_path):
        if 'database_description' in root:
            name = root.split('/')[-2]
            schema_description_prompt = get_schema_description_prompt(root)
            schema_descriptions[name] = schema_description_prompt

    dataset = []
    with open(question_path, 'r', encoding='utf-8') as file:
        questions = json.loads(file.read())
        progress_bar = tqdm(questions, desc="处理问题", unit="个")

        for q in progress_bar:
            time.sleep(0.05)

            # meta info
            db_id = q['db_id']
            db_path = database_path + '/' + db_id + '/' + f'{db_id}.sqlite'
            question = q['question']
            evidence = q['evidence']
            ground_truth_sql = q['SQL']
            
            # user question + schema description
            schemas_description = schema_descriptions[db_id]
            user_prompt = USER_PROMPT.format(schemas=schemas_description, query=question, evidence=evidence)

            dataset.append({
                "db_id": db_id,
                "raw_question": question,
                "user_prompt": user_prompt,
                "db_path": db_path,
                "ground_truth_sql": ground_truth_sql
            })

            progress_bar.set_postfix({"当前数据库": db_id})

    progress_bar.close()

    # split dataset
    random.seed(114514)
    random.shuffle(dataset)
    train_dataset = Dataset.from_list(dataset[:TRAIN_SIZE])
    eval_dataset = Dataset.from_list(dataset[TRAIN_SIZE:])

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            user_prompt = example.pop("user_prompt")
            ground_truth_sql = example.pop("ground_truth_sql")
            db_path = example.pop("db_path")

            data = {
                "data_source": 'BIRD',
                "prompt": [
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                "ability": "nl2sql",
                "reward_model": {"style": "rule", "ground_truth": {"ground_truth_sql": ground_truth_sql, "db_path": db_path}},
                "extra_info": {
                    "split": split,
                    "index": idx
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    eval_dataset = eval_dataset.map(function=make_map_fn("eval"), with_indices=True)

    train_dataset.to_parquet("./train.parquet")
    eval_dataset.to_parquet("./eval.parquet")
