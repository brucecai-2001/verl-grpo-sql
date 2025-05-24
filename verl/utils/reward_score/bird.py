import re
from nl2sql.utils import query_database

def compute_score(solution_str: str, ground_truth: dict) -> float:
    """
        Compute reward for NL2SQL task.
        Consists of format reward and accuracy reward.

    Args:
        solution_str (str): rollout response
        ground_truth (dict): ground truth sql

    Returns:
        float: 0.2 * format_reward + 0.8 * acc_reward
    """

    # extract sql in <think> cot </think> <answer> ```sql SQL ``` </answer>
    pattern = re.compile(r"<think>\s*([\s\S]+?)\s*</think>\s*<answer>\s*```sql\s*([\s\S]+?)\s*```\s*</answer>", re.DOTALL)
    match_result = re.fullmatch(pattern, solution_str)
    # format reward
    format_reward = 1.0 if match_result else 0.0

    if match_result is not None:
        rollout_sql = match_result.group(2)
    else:
        # try to extract ```sql SQL ```
        pattern2 = re.compile(r"```sql\s*([\s\S]+?)\s*```", re.DOTALL)
        match_result2 = re.search(pattern2, solution_str)
        if match_result2 is not None:
            rollout_sql = match_result2.group(1)
            format_reward = 0.5
        else:
            return 0.0

    # acc reward
    rollout_sql = match_result.group(2)
    gt_sql = ground_truth["ground_truth_sql"]
    db_path = ground_truth["db_path"]

    gt_res = query_database(db_path, gt_sql)
    rollout_res = query_database(db_path, rollout_sql)
    acc_reward = 1.0 if gt_res == rollout_res else 0.0

    return 0.2 * format_reward + 0.8 * acc_reward
