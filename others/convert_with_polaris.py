import json
import os

# 输入输出路径（修改成你的）
input_path = "D:/polaris/POLARIS/evaluation/benchmarks/GSM8K/gsm8k/test.jsonl"      # GSM8K / MATH / MATH500 都支持
output_path = "D:/polaris/POLARIS/evaluation/benchmarks/GSM8K/gsm8k/test_polaris.jsonl"

def extract_problem(item):
    """自动识别题目字段：problem / question"""
    return item.get("problem") or item.get("question")

def extract_answer(item):
    """自动识别答案字段：answer / solution，并处理 #### 格式"""
    ans = item.get("answer") or item.get("solution") or ""
    ans = str(ans).strip()

    # GSM8K 格式: "some reasoning... #### 27"
    if "####" in ans:
        ans = ans.split("####")[-1].strip()

    return ans

def convert_item(item, index):
    problem = extract_problem(item)
    answer = extract_answer(item)

    # -------------------------
    # 生成 prompt 内容
    # -------------------------
    prompt_text = problem.strip()
    prompt_text += " Let's think step by step and output the final answer within \\boxed{}."

    prompt = [
        {
            "role": "user",
            "content": prompt_text
        }
    ]

    # -------------------------
    # reward model（必需字段）
    # -------------------------
    reward_model = {
        "ground_truth": answer,
        "style": "rule"
    }

    # -------------------------
    # extra_info（可选）
    # -------------------------
    extra_info = {
        "index": index,
        "unique_id": f"idx_{index}",
        "split": "train",
    }

    new_item = {
        "data_source": "math",
        "prompt": prompt,
        "ability": "math",
        "reward_model": reward_model,
        "extra_info": extra_info
    }

    return new_item


def main():
    new_items = []
    with open(input_path, "r", encoding="utf-8") as fin:
        items = [json.loads(line) for line in fin if line.strip()]

    for idx, item in enumerate(items):
        new_items.append(convert_item(item, idx))

    with open(output_path, "w", encoding="utf-8") as fout:
        for obj in new_items:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Finished: {len(new_items)} items written to {output_path}")


if __name__ == "__main__":
    main()
