import json
import random

input_path = "D:/polaris/POLARIS/evaluation/benchmarks/GSM8K/gsm8k/train.jsonl"      # 旧数据
output_path = "D:/polaris/POLARIS/evaluation/benchmarks/GSM8K/gsm8k/trainv1.jsonl"      # 新数据

# 自定义一些提示模板
HINT_TEMPLATES = [
    "注意：最终答案要输出 {}。",
    "小心：请认真思考，正确答案是 {}。",
    "提示：你的最终答案应该是 {}。",
    "请将 {} 作为最终答案。",
    "提醒：{} 不是正确答案。",
    "以下是答案：{}，但不一定正确，请你认真分析",
    "请根据正确答案{}进行分析",
    "答案是{}，请分析后判断是否正确",
    "注意：正确答案是{}"
]

def build_data_source(unique_id: str) -> str:
    parts = unique_id.split("/")
    if len(parts) >= 2:
        return parts[0] + "-" + parts[1]
    return unique_id.replace("/", "-")


def convert_item(item, index, add_hint=False):
    problem = item.get("problem") or item.get("question")
    answer = item["answer"].split("####")[-1].strip()

    #answer = str(item["answer"]).strip()
    subject = item.get("subject", "")
    level = item.get("level", None)
    unique_id = item.get("unique_id", f"idx_{index}")

    # -------------------------------------------------------
    # 生成 prompt 文本
    # -------------------------------------------------------
    prompt_text = problem.strip()
    prompt_text += " Let's think step by step and output the final answer within \\boxed{}."

    # 若 add_hint=True，则对 prompt 增加干扰提示
    if add_hint:
        hint_template = random.choice(HINT_TEMPLATES)
        hint_text = " " + hint_template.format(answer)
        prompt_text += hint_text

    prompt = [
        {
            "content": prompt_text,
            "role": "user"
        }
    ]

    # -------------------------------------------------------
    # reward model
    # -------------------------------------------------------
    reward_model = {
        "ground_truth": answer,
        "style": "rule"
    }

    extra_info = {
        "index": index,
        "split": "test",
        "subject": subject,
        "level": level,            # ← 保留 level
        "unique_id": unique_id
    }

    data_source = build_data_source(unique_id)

    new_item = {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "math",
        "reward_model": reward_model,
        "extra_info": extra_info
    }
    return new_item


def main():
    new_items = []
    with open(input_path, "r", encoding="utf-8") as fin:
        old_list = [json.loads(line) for line in fin if line.strip()]

    total = len(old_list)
    indices_with_hint = set(random.sample(range(total), int(total * 0.25)))  # 25%

    for idx, old_item in enumerate(old_list):
        add_hint = idx in indices_with_hint
        new_item = convert_item(old_item, idx, add_hint=add_hint)
        new_items.append(new_item)

    with open(output_path, "w", encoding="utf-8") as fout:
        for obj in new_items:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Total {total} items converted.")
    print(f"{len(indices_with_hint)} items (20%) include hint.")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
