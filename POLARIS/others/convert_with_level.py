import json
import random

# 旧数据
input_path = "D:/polaris/POLARIS/evaluation/benchmarks/MATH500/test.jsonl"
# 新数据
output_path = "D:/polaris/POLARIS/evaluation/benchmarks/MATH500/math500v2.jsonl"

# 提示模板（不会泄露答案，但提醒模型不要输出答案）
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
    """从 unique_id 自动生成 data_source"""
    parts = unique_id.split("/")
    if len(parts) >= 2:
        return parts[0] + "-" + parts[1]
    return unique_id.replace("/", "-")


def convert_item(item, index):
    problem = item["problem"]
    answer = str(item["answer"]).strip()
    subject = item.get("subject", "")
    level = item.get("level", None)
    unique_id = item.get("unique_id", f"idx_{index}")

    # ============================
    # 是否基于 level 添加提示？
    # 默认：level >= 4 加提示
    # ============================
    add_hint = False
    if level is not None and level >= 4:
        add_hint = True

    # -----------------------------
    # 生成 prompt 内容
    # -----------------------------
    prompt_text = problem.strip()
    prompt_text += " Let's think step by step and output the final answer within \\boxed{}."

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

    # -----------------------------
    # RL 标签
    # -----------------------------
    reward_model = {
        "ground_truth": answer,
        "style": "rule"
    }

    # -----------------------------
    # 额外信息，保留所有字段
    # -----------------------------
    extra_info = {
        "index": index,
        "split": "test",
        "subject": subject,
        "level": level,
        "unique_id": unique_id
    }

    data_source = build_data_source(unique_id)

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "math",
        "reward_model": reward_model,
        "extra_info": extra_info
    }


def main():
    new_items = []

    with open(input_path, "r", encoding="utf-8") as fin:
        old_list = [json.loads(line) for line in fin if line.strip()]

    for idx, old_item in enumerate(old_list):
        new_item = convert_item(old_item, idx)
        new_items.append(new_item)

    with open(output_path, "w", encoding="utf-8") as fout:
        for obj in new_items:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Converted {len(new_items)} items → {output_path}")
    print("Level-based hint added for items with level >= 4.")


if __name__ == "__main__":
    main()
