import pandas as pd
import json

for split in ("train", "test"):
    df = pd.read_parquet(f"D:/polaris/POLARIS/evaluation/benchmarks/GSM8K/gsm8k/main/{split}-00000-of-00001.parquet")
    with open(f"{split}.jsonl", "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
