import pandas as pd
import argparse
import json


def process_jsonl(jsonl_path):
    data_list = []
    with open(jsonl_path) as f:
        for line in f:
            data_list.append(json.loads(line))
    data_list.reverse()
    remove_set = {}
    for data in data_list:
        indice_list = data["index"]
        score_list = data["score"]
        for i in range(len(indice_list)):
            indice = indice_list[i]
            score = score_list[i]
            if indice not in remove_set:
                remove_set[indice] = [score]
            else:
                remove_set[indice].append(score)
   # Compute average score for each index and filter those with avg > 0.9
    result_indices = set()
    for indice, scores in remove_set.items():
        avg_score = sum(scores) / len(scores)
        if avg_score > 0.9:
            result_indices.add(indice)

    return result_indices

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="")
parser.add_argument('--experiment_name', type=str, default="")
parser.add_argument('--output', type=str, default="")
args = parser.parse_args()
parquet_file = args.data_path.replace(".parquet", "")
parquet_file_path = args.data_path
jsonl_path = f"{parquet_file}/{args.experiment_name}.jsonl"

rm_index_set = process_jsonl(jsonl_path)
df = pd.read_parquet(parquet_file_path)
print(df)
df_filtered = df[~df['extra_info'].apply(lambda x: x['index']).isin(rm_index_set)]
print(df_filtered)

df_filtered.to_parquet(args.output)
