# verl/trainer/ppo/metric_utils_passk.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 在最上面
import os
import json
from collections import defaultdict


def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    from math import comb
    return 1 - comb(n - c, k) / comb(n, k)

def evaluate_passk_distribution(samples, k_values=[1, 2, 4, 8, 16, 32, 64, 128], save_dir=None, step=0):
    """
    samples: list of dict, each with keys { "n": int, "c": int }
    """
    results = {}
    for k in k_values:
        vals = [pass_at_k(s["n"], s["c"], k) for s in samples]
        vals = np.array(vals)
        stats = {
            "mean": np.mean(vals),
            "std": np.std(vals),
            "var": np.var(vals),
            "median": np.median(vals),
        }
        results[k] = stats

    # 绘制趋势图
    plt.figure(figsize=(8, 5))
    means = [results[k]["mean"] for k in k_values]
    stds = [results[k]["std"] for k in k_values]
    plt.plot(k_values, means, marker='o', label='mean')
    plt.fill_between(k_values, np.array(means)-np.array(stds),
                     np.array(means)+np.array(stds), alpha=0.2, label='±1 std')
    plt.xlabel("k")
    plt.ylabel("pass@k")
    plt.title(f"Pass@k Trend (step {step})")
    plt.legend()
    plt.grid(True)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, f"passk_step{step}.png")
        plt.savefig(img_path)
        with open(os.path.join(save_dir, f"passk_step{step}.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"[pass@k] Saved curve and stats to {save_dir}")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({f"pass@k_curve_step{step}": wandb.Image(img_path)}, step=step)
        except Exception as e:
            print(f"[pass@k] wandb upload failed: {e}")

    else:
        plt.show()

    ability_groups = defaultdict(list)
    for s in samples:
        ability = s.get("ability", "unknown")
        ability_groups[ability].append(s)

    grouped_results = {}
    for ability, group in ability_groups.items():
        grouped_results[ability] = {}
        for k in k_values:
            vals = [pass_at_k(s["n"], s["c"], k) for s in group]
            grouped_results[ability][k] = float(np.mean(vals)) if len(vals) else 0.0

    # ===== 绘制分难度曲线 =====
    if len(grouped_results) > 1:
        plt.figure(figsize=(8, 5))
        for ability, res in grouped_results.items():
            plt.plot(k_values, [res[k] for k in k_values], marker='o', label=ability)
        plt.xlabel("k")
        plt.ylabel("pass@k")
        plt.title(f"Pass@k by Ability (step {step})")
        plt.legend()
        plt.grid(True)

        if save_dir:
            img_path = os.path.join(save_dir, f"passk_difficulty_step{step}.png")
            plt.savefig(img_path)
            with open(os.path.join(save_dir, f"passk_difficulty_step{step}.json"), "w") as f:
                json.dump(grouped_results, f, indent=2, ensure_ascii=False)
            print(f"[pass@k] Saved per-difficulty curve and stats to {save_dir}")
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({f"pass@k_difficulty_step{step}": wandb.Image(img_path)}, step=step)
            except Exception as e:
                print(f"[pass@k_difficulty] wandb upload failed: {e}")

    return {"global": results, "by_ability": grouped_results}
