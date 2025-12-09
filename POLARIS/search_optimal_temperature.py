import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=float, default=1.4)
parser.add_argument("--end", type=float, default=1.6)
parser.add_argument("--step", type=float, default=0.05)
parser.add_argument("--model", type=str, default="")
parser.add_argument("--n", type=int, default=16)
parser.add_argument("--max_length", type=int, default=50000)
parser.add_argument("--output_dir", type=str, default="t-search")

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
name = args.model.split("/")[-1]
if name == "":
    name = args.model.split("/")[-2]


num_steps = int((args.end - args.start) / args.step) + 1
temperatures = [round(args.start + i * args.step, 2) for i in range(num_steps)]

print("============== Info ==============")
print( "You are going to search:",  temperatures)


for temperature in temperatures:
    temperature = round(temperature, 2)
    print("Testing the model with temperature ", temperature)
    cmd = f"python scripts/eval/eval_vllm_aime24.py --model {args.model} --experiment_name {name} --n {args.n} --max_length {args.max_length} --t {temperature} --output {args.output_dir}  --k -1"
    print(cmd)
    os.system(cmd)
    
