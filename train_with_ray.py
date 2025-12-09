import subprocess

output = subprocess.check_output("ifconfig", encoding="utf-8")

# Extract IPv4 addresses
ip_addresses = []
for line in output.splitlines():
    if "inet " in line and "127.0.0.1" not in line:
        ip = line.split()[1]
        ip_addresses.append(ip)

# print("Available IP addresses:", ip_addresses)
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--sh', type=str, help='the training script', default="scripts/xtemplate/run_deepscaler_7b_16k_ray.sh")
parser.add_argument('--experiment_name', type=str, default="polaris-4b")
parser.add_argument('--model', type=str, default="/path/to/qwen3-4b")
parser.add_argument('--n_nodes', type=int, default=1)
parser.add_argument('--head', action="store_true")
parser.add_argument('--data_path', type=str, default=None)
args = parser.parse_args()
import time
import os

n_nodes = args.n_nodes

if args.head:
    ip = ip_addresses[1]
    start_cmd = f"ray start --head --port=6379 --node-ip-address={ip}"
    if os.path.exists(f"ray_address/{args.experiment_name}.ip"):
        print("============ error ===============")
        print(f"The experiment name has been used. Please run `rm ray_address/{args.experiment_name}.ip` and restart the training")
        exit(0)
    with open(f"ray_address/{args.experiment_name}.ip", "w") as f:
        f.write(f"{ip}")
    print(start_cmd)
    os.system(start_cmd)
    train_cmd = f"./{args.sh} --n_node {n_nodes} --experiment_name {args.experiment_name} --model {args.model} --data_path {args.data_path}"
    print(train_cmd)
    os.system(train_cmd)
else:
    while True:
        try:
            with open(f"ray_address/{args.experiment_name}.ip", "r") as f:
                ip = f.read()
            start_cmd = f"ray start --address={ip}:6379"
            print(start_cmd)
            os.system(start_cmd)
            break
        except:
            time.sleep(10)
    for i in range(60*100):
        time.sleep(60)