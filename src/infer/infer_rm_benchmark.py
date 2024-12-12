import argparse
import numpy as np
import os
from tqdm import tqdm
from src.evaluations.data_process import * 
from src.models.rm_infer import *
import json

"""
输入的data是来自infer_benchmark采样后得到的response数据
"""

def parse_argument():
    parser = argparse.ArgumentParser(description="vllm_inference")
    parser.add_argument("--dataset", type=str, default='math', choices=['math', 'gsm8k', 'bbh', 'aime'])
    parser.add_argument("--task", type=str, default='rm')
    parser.add_argument("--model_name", type=str, default="/map-vepfs/huggingface/models/Qwen/Qwen2.5-Math-RM-72B")
    parser.add_argument("--dtype", type=str, default='float16')
    parser.add_argument("--pt_model_name", type=str, default="/map-vepfs/huggingface/models/OpenO1/OpenO1-Qwen-7B-v0.1/checkpoint-1000", help="pt model name")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--max_token", type=int, default=4096)
    parser.add_argument("--sample_num", type=int, default=64)
    parser.add_argument("--cuda_ind", type=int, default=0)
    parser.add_argument("--tensor_parallel", type=int, default=4)
    parser.add_argument("--cuda_start", type=int, default=0)
    parser.add_argument("--cuda_num", type=int, default=4)
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args

def save_jsonl(data, filename):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def infer(args):
    # 准备model
    engine = LocalLM(args)
    
    # 准备data 输入的是prompt+response的list
    inputs = data_process([args.dataset]) 
    print("数据处理完成")
    # cuda切片
    cuda_pieces = np.array_split(range(len(inputs)), args.cuda_num // args.tensor_parallel)

    if not os.path.exists(args.save_dir):
        try:
            os.makedirs(args.save_dir)
        except:
            print('makedirs error')
            pass
    
    start = cuda_pieces[args.cuda_start + args.cuda_ind // args.tensor_parallel][0]
    end = cuda_pieces[args.cuda_start + args.cuda_ind // args.tensor_parallel][-1] + 1

    # tokenize data
    batch_prompts, batch_datas = [], []
    results = []
    sliced_inputs = {k: v for i, (k, v) in enumerate(inputs.items()) if start <= i < end}
    for input_prompt, input_response in sliced_inputs.items():
        batch_datas.extend([{"prompt":input_prompt, "response": i} for i in input_response])
        batch_prompts.extend(engine.apply_chat_template(input_prompt, input_response))
    print("加载批数据完成")
    # get reward
    if len(batch_prompts) > 0:
        engine.reset_seed(args.seed)
        rewards = engine.batch_generate(batch_prompts)

        for prompt, reward in zip(batch_datas, rewards):
            prompt_with_reward = dict(prompt)
            prompt_with_reward["reward"] = reward
            results.append(prompt_with_reward)

    # save data 格式 dict：{prompt:xxx, response:xxx, reward:xxx}
    print(f"{args.save_dir}/{args.worker_id}-{args.cuda_ind // args.tensor_parallel + args.cuda_start}.json")
    save_jsonl(results, f"{args.save_dir}/{args.worker_id}-{args.cuda_ind // args.tensor_parallel + args.cuda_start}.json")


if __name__ == "__main__":
    args = parse_argument()
    infer(args)


    
