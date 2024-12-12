import numpy as np
import argparse
import os
from tqdm import tqdm
import argparse
from src.evaluations.eval_util import *
from src.infer.infer_utils import *
from src.models.lm_infer import *

def generate_prompt(dataset, task, data):
    # sys_prompt = "You're skilled at mimicking human reasoning and analyzing how and why it works. You can explain every detail clearly, step by step."
    sys_prompt = ""
    promt_template = rl_zero_shot_prompt.format(problem=data[key_map[args.dataset]])
    return sys_prompt, promt_template


def get_inputs(args):
    base_dir = os.getcwd()
    datas = read_data(f"{base_dir}/benchmark_data/Qwen2.5-72B-Instruct_exact_data_sample_9057_exact_gt_filter_digital_only.jsonl")
    inputs = []
    for idx, data in enumerate(datas):
        if idx % args.worker_num != args.worker_id:
            continue
        inputs.append(data)
    return inputs


def parse_arguments():
    parser = argparse.ArgumentParser(description="vllm_inference")
    parser.add_argument("--dataset", type=str, choices=['math', 'mmlu', 'gsm8k', 'bbh', 'arc_c', 'hellaswag', 'rl_data'])
    parser.add_argument("--task", type=str, default='zs', choices=['zs', 'zs_o1', 'zs_cot', 'zs_sys'])

    parser.add_argument("--model", type=str, default='/map-vepfs/models/openo1/llama3_1_blend_sft')
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="The parameter for repetition penalty. 1.0 means no penalty.")
    parser.add_argument("--temperature", type=float, default=0.2, help="The value used to modulate the next token probabilities.")
    parser.add_argument("--top_p", type=float, default=0.95, help="The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept.")
    parser.add_argument("--top_k", type=int, default=50, help="The number of highest probability vocabulary tokens to keep for top-k filtering.")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Exponential penalty to the length that is used with beam-based generation.")
    parser.add_argument("--model_revision", type=str, default="main", help="The specific model version to use (can be a branch name, tag name or commit id).")
    parser.add_argument("--hf_hub_token", type=str, default=None, help="Auth token to log in with Hugging Face Hub.")
    parser.add_argument("--use_fast_tokenizer", type=str2bool, default=True, help="Whether or not to use one of the fast tokenizer (backed by the tokenizers library).")
    parser.add_argument("--split_special_tokens", type=str2bool, default=False, help="Whether or not the special tokens should be split during the tokenization process.")
    parser.add_argument("--new_special_tokens", type=str, default=None, help="Special tokens to be added into the tokenizer.")
    parser.add_argument("--resize_vocab", type=str2bool, default=False, help="Whether or not to resize the tokenizer vocab and the embedding layers.")

    parser.add_argument("--sample_num", type=int, default=2) 
    parser.add_argument("--cuda_ind", type=int, default=0)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--cuda_start", type=int, default=0)
    parser.add_argument("--cuda_num", type=int, default=8)
    parser.add_argument("--load_in_8bit", type=str2bool, default=False)
    parser.add_argument("--use_typewriter", type=int, default=0)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--worker_num", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop_token_ids", type=list, default=[102])
    parser.add_argument("--dtype", type=str, default='float16')

    args = parser.parse_args()
    return args


def infer(args):
    engine = LocalLM(args)
    inputs = get_inputs(args)

    cuda_pieces = np.array_split(range(len(inputs)), args.cuda_num // args.tensor_parallel)
    
    if not os.path.exists(args.save_dir):
        try:
            os.makedirs(args.save_dir)
        except:
            print('makedirs error')
            pass
    start = cuda_pieces[args.cuda_start + args.cuda_ind // args.tensor_parallel][0]
    end = cuda_pieces[args.cuda_start + args.cuda_ind // args.tensor_parallel][-1] + 1

    # data tokenize preprocess
    batch_prompts, batch_datas = [], []
    results = []
    for inpt in tqdm(inputs[start:end], desc="add template"):
        ori_data = inpt

        sys_prompt, prompt = generate_prompt(args.dataset, args.task, inpt)

        batch_datas.append(ori_data)
        batch_prompts.append(engine.apply_chat_template(prompt, sys_prompt))
    
    # randomly sample generation
    if len(batch_prompts) > 0:
        for i in range(args.sample_num):
            engine.reset_seed(args.seed + i)
            resps = engine.batch_generate(batch_prompts)
    
            if i == 0:
                for ori_data, resp in zip(batch_datas, resps):
                    ori_data = dict(ori_data)
                    ori_data["response_" + str(i)] = resp
                    results.append(ori_data)
            else:
                for num, resp in enumerate(resps):
                    assert results[num][key_map[args.dataset]] == batch_datas[num][key_map[args.dataset]]
                    results[num]["response_"+str(i)] = resp

    save_jsonl(results, f"{args.save_dir}/{args.worker_id}-{args.cuda_ind // args.tensor_parallel + args.cuda_start}.json")

def filter_2_to_5():
    # 把
    models = [('OpenO1-Qwen-7B-v0.1', 'zs')]
    base_path = os.getcwd() 
    datasets = ['rl_data']
    for (model, task) in models:
            for dataset in datasets:
                print(f'------{dataset}-{model}-{task}--------')
                path = os.path.join(base_path, 'infer_result', 'filter_result', 'Bo8', dataset, model, task)
                datas = merge_data(path)
                

if __name__ == "__main__":
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    # infer response
    infer(args)
    # filter_2_to_5() 
    # response to reward
    