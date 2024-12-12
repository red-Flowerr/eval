import numpy as np
import argparse
import os
from tqdm import tqdm

from src.evaluations.data_process import data_process
from src.evaluations.eval_util import *
from src.infer.infer_utils import *
from src.models.lm_infer import *

def generate_prompt(dataset, task, data):
    """
    add prompt template
    support benchmark: 
    --'mmlu', 'arc_c', 'hellaswag', 'bbh', 'aime', 'cmath', 
    --'gsm8k', 'math', 'gaokao2024', 'gaokao_math_qa', 'gaokao_math_cloze', 'amc23', 
    --'olympiadbench', 'gaokao2023en', 'college_math', 'omni_math', 'gpqa_diamond'
    """

    if "sys" in task:
        sys_prompt = "You're skilled at mimicking human reasoning and analyzing how and why it works. You can explain every detail clearly, step by step."
    else:
        sys_prompt = ""

    if dataset in ['mmlu', 'arc_c', 'hellaswag']:
        promt_template = choice_zero_shot_prompt.format(problem=data[key_map[args.dataset]])
    elif dataset in ['bbh']:
        promt_template = normal_zero_shot_prompt.format(problem=data[key_map[args.dataset]])
    elif dataset in ['data_from_train_math', 'aime','gsm8k', 'cmath', 'math', 'gaokao2024', 'gaokao_math_qa', 'gaokao_math_cloze', 'amc23', 'olympiadbench', 'gaokao2023en', 'college_math', 'omni_math', 'gpqa_diamond']:
        promt_template = boxed_zero_shot_prompt.format(problem=data[key_map[args.dataset]])

    return sys_prompt, promt_template.strip()


def parse_arguments():
    parser = argparse.ArgumentParser(description="vllm_inference")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--task", type=str, default='zs', choices=['zs', 'zs_o1', 'zs_cot', 'zs_sys'])

    parser.add_argument("--model", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--max_tokens", type=int, default=32768) 
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
    # init model
    
    engine = LocalLM(args) 
    # datasets = ['aime', 'math', 'gsm8k', 'data_from_train_math', 'gaokao2024', 'gaokao_math_qa', 'gaokao_math_cloze',
    #              'amc23', 'olympiadbench', 'gaokao2023en', 'college_math', 'omni_math', 'gpqa_diamond']
    # for dataset in datasets:
    #     args.dataset = dataset
    # prepare data
    if args.dataset == 'bbh':
        inputs = get_question_bbh(args)
    elif args.dataset == 'math':
        inputs = get_question_math(args)
    elif args.dataset == 'mmlu':
        inputs = get_question_mmlu(args)
    elif args.dataset == 'arc_c':
        inputs = get_question_arc_c(args)
    elif args.dataset == 'hellaswag':
        inputs = get_question_hellaswag(args)
    elif args.dataset == 'gaokao_math_qa':
        inputs = get_question_gaokao_math_qa(args)
    else:
        inputs = get_question(args)
    

    cuda_pieces = np.array_split(range(len(inputs)), args.cuda_num // args.tensor_parallel)

    if not os.path.exists(args.save_dir):
        try:
            os.makedirs(args.save_dir)
        except:
            print('makedirs error')
            pass
    print(len(cuda_pieces))
    print(args.cuda_start)
    print(args.cuda_ind // args.tensor_parallel)
    start = cuda_pieces[args.cuda_start + args.cuda_ind // args.tensor_parallel][0]
    
    end = cuda_pieces[args.cuda_start + args.cuda_ind // args.tensor_parallel][-1] + 1
    # data tokenize preprocess
    batch_prompts, batch_datas = [], []
    results = []
    for inpt in tqdm(inputs[start:end], desc="add template"):

        sys_prompt, prompt = generate_prompt(args.dataset, args.task, inpt) # add format prompt to question

        batch_datas.append(inpt)
        print(inpt)
        batch_prompts.append(engine.apply_chat_template(prompt, sys_prompt))

    # randomly sample generation
    if len(batch_prompts) > 0:
        for i in range(args.sample_num):
            engine.reset_seed(args.seed + i)
            resps = engine.batch_generate(batch_prompts)
        
            assert len(batch_datas) == len(resps)
            if i == 0:
                for ori_data, resp in zip(batch_datas, resps):
                    ori_data = dict(ori_data)
                    ori_data["response_" + str(i)] = resp
                    results.append(ori_data)
            else:
                assert len(results) == len(resps)
                for num, resp in enumerate(resps):
                    assert results[num][key_map[args.dataset]] == batch_datas[num][key_map[args.dataset]]
                    results[num]["response_"+str(i)] = resp

    save_jsonl(results, f"{args.save_dir}/{args.worker_id}-{args.cuda_ind // args.tensor_parallel + args.cuda_start}.json")

def test(args, prompt):
    args.worker_num = 1
    args.worker_id = 0
    args.cuda_num = 4
    args.tensor_parallel = 1
    args.sample_num = 1
    args.max_tokens = 16384
    args.save_name = "openo1-qwen2.5-7B-sft-template-ckpt1600"
    args.model = "/map-vepfs/huggingface/models/openo1-qwen2.5-7B-sft-fix-template/checkpoint-1600"

    engine = LocalLM(args)
    
    args.temperature = 0.0
    
    test_input = {
        "question": prompt,
        "context": "Geography"
    }
    
    input = engine.apply_chat_template(prompt, "")
    response_1 = engine.forward(input)
    print("Prompt:", prompt)
    print("LLM Output 1:", response_1)

if __name__ == "__main__":
    args = parse_arguments() 
    print('*****************************')
    print(args)
    print('*****************************')

    infer(args)
    print('infer end')



