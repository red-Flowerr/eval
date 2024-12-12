# coding: utf-8
import jsonlines
import argparse
from datasets import load_dataset
# from .util import *
from src.evaluations.eval_util import *
from vllm import LLM, SamplingParams

import sys
MAX_INT = sys.maxsize

invalid_outputs = []
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False

def compare_output_with_answers(problem, output, answer):
    extract_ans_temp = output.strip()
    if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
        extract_ans = extract_ans_temp[0:-1]
    else:
        extract_ans = extract_ans_temp
    extract_ans = extract_ans.strip()
    if is_equiv(extract_ans, answer) or answer in extract_ans:  ## here we use in, is it correct？
        return True
    else:
        return False

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def test_hendrycks_math(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('promt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item['output']
            temp_ans = remove_boxed(last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('lenght ====', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)
    print('sampleing =====', sampling_params)
    llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)
    res_completions = []
    for idx, (prompt, prompt_answer) in enumerate(zip(batch_hendrycks_math_ins, hendrycks_math_answers)):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            prompt_temp = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer)
        results.append(res)

    acc = sum(results) / len(results)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('start===', start, ', end====',end)
    print('length====', len(results), ', acc====', acc)


def test_ours_math(data_name, data_path, type="cot", start=0, end=MAX_INT):
    if data_name == "math":
        data = load_dataset("lighteval/MATH", "all", cache_dir="./hf_datasets")

        if "train" in data:
            data = data["train"]  # only for train data

    # read our generation from save path
    generations = read_json(data_path)[start:end]

    results = []
    for idx, generation in enumerate(generations):
        problem = data[idx]["problem"]
        solution = data[idx]["solution"]
        temp_ans = remove_boxed(last_boxed_only_string(solution))
        completion = extract_output(generation["prediction"], type=type)  # extract content inside <output> for evaluation
        res = compare_output_with_answers(problem, completion, temp_ans)
        results.append(res)

    acc = sum(results) / len(results)
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('start===', start, ', end====',end)
    print('length====', len(results), ', acc====', acc)

def diff_data(data_name, data_path1, data_path2, start=0, end=MAX_INT):
    if data_name == "math":
        data = load_dataset("lighteval/MATH", "all", cache_dir="./hf_datasets")

        if "train" in data:
            data = data["train"]  # only for train data

    # read our generation from save path
    generations_one = read_json(data_path1)[start:end]
    generations_two = read_json(data_path2)[start:end]

    res_one_better, res_two_better = 0, 0
    results_one_better, results_two_better = [], []
    reflection_count, multi_reflection_count = 0, 0
    for idx, (gen1, gen2) in enumerate(zip(generations_one, generations_two)):
        problem = data[idx]["problem"]
        solution = data[idx]["solution"]
        temp_ans = remove_boxed(last_boxed_only_string(solution))
        completion_one = extract_output(gen1["prediction"], type="cot-zero-shot")  # extract content inside <output> for evaluation
        completion_two = extract_output(gen2["prediction"], type="output")

        # evaluate the results
        res_one = compare_output_with_answers(problem, completion_one, temp_ans)
        res_two = compare_output_with_answers(problem, completion_two, temp_ans)

        if res_one and not res_two: # which means res_one is correct while res_two is wrong
            res_one_better += 1
        
            results_one_better.append({
                "problem": problem,
                "solution": solution,
                "generation_one": gen1["prediction"],
                "generation_two": gen2["prediction"],
                "ground_truth": temp_ans,
                "answer_one": completion_one,
                "answer_two": completion_two
            })

        elif res_two and not res_one:
            res_two_better += 1

            if gen2.count("<reflection>") > 0:
                reflection_count += 1
            
            if gen2.count("<reflection>") > 1:
                multi_reflection_count += 1
            
            results_two_better.append({
                "problem": problem,
                "solution": solution,
                "generation_one": gen1["prediction"],
                "generation_two": gen2["prediction"],
                "ground_truth": temp_ans,
                "answer_one": completion_one,
                "answer_two": completion_two
            })

    save_json(results_one_better, "./case_studies/2024.9.18/results_one_better.json")
    save_json(results_two_better, "./case_studies/2024.9.18/results_two_better.json")
    
    print("===== basic info =====")
    print("res one (cot-zero-shot) better: {}".format(res_one_better))
    print("res two (ours) better: {}".format(res_two_better))
    print("reflection count: {}, multiple reflection count: {}".format(reflection_count, multi_reflection_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")  # model path
    parser.add_argument("--data_path", type=str, default='./outputs/2024.9.14/math_baseline_gpt4o_top_500.json')  # data path
    parser.add_argument("--type", type=str, default="output", help="cot/output")  # type of extract method
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=500)  # end index
    # parser.add_argument("--batch_size", type=int, default=50)  # batch_size
    # parser.add_argument("--tensor_parallel_size", type=int, default=1)  # tensor_parallel_size
    args = parser.parse_args()
    test_ours_math(data_name=args.data_name, data_path=args.data_path, type="output", start=args.start, end=args.end)

    # diff_data
    # diff_data(data_name=args.data_name, data_path1=args.data_path, data_path2='./outputs/math_ours_gpt4o_top_200.json')

