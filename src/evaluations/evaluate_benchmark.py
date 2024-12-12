import random
random.seed(0)
import numpy as np
import copy

from src.evaluations.eval_util import *
from src.evaluations.benchmark.bbh_eval import *
from src.evaluations.benchmark.gsm8k_eval import *
from src.evaluations.benchmark.math_eval import *
from src.evaluations.benchmark.mmlu_arc_hellaswag_eval import *
from src.evaluations.benchmark.aime_eval import *
from src.evaluations.benchmark.gaokao_eval import *
from src.evaluations.benchmark.amc23_eval import *
from src.evaluations.benchmark.olympiadbench_eval import *
from src.evaluations.benchmark.normal_eval import *
from src.evaluations.benchmark.cmath_eval import *
BATCH_SIZE=64 
SAMPLE_NUM=1 # 评测bon的n在这里修改
EVAL_APPROACH = "sc" # 评测方法在这里修改. option：sc or passK
"""
根据infer的结果, 进行accuracy的计算
"""

bbh_evaluator = BBHEvaluator()
bbh_mcq_evaluator = BBHEvaluator_mcq()
gsm8k_evaluator = Gsm8kEvaluator()
math_evaluator = MATHEvaluator(version='v2')
mcq_evaluator = AccwithDetailsEvaluator()
amie_evaluator = AIMEEvaluator()
gaokao_evaluator = GAOKAOEvaluator()
amc23_evaluator = AMC23Evaluator()
olympiadbench_evaluator = OlympiadbenchEvaluator()
normal_evaluator = NormalEvaluator()
cmath_evaluator = CMATHEvaluator()
def get_score(datas, dataset):
    scores = {}
    num_none = 0
    if dataset in ['bbh']:
        data_dict = {}
        for data in datas: # 按type进行聚类
            if data['type'] not in data_dict.keys():
                data_dict[data['type']] = [data]
            else:
                data_dict[data['type']].append(data)

        for name, datas in data_dict.items():
            if name in bbh_multiple_choice_sets:
                goldens = []
                predictions = {}
                prompts = []
                for data in datas:
                    solution = data["target"]
                    ans = bbh_mcq_postprocess(solution, data['input']) # 处理ans格式
                    goldens.append(ans)
                    prompts.append(data['prompt'])
                    preds = []
                    ok_preds = {}
                    for i in range(SAMPLE_NUM): # sample number
                        response = data["response_"+str(i)]
                        pred = bbh_mcq_postprocess(response, data['input'])
                        preds.append(pred)
                        if i in [0, 1, 3, 7, 15, 31, 63]:
                            ok_preds[i+1] = copy.deepcopy(preds)
                    
                    for key, value in ok_preds.items():
                        if key not in predictions:
                            predictions[key] = [value]
                        else:
                            predictions[key].append(value)
                if EVAL_APPROACH == 'sc':
                    for key, value in predictions.items():
                        score = bbh_mcq_evaluator.score_self_consistency(value, goldens, prompts)
                        score_value = score['score']
                        if name not in scores:
                            scores[name] = {key: [score_value]}
                        else:
                            if key not in scores[name]:
                                scores[name][key] = [score_value]
                            else:
                                scores[name][key].append(score_value)
                elif EVAL_APPROACH == 'passk':
                    for key, value in predictions.items():
                        score = bbh_mcq_evaluator.score(value, goldens, prompts)
                        score_value = score['score']
                        if name not in scores:
                            scores[name] = {key: [score_value]}
                        else:
                            if key not in scores[name]:
                                scores[name][key] = [score_value]
                            else:
                                scores[name][key].append(score_value)

            if name in bbh_free_form_sets:
                goldens = []
                predictions = {}
                prompts = []
                for data in datas:
                    solution = data["target"]
                    goldens.append(solution)
                    prompts.append(data['prompt'])
                    
                    preds = []
                    ok_preds = {}
                    for i in range(SAMPLE_NUM): # sample number
                        response = data["response_"+str(i)]
                        pred = bbh_freeform_postprocess(response)
                        preds.append(pred)
                        if i in [0, 1, 3, 7, 15, 31, 63]:
                            ok_preds[i+1] = copy.deepcopy(preds)
                    
                    for key, value in ok_preds.items():
                        if key not in predictions:
                            predictions[key] = [value]
                        else:
                            predictions[key].append(value)
                if EVAL_APPROACH == 'sc':
                    for key, value in predictions.items():
                        score = bbh_evaluator.score_self_consistency(value, goldens, prompts)
                        score_value = score['score']
                        if name not in scores:
                            scores[name] = {key: [score_value]}
                        else:
                            if key not in scores[name]:
                                scores[name][key] = [score_value]
                            else:
                                scores[name][key].append(score_value)
                elif EVAL_APPROACH == 'passk':
                    for key, value in predictions.items():
                        score = bbh_evaluator.score(value, goldens, prompts)
                        score_value = score['score']
                        if name not in scores:
                            scores[name] = {key: [score_value]}
                        else:
                            if key not in scores[name]:
                                scores[name][key] = [score_value]
                            else:
                                scores[name][key].append(score_value)
            
    else:
        predictions = {}
        goldens = []
        prompts = []
        
        for data in datas:  
            # 1.extract prompt
            prompts.append(data[key_map[dataset]])
            # 2.extract gt
            if dataset in ['mmlu', 'arc_c', 'hellaswag']:
                solution = data[golden_map[dataset]]
            elif dataset in ['math']:
                solution = data[golden_map[dataset]]
            elif dataset in ['gsm8k']:
                solution = gsm8k_dataset_postprocess(data[golden_map[dataset]])
            elif dataset in ['aime']:
                solution = data[golden_map[dataset]]
            elif dataset in ['cmath']:
                solution = data[golden_map[dataset]]
            elif dataset in ['gaokao2023en']:
                solution = gaokao2023en_dataset_postprocess(data[golden_map[dataset]])
            elif dataset in ['olympiadbench']:
                solution = olympiadbench_dataset_postprocess(data[golden_map[dataset]])
            elif dataset in ['college_math']:
                solution = college_math_dataset_postprocess(data[golden_map[dataset]])
            elif dataset in ['gaokao_math_cloze']:
                solution = gaokao_math_cloze_dataset_postprocess(data[golden_map[dataset]])
            elif dataset in ['gaokao2024','gaokao_math_qa', 'omni_math', 'amc23', 'gpqa_diamond']:
                solution = data[golden_map[dataset]]
            else:
                solution = data["response"].strip()

            if dataset not in ["carp_en"]:
                solution = strip_string_new(solution)
            else:
                solution = (
                    solution.replace("\\neq", "\\ne")
                    .replace("\\leq", "\\le")
                    .replace("\\geq", "\\ge")
                )
                
            goldens.append(solution)
        
            preds = []
            ok_preds = {}
            # 3.extract the llm of response
            for i in range(SAMPLE_NUM): # sample number 提取generate生成的答案
                response = data["response_"+str(i)]
                if dataset in ['mmlu', 'arc_c', 'hellaswag']:
                    pred = first_option_postprocess(response, options=option_map[dataset])
                elif dataset in ['data_from_train_math', 'gsm8k', 'math', 'gaokao2024', 'gaokao_math_qa', 'gaokao_math_cloze', 'amc23', 'olympiadbench', 'gaokao2023en', 'college_math', 'omni_math', 'gpqa_diamond']:
                    pred = math_postprocess_v2(response)
                elif dataset in ['aime']:
                    pred = aime_postprocess(response)
                elif dataset in ['cmath']:   
                    pred = math_postprocess_v2(response)
                elif dataset in ['gaokao2023en']:
                    pred = gaokao2023en_extract(response)
                if pred == "0x3f3f3f3f":
                    num_none += 1
                preds.append(pred)
                if i in [0, 1, 3, 7, 15, 31, 63]: # Bo1, Bo2, Bo4, Bo8, Bo16, Bo32, Bo64
                    ok_preds[i+1] = copy.deepcopy(preds)

            for key, value in ok_preds.items():
                if key not in predictions:
                    predictions[key] = [value]
                else:
                    predictions[key].append(value)
        # 4.compute score
        if EVAL_APPROACH == 'sc':
            # predictions是多个bon的字典，一个value对应的是所有问题的预测答案
            for key, value in predictions.items():
                if dataset in ['mmlu', 'arc_c', 'hellaswag', 'gaokao_math_qa']:
                    score = mcq_evaluator.score_self_consistency(value, goldens, prompts)
                elif dataset in ['data_from_train_math', 'math', 'college_math', 'omni_math']:
                    score = math_evaluator.score_self_consistency(value, goldens)
                elif dataset in ['cmath']:   
                    score = cmath_evaluator.score_self_consistency(value, goldens)
                elif dataset in ['gsm8k']:
                    score = gsm8k_evaluator.score_self_consistency(value, goldens)
                elif dataset in ['aime']:
                    score = amie_evaluator.score_self_consistency(value, goldens)
                elif dataset in ['gaokao2024', 'gaokao2023en']:
                    score = gaokao_evaluator.score_self_consistency(value, goldens)
                elif dataset in ['amc23']:
                    score = amc23_evaluator.score_self_consistency(value, goldens)
                elif dataset in ['olympiadbench']:
                    score = olympiadbench_evaluator.score_self_consistency(value, goldens)
                else:
                    score = normal_evaluator.score(value, goldens)
                scores[key] = score
                
                # print(value)
                print(f"截断率：{100*num_none/len_dict[dataset]:.2f}%")
        elif EVAL_APPROACH == 'passk':
            # predictions是多个bon的字典，一个value对应的是所有问题的预测答案
            for key, value in predictions.items():
                if dataset in ['mmlu', 'arc_c', 'hellaswag', 'gaokao_math_qa']:
                    score = mcq_evaluator.score(value, goldens, prompts)
                elif dataset in ['math']:
                    score = math_evaluator.score(value, goldens)
                elif dataset in ['gsm8k']:
                    score = gsm8k_evaluator.score(value, goldens)
                elif dataset in ['aime']:
                    score = amie_evaluator.score(value, goldens)
                elif dataset in ['gaokao2024_I', 'gaokao2024_II', 'gaokao2024_mix', 'gaokao2023en']:
                    score = gaokao_evaluator.score(value, goldens)
                elif dataset in ['amc23']:
                    score = amc23_evaluator.score(value, goldens)
                else:
                    score = normal_evaluator.score(value, goldens)
                scores[key] = score
                print(f"截断数：{num_none}")
    return scores


if __name__ == "__main__":
    # python -m src.evaluations.evaluate_benchmark
    models = [
        # ('expv3_greedy', 'zs')
        # ('llama-8b-sft-pro-data', 'zs')
        # ('llama-8b-instruct', 'zs')
        ('expv5_step10_greedy', 'zs')
        # ('open-o1-llama3.1-8B-sft-v0.2-1000-refinforce-v0.1_Bo64-difficulty-mixture-v3-all-medium-no-numina-5k_32gpu_Bo32-Ep1-len-8k', 'zs')
        # ('openo1-llama-8B-sft-v0.2', 'zs')
        # ('open-o1-llama3.1-8B-sft-v0.2-1000-refinforce-v0.1-refinforce-v0.1_Bo64-difficulty-mixture-v2-hard-5k_Bo32-Ep1-len-8k_step20', 'zs'), 
        # ('open-o1-llama3.1-8B-sft-v0.2-1000-refinforce-v0.1_Bo64-difficulty-mixture-v2-hard-5k_Bo32-Ep1-len-8k_step30', 'zs')
        # ('open-o1-llama3.1-8B-sft-v0.2-1000-refinforce-v0.1-refinforce-v0.1_Bo64-difficulty-mixture-v2-hard-5k_32gpu_Bo32-Ep1-len-8k-final', 'zs')
        # ('llama3.1-8B-sft-v0.2_refinforce-v0.1_data-mix-v3-5k_eval_debug', 'zs')
    ]


    base_path = os.getcwd() 
    # datasets = ['cmath']

    datasets = ['data_from_train_math','cmath','aime','gsm8k','math','gaokao2024', 'gaokao2023en','olympiadbench', 'gaokao_math_qa', 'gaokao_math_cloze', 'amc23', 'college_math', 'omni_math', 'gpqa_diamond']
    # datasets = ['gpqa_diamond']
    # datasets = ['gaokao2024']
    # datasets = ['gaokao2023en']
    # datasets = ['omni_math']
    # datasets = ['gaokao_math_cloze']
    # datasets = [
    # 'mmlu', 'arc_c', 'hellaswag', 'bbh', 'aime', 'gsm8k','math',
    # 'cmath', 'gaokao2024', 'gaokao_math_qa', 'gaokao_math_cloze', 
    # 'amc23', 'olympiadbench', 'gaokao2023en', 'college_math', 'omni_math', 'gpqa_diamond']
    len_dict = {
        'arc_c': 1172,
        'mmlu': 14042,
        'hellaswag': 10042,
        "math": 5000,
        'gsm8k': 1319,
        'bbh': 6511,
        'data_from_train_math': 3600,
        'aime': 90,
        'amc23': 40,
        'cmath': 600,
        'college_math': 2818,
        'gaokao2023en': 385,
        'gaokao2024': 133,
        'gaokao_math_qa': 351,
        'gaokao_math_cloze': 118, 
        'gpqa_diamond': 198, 
        'olympiadbench': 675, 
        'omni_math': 4428,
    } # 方便比对
    for (model, task) in models:
            for dataset in datasets:
                print(f'------{dataset}-{model}-{task}--------')
                # path = os.path.join(base_path, 'infer_result_release', 'sft_result', model, 'Bo1', dataset, task)
                # path = os.path.join(base_path, 'infer_result_release', 'ppo_result', model, 'Bo1', dataset, task)
                # path = os.path.join(base_path, 'infer_result_release', 'ppo_result', model, 'Bo1', dataset, task)
                path = os.path.join(base_path, 'infer_result_release', 'reinforce_result', model, 'Bo1', dataset, task)
                
                datas = merge_data(path)
                assert len(datas) == len_dict[dataset], f"当前 {dataset} 的条数为：{len(datas)}"
                scores = get_score(datas, dataset)
                if dataset in ['data_from_train_math', 'mmlu', 'arc_c', 'hellaswag', 'math', 'gsm8k', 'aime', 'cmath', 'gaokao2024', 'gaokao_math_cloze', 'gaokao_math_qa', 'gaokao2023en', 'amc23', 'olympiadbench', 'college_math', 'omni_math', 'gpqa_diamond']:
                    for key, value in scores.items():
                        print('Bo{}: {}'.format(key, value['accuracy']))
                elif dataset in ['bbh']:
                    bbh_scores = {}
                    for key, value in scores.items():
                        for k, v in value.items():
                            if k not in bbh_scores:
                                bbh_scores[k] = v
                            else:
                                bbh_scores[k].append(v[0])
                    for ke, va in bbh_scores.items():
                        print('Bo{}: {}'.format(ke, np.mean(va)))
                # path = os.path.join(base_path, 'infer_result_release', 'ppo_result', model, 'Bo1', dataset)
                # for key, value in scores.items():
                #     details_path = os.path.join(path, f'{task}_eval_details_{key}_2.jsonl')  # 创建每个 key 对应的文件路径
                #     with open(details_path, 'w') as jsonl_file:  # 创建 JSONL 文件
                #         for detail in value['details']:  # 遍历每个 detail
                #             jsonl_file.write(json.dumps(detail) + '\n')  # 按条写入 JSONL 格式
