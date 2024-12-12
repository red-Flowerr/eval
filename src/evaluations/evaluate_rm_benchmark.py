import json
import random
from tqdm import tqdm
random.seed(0)
from collections import Counter, defaultdict
import numpy as np
import copy
from src.evaluations.eval_util import *
from src.evaluations.benchmark.bbh_eval import *
from src.evaluations.benchmark.gsm8k_eval import *
from src.evaluations.benchmark.math_eval import *
from src.evaluations.benchmark.mmlu_arc_hellaswag_eval import *

"""
根据RM infer的结果, 进行accuracy的计算
"""

bbh_evaluator = BBHEvaluator()
bbh_mcq_evaluator = BBHEvaluator_mcq()
gsm8k_evaluator = Gsm8kEvaluator()
math_evaluator = MATHEvaluator(version='v2')
mcq_evaluator = AccwithDetailsEvaluator()


def get_score(datas, dataset):
    scores = {}
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
                responses = {}
                for data in datas:
                    solution = data["target"]
                    ans = bbh_mcq_postprocess(solution, data['input']) # 处理ans格式
                    goldens.append(ans)
                    prompts.append(data['prompt'])
                    responses[data[key_map[dataset]]] = []
                    preds = []
                    ok_preds = {}
                    for i in range(SAMPLE_NUM): # sample number
                        response = data["response_"+str(i)]
                        responses[data[key_map[dataset]]].append(response)
                        pred = bbh_mcq_postprocess(response, data['input'])
                        preds.append(pred)
                        if i in [0, 1, 3, 7, 15, 31, 63]:
                            ok_preds[i+1] = copy.deepcopy(preds)
                    
                    for key, value in ok_preds.items():
                        if key not in predictions:
                            predictions[key] = [value]
                        else:
                            predictions[key].append(value)

                for key, value in predictions.items():
                    score = bbh_mcq_evaluator.score_sc_rm(prompts, responses, value, goldens)
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
                responses = {}
                for data in datas:
                    solution = data["target"]
                    goldens.append(solution)
                    prompts.append(data['prompt'])
                    responses[data[key_map[dataset]]] = []
                    
                    preds = []
                    ok_preds = {}
                    for i in range(SAMPLE_NUM): # sample number
                        response = data["response_"+str(i)]
                        responses[data[key_map[dataset]]].append(response)
                        pred = bbh_freeform_postprocess(response)
                        preds.append(pred)
                        if i in [0, 1, 3, 7, 15, 31, 63]:
                            ok_preds[i+1] = copy.deepcopy(preds)
                    
                    for key, value in ok_preds.items():
                        if key not in predictions:
                            predictions[key] = [value]
                        else:
                            predictions[key].append(value)

                for key, value in predictions.items():
                    if key == 64:
                        score = bbh_evaluator.score_sc_rm(prompts, responses, value, goldens)
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
        responses = {}
        
        for data in datas:  # 提取答案
            prompts.append(data[key_map[dataset]])
            responses[data[key_map[dataset]]] = []
            if dataset in ['mmlu', 'arc_c', 'hellaswag']:
                solution = data[golden_map[dataset]]
            elif dataset in ['math']:
                solution = math_postprocess_v2(data['solution'])
            elif dataset in ['gsm8k']:
                solution = gsm8k_dataset_postprocess(data["answer"])
            elif dataset in ['gsm8k']:
                solution = data["target"]
            
            goldens.append(solution)
        
            preds = []
            ok_preds = {}
            
            for i in range(SAMPLE_NUM): # sample number 提取generate生成的答案
                response = data["response_"+str(i)]
                responses[data[key_map[dataset]]].append(response)
                if dataset in ['mmlu', 'arc_c', 'hellaswag']:
                    pred = first_option_postprocess(response, options=option_map[dataset])
                elif dataset in ['math']:
                    pred = math_postprocess_v2(response)
                elif dataset in ['gsm8k']:
                    pred = math_postprocess_v2(response)

                preds.append(pred)
                if i in [0, 1, 3, 7, 15, 31, 63]:
                    # counter = Counter(preds)
                    # most_common_element, most_common_count = counter.most_common(1)[0]
                    ok_preds[i+1] = copy.deepcopy(preds)

            for key, value in ok_preds.items():
                if key not in predictions:
                    predictions[key] = [value]
                else:
                    predictions[key].append(value)

        # predictions是多个bon的字典，一个value对应的是所有问题的预测答案
        for key, value in predictions.items():
            if dataset in ['mmlu', 'arc_c', 'hellaswag']:
                score = mcq_evaluator.score_self_consistency(value, goldens, prompts)
            elif dataset in ['math']:
                score = math_evaluator.score_self_consistency(value, goldens)
            elif dataset in ['gsm8k']:
                score = compute_with_score(prompts, value, goldens, f"{base_path}/infer_result/BoN_rm_result/reward_score/bon_score_gsm.jsonl", key)
                # score = gsm8k_evaluator.score_self_consistency(value, goldens)
                # score = gsm8k_evaluator.score_sc_rm(prompts, responses, value, goldens)
                
            scores[key] = score

    return scores


def is_equal(pred, refer):
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

def compute_with_score(prompts, predictions, references, reward_file_path, sample_num):
    """
    prompts: 每个问题的prompt
    responses: 每个问题的responses
    predictions: 每个问题的预测
    references: 每个问题的标准答案
    reward_file_path: 存储预计算 reward 的 JSONL 文件路径
    sample_num: BoN 中选择的 N 值
    """
    
    # Step 1: 加载预计算的 rewards，按每 64 行为一组
    all_rewards = []
    with open(reward_file_path, 'r') as reward_file:
        all_rewards = [json.loads(line.strip()) for line in reward_file]

    if len(predictions) != len(references):
        return {'error': 'predictions and references have different length'}

    correct = 0
    total = 0
    details = []
    reward_index = 0  # 用于按 64 行分批索引

    # Step 2: 对每个 prompt 的 responses 使用 BoN 策略
    for prompt, pred_list, ref in tqdm(zip(prompts, predictions, references), total=len(prompts), desc="Processing"):
        # 读取当前 prompt 的前 N 个 response 的 reward
        response_rewards = all_rewards[reward_index : reward_index + sample_num]
        selected_preds = pred_list[:sample_num]  # 取对应的前 N 个 pred
        reward_index += 64  # 每个 prompt 的所有 64 个 response 已被读取完

        # 构建一个 pred -> rewards 的映射，只保留前 N 个 response
        pred_to_rewards = defaultdict(list)
        for pred, reward in zip(selected_preds, response_rewards):
            pred_to_rewards[pred].append(reward)  # 只取 reward 值

        # 聚类统计
        pred_counts = Counter(pred_to_rewards)
        scores = []
        for pred, count in pred_counts.items():

            cluster_score = sum(pred_to_rewards[pred])  # 聚类中所有 reward 求和
            scores.append({
                'pred': pred,
                'score': cluster_score
            })

        # 选择得分最高的 pred
        best = max(scores, key=lambda x: x['score'])
        best_pred = best['pred']

        # 记录结果和统计信息
        detail = {
            'pred': pred_list,
            'best_pred': best_pred,
            'answer': ref,
            'correct': False
        }
        total += 1

        if is_equal(best_pred, ref):
            correct += 1
            detail['correct'] = True

        details.append(detail)

    # 返回结果
    result = {
        'accuracy': 100 * correct / total,
        'details': details
    }
    return result

if __name__ == "__main__":
    
    models = [('OpenO1-Qwen-7B-v0.1-blend', 'zs')]

    # datasets = ['bbh', 'gsm8k', 'arc_c']
    datasets = ['gsm8k']
    base_path = os.getcwd() 
    for (model, task) in models:
            for dataset in datasets:
                print(f'------{dataset}-{model}-{task}--------')
                path = os.path.join(base_path, 'infer_result', 'blend_result', 'Bo64', dataset, model, task)
                datas = merge_data(path)

                scores = get_score(datas, dataset)
                if dataset in ['mmlu', 'arc_c', 'hellaswag', 'math', 'gsm8k']:
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
