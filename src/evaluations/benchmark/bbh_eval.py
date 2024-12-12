import re
from collections import Counter
from tqdm import tqdm
import time
import requests
from src.evaluations.eval_util import *


bbh_multiple_choice_sets = [
    'temporal_sequences',
    'disambiguation_qa',
    'date_understanding',
    'tracking_shuffled_objects_three_objects',
    'penguins_in_a_table',
    'geometric_shapes',
    'snarks',
    'ruin_names',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_five_objects',
    'logical_deduction_three_objects',
    'hyperbaton',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'movie_recommendation',
    'salient_translation_error_detection',
    'reasoning_about_colored_objects',
]
bbh_free_form_sets = [
    'multistep_arithmetic_two',
    'navigate',
    'dyck_languages',
    'word_sorting',
    'sports_understanding',
    'boolean_expressions',
    'object_counting',
    'formal_fallacies',
    'causal_judgement',
    'web_of_lies',
]

def bbh_mcq_postprocess(text: str, ques: str) -> str:
    ans = text
    ans_line = ans.split('answer is')
    if len(ans_line) != 1:
        ans = ans_line[-1].strip()
        # ans = ans_line[1].strip()
    match = re.search(r'\(([A-Z])\)*', ans)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])', ans)
    if match:
        return match.group(1)
    if ans.endswith('.'):
        ans = ans[:-1]
    if ans.startswith(':'):
        ans = ans[1:]
    if ans.strip() in ques:
        ind = ques.find(ans)
        while ques[ind] not in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']:
            ind -=1
        return ques[ind]
    return ans

def bbh_freeform_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is')
    if len(ans_line) != 1:
        # ans = ans_line[1].strip()
        ans = ans_line[-1].strip()
    ans = ans.split('\n')[0]
    if ans.endswith('.'):
        ans = ans[:-1]
    if ans.startswith(':'):
        ans = ans[1:]
    return ans.strip()

class BBHEvaluator_mcq:

    def score(self, predictions, references, prompts):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        details = []
        cnt = 0
        for pred, ref, prompt in zip(predictions, references, prompts):
            detail = {'prompt': prompt ,'pred': pred, 'answer': ref, 'correct': False}
            for p in pred:
                if p == ref:
                    cnt += 1
                    detail['correct'] = True
                    break

            details.append(detail)

        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details}
    
    def score_self_consistency(self, predictions, references, prompts):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        details = []
        cnt = 0

        for pred_list, ref, prompt in zip(predictions, references, prompts):
            pred_counts = Counter(pred_list)
            
            most_common_pred, _ = pred_counts.most_common(1)[0]

            detail = {
                'prompt': prompt,
                'pred': pred_list,
                'most_common_pred': most_common_pred,
                'answer': ref,
                'correct': False
            }

            if most_common_pred == ref:
                cnt += 1
                detail['correct'] = True

            details.append(detail)

        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details}
    
    def score_sc_rm(self, prompts, responses, predictions, references):
        """
        prompts: 每个问题的prompt
        responses: 每个问题的responses
        predictions: 每个问题的预测
        references: 每个问题的标准答案
        """
        
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        correct = 0
        total = 0
        details = []

        # Step 1: 统一批量计算 reward
        all_rewards = []  # 用于存储所有 prompt 和 response 的 reward
        batch_prompts = []
        batch_responses = []

        for prompt in tqdm(prompts):
            # 将当前 prompt 的所有 responses 添加到批处理中
            for response in responses[prompt]:
                batch_prompts.append(prompt)
                batch_responses.append(response)

            
                # 每当批量达到 BATCH_SIZE，就调用 get_reward_batch 计算 reward
                if len(batch_prompts) >= BATCH_SIZE:
                    rewards = get_reward_batch(batch_prompts, batch_responses)
                    all_rewards.extend(rewards)  # 累积到 all_rewards 中
                    batch_prompts = []
                    batch_responses = []
                    # save
            with open("/map-vepfs/openo1/eval_benchmark/meta-reasoning-tuning/evaluations/bon_score_bbh.jsonl", 'w') as res:
                for item in all_rewards:
                    res.write(json.dumps(item[0]) + '\n')
                    
        # 处理剩余的不足 BATCH_SIZE 的部分
        if batch_prompts:
            rewards = get_reward_batch(batch_prompts, batch_responses)
            all_rewards.extend(rewards)
        


        # Step 2: 对每个问题的 responses 进行聚类处理
        reward_index = 0
        for prompt, pred_list, ref in tqdm(zip(prompts, predictions, references), total=len(prompts), desc="Processing"):
            pred_counts = Counter(pred_list)
            clusters = pred_counts.most_common()
            scores = []

            # 使用事先计算好的 reward，按顺序提取出当前 prompt 对应的所有 response reward
            response_rewards = []
            for response in responses[prompt]:
                response_rewards.append(all_rewards[reward_index])
                reward_index += 1

            # 计算每个聚类的总 reward 分数
            for cluster in clusters:
                pred, _ = cluster
                cluster_score = sum([item[0] for item in response_rewards]) # 对该聚类的所有 response reward 求和
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

            if best_pred == ref:
                correct += 1
                detail['correct'] = True

            details.append(detail)

        # 返回结果
        result = {
            'score': 100 * correct / total,
            'details': details
        }
        return result


class BBHEvaluator:

    def score(self, predictions, references, prompts):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        # predictions = [bbh_freeform_postprocess(pred) for pred in predictions]

        details = []
        cnt = 0
        for pred, ref, prompt in zip(predictions, references, prompts):
            detail = {'prompt': prompt ,'pred': pred, 'answer': ref, 'correct': False}
            for p in pred:
                if p == ref:
                    cnt += 1
                    detail['correct'] = True
                    break

            details.append(detail)

        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details}
    
    def score_self_consistency(self, predictions, references, prompts):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        details = []
        cnt = 0
        for pred_list, ref, prompt in zip(predictions, references, prompts):
            # 统计每个预测项的出现频率
            pred_counts = Counter(pred_list)
            
            # 获取出现频率最高的预测项
            most_common_pred, _ = pred_counts.most_common(1)[0]

            # 初始化详细记录
            detail = {
                'prompt': prompt,
                'pred': pred_list,
                'most_common_pred': most_common_pred,
                'answer': ref,
                'correct': False
            }

            # 判断最高频预测项是否等于参考答案
            if most_common_pred == ref:
                cnt += 1
                detail['correct'] = True

            # 将该条详细记录添加到 details 中
            details.append(detail)

        # 计算准确率
        score = cnt / len(predictions) * 100

        return {'score': score, 'details': details}
    

    def score_sc_rm(self, prompts, responses, predictions, references):
        """
        prompts: 每个问题的prompt
        responses: 每个问题的responses
        predictions: 每个问题的预测
        references: 每个问题的标准答案
        """
        
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}

        correct = 0
        total = 0
        details = []

        # Step 1: 统一批量计算 reward
        all_rewards = []  # 用于存储所有 prompt 和 response 的 reward
        batch_prompts = []
        batch_responses = []

        for prompt in tqdm(prompts):
            # 将当前 prompt 的所有 responses 添加到批处理中
            for response in responses[prompt]:
                batch_prompts.append(prompt)
                batch_responses.append(response)

            
                # 每当批量达到 BATCH_SIZE，就调用 get_reward_batch 计算 reward
                if len(batch_prompts) >= BATCH_SIZE:
                    rewards = get_reward_batch(batch_prompts, batch_responses)
                    all_rewards.extend(rewards)  # 累积到 all_rewards 中
                    batch_prompts = []
                    batch_responses = []
                    # save
            with open("/map-vepfs/openo1/eval_benchmark/meta-reasoning-tuning/evaluations/bon_score_bbh.jsonl", 'w') as res:
                for item in all_rewards:
                    res.write(json.dumps(item[0]) + '\n')
                    
        # 处理剩余的不足 BATCH_SIZE 的部分
        if batch_prompts:
            rewards = get_reward_batch(batch_prompts, batch_responses)
            all_rewards.extend(rewards)
        


        # Step 2: 对每个问题的 responses 进行聚类处理
        reward_index = 0
        for prompt, pred_list, ref in tqdm(zip(prompts, predictions, references), total=len(prompts), desc="Processing"):
            pred_counts = Counter(pred_list)
            clusters = pred_counts.most_common()
            scores = []

            # 使用事先计算好的 reward，按顺序提取出当前 prompt 对应的所有 response reward
            response_rewards = []
            for response in responses[prompt]:
                response_rewards.append(all_rewards[reward_index])
                reward_index += 1

            # 计算每个聚类的总 reward 分数
            for cluster in clusters:
                pred, _ = cluster
                cluster_score = sum([item[0] for item in response_rewards]) # 对该聚类的所有 response reward 求和
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
            if best_pred == ref:
                correct += 1
                detail['correct'] = True

            details.append(detail)

        # 返回结果
        result = {
            'score': 100 * correct / total,
            'details': details
        }
        return result
    
def get_reward(prompt, response, server_url="http://192.168.0.27:5002/get_reward"):
    queries = [f"{prompt} {response}"]  
    payload = {"query": queries}

    try:
        response = requests.post(server_url, json=payload)
        response.raise_for_status() 
        result = response.json() 
        return result.get("rewards", [0])[0]  
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return 0  
    
    
def get_reward_batch(prompts, responses, server_url="http://192.168.0.27:5002/get_reward"):
    
    # prompt_and_response = tokenizer.apply_chat_tempaltes
    # payload = {"query": [prompt_and_response, prompt_and_response, prompt_and_response...]}
    payload = {"query": [prompts, responses]}  # Sending the chat structure to the server

    while True:  # Retry loop
        try:
            response = requests.post(server_url, json=payload)
            response.raise_for_status()  # Check for request errors
            result = response.json()
            return result.get("rewards", [])  # Get the rewards from the response
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}, {1} 秒后重试...")
            time.sleep(1)  # Wait before retrying
