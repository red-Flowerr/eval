import re
from collections import Counter
from tqdm import tqdm
import time
import requests
from src.evaluations.eval_util import *

class Gsm8kEvaluator:

    def is_equal(self, pred, refer):
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            for i_ in i:
                if self.is_equal(i_, j):
                    correct += 1
                    detail['correct'] = True
                    break
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result

    def score_self_consistency(self, predictions, references):
        # 检查 predictions 和 references 长度是否一致
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different length'
            }

        correct = 0
        total = 0
        details = []

        # 遍历 predictions 和 references 的每对 (pred, ref)
        for pred_list, ref in zip(predictions, references):

            pred_counts = Counter(pred_list) # Counter({值: 次数})
            
            most_common_pred, _ = pred_counts.most_common(1)[0]

            # 初始化详细记录
            detail = {
                'pred': pred_list,
                'most_common_pred': most_common_pred,
                'answer': ref,
                'correct': False
            }
            total += 1

            # 判断最常见预测项是否等于参考答案
            if self.is_equal(most_common_pred, ref):
                correct += 1
                detail['correct'] = True

            # 将该条详细记录添加到 details 中
            details.append(detail)

        # 计算准确率
        result = {
            'accuracy': 100 * correct / total,
            'details': details
        }

        return result
    

    def score_sc_rm_ori(self, prompts, responses, predictions,references):
        """
        prompts: 每个问题的prompt
        responses: 每个问题的responses
        references: 每个问题的标准答案
        """
        
        if len(predictions) != len(references):
                    return {
                        'error': 'predictions and references have different length'
                    }

        correct = 0
        total = 0
        details = []

        for prompt, pred_list, ref in tqdm(zip(prompts, predictions, references), total=len(prompts), desc="Processing"):
            # 每个问题的层面: 需要prompt，response给rm；prediction，reference给cluser

            pred_counts = Counter(pred_list)
            clusters = pred_counts.most_common()
            scores = []
            score = 0
            response = responses[prompt]
            for cluster in clusters:
                # 每个问题的不同response层面
                pred,_ = cluster
                # 取prompt和response送给rm，rm给出得分, 计算每个response的得分，然后求和
                for resp in response:
                    score += get_reward(prompt, resp)

                scores.append({
                    'pred': pred, 
                    'score': score
                })
            
            # 取得score总和分数最大时对应的pred
            best = max(scores, key=lambda x: x['score'])
            best_pred = best['pred']

            detail = {
                'pred': pred_list,
                'best_pred': best_pred,
                'answer': ref,
                'correct': False
            }
            total += 1

            if self.is_equal(best_pred, ref):
                correct += 1
                detail['correct'] = True
            
            details.append(detail)

        result = {
            'accuract': 100 * correct / total,
            'details': details
        }
        return result


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
            with open("/map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark/infer_result/BoN_rm_result/reward_score/bon_score_gsm.jsonl", 'w') as res:
                for item in all_rewards:
                    res.write(json.dumps(item[0]) + '\n')
                    
        # 处理剩余的不足 BATCH_SIZE 的部分
        if batch_prompts:
            rewards = get_reward_batch(batch_prompts, batch_responses)
            all_rewards.extend(rewards)

        # 对每个问题的 responses 进行聚类处理
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

            if self.is_equal(best_pred, ref):
                correct += 1
                detail['correct'] = True

            details.append(detail)

        # 返回结果
        result = {
            'accuracy': 100 * correct / total,
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


def gsm8k_dataset_postprocess(text: str) -> str:
    return text.split('#### ')[1].replace(',', '')


def gsm8k_postprocess(text: str) -> str:
    text = text.split('Question:')[0]
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    if not numbers:
        return 'NULL'
    return numbers[-1]