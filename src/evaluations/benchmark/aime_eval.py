import re
from collections import Counter
from tqdm import tqdm
import time
import requests
from src.evaluations.eval_util import *

class AIMEEvaluator:

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
    


def aime_postprocess(text: str, type="output") -> str:

    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        return cand_ans

    pattern = r"<Output>\s*(.*?)\s*</Output>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)  # 忽略大小写
    if match:
        extracted = match.group(1).strip()  # 提取并去除首尾空格
        # 检查是否包含 "answer is" 或 "final answer"
        if re.search(r"(final answer|answer is)", extracted.lower()):
            # 从提取内容中进一步匹配最后的数字
            num_pattern = r"(\d+)"  # 匹配数字
            num_match = re.search(num_pattern, extracted)
            if num_match:
                return num_match.group(1)  # 返回第一个匹配的数字
    return "0x3f3f3f3f"