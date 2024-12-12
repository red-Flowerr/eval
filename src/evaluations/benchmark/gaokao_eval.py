import re
from collections import Counter
from src.evaluations.eval_util import *
from src.evaluations.grader_new import math_equal
from sympy import *
from sympy.parsing.latex import parse_latex
class GAOKAOEvaluator:

    def __init__(self, version='v1'):
        assert version in ['v1', 'v2']
        self.version = version

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length'}
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            for i_ in i:
                if self.is_equiv(i_, j):
                    correct += 1
                    detail['correct'] = True
                    break
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result
    
    def score_self_consistency(self, predictions, references):

        if len(predictions) != len(references):
            return {'error': 'predictions and references have different lengths'}

        correct = 0
        count = 0
        details = []


        for pred_list, ref in zip(predictions, references):

            pred_counts = Counter(pred_list)


            most_common_pred, _ = pred_counts.most_common(1)[0]


            is_correct = self.is_equiv(most_common_pred, ref)
            correct += is_correct
            count += 1


            details.append({
                'pred': pred_list,
                'most_common_pred': most_common_pred,
                'answer': ref,
                'correct': is_correct
            })


        result = {
            'accuracy': 100 * correct / count,
            'details': details
        }

        return result

    def is_equiv(self, str1, str2, verbose=False):
        return math_equal(str1, str2)
        



def gaokao2023en_dataset_postprocess(text: str) -> str:
    return text.replace("$", "").strip()

def gaokao2023en_extract(text: str) -> str:
    # 定义正则表达式模式来匹配 \boxed{} 内容
    pattern = r'\\boxed\{(.*?)\}'
    
    # 使用 findall 方法找到所有匹配的内容
    matches = re.findall(pattern, text)
    
    processed_answers = []
    
    for match in matches:
        try:
            # 尝试解析 LaTeX 表达式为 SymPy 表达式
            expr = parse_latex(match)
            
            # 检查是否是可计算的表达式
            if isinstance(expr, (Add, Mul, Pow, Integer, Float)):
                # 如果是可计算的表达式，则尝试计算结果
                result = expr.evalf()
                processed_answers.append(f"Result: {result}")
            else:
                # 否则，简化表达式
                simplified_expr = simplify(expr)
                processed_answers.append(f"Simplified: ${latex(simplified_expr)}$")
        except Exception as e:
            # 如果解析失败，保留原始 LaTeX 表达式
            processed_answers.append(f"Original: ${match}$")
    
    return processed_answers
