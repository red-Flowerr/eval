import re
from collections import Counter
from src.evaluations.eval_util import *
from src.evaluations.grader_new import math_equal

class OlympiadbenchEvaluator:

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
        


def olympiadbench_predict_postprocess(text: str) -> str:
    cand_ans = extract_boxed_answer(text, strip_double_curly_brace=True)
    if cand_ans:
        return cand_ans

    for maybe_ans in text.split('.'):
        # if 'final answer' in maybe_ans.lower():
        if re.search('final answer|answer is', maybe_ans.lower()):
            return normalize_final_answer(maybe_ans)
    return normalize_final_answer(text.split('.')[0])



def olympiadbench_dataset_postprocess(text: str) -> str:
    return text[0].strip("$")