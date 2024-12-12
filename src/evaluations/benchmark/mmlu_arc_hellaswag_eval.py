import re
from collections import Counter

class AccwithDetailsEvaluator:

    def score(self, predictions, references, origin_prompt) -> dict:

        if len(predictions) != len(references):
            return {'error': 'preds and refrs have different length.'}

        details = {}
        correct, total = 0, 0
        for index, (pred, ref) in enumerate(zip(predictions, references)):
            is_correct = False
            for p in pred:
                if p == ref:
                    is_correct = True
                    break
            # is_correct = pred == ref
            correct += is_correct
            details[str(index)] = {
                'prompt': origin_prompt[index],
                'pred': pred,
                'refr': ref,
                'is_correct': is_correct,
            }
            total += 1

        results = {'accuracy': correct / total * 100, 'details': details}

        return results

    def score_self_consistency(self, predictions, references, origin_prompt) -> dict:

        if len(predictions) != len(references):
            return {'error': 'predictions and references have different lengths.'}

        details = {}
        correct, total = 0, 0

        for index, (pred, ref) in enumerate(zip(predictions, references)):
            pred_counts = Counter(pred)
            
            most_common_pred, _ = pred_counts.most_common(1)[0]

            is_correct = most_common_pred == ref
            correct += is_correct

            details[str(index)] = {
                'prompt': origin_prompt[index],
                'pred': pred,
                'most_common_pred': most_common_pred,
                'refr': ref,
                'is_correct': is_correct,
            }
            total += 1

        results = {'accuracy': correct / total * 100, 'details': details}
        
        return results
    

def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    patterns = [
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案为\s*([{options}])',
        f'答案选\s*([{options}])',
        f'选择?\s*([{options}])',
        f'故选?\s*([{options}])'
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'(\s|^)[{options}](\s|$)',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'[{options}]',
    ]

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''