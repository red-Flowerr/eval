import json
import os
import re
import os.path as osp


# template
########################################################################################################
boxed_zero_shot_prompt = '''
{problem}

Please provide the answer like this "the answer is \\boxed{{X}}" where X is your final answer.
'''.strip()

normal_zero_shot_prompt = '''
{problem}

Please provide the answer like this "the answer is X" where X is your final answer.
'''.strip()

choice_zero_shot_prompt = '''
{problem}

Please provide the answer like this "the answer is (X)" where X is the correct letter choice.
'''.strip()

########################################################################################################


mmlu_all_sets = [
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_physics',
    'electrical_engineering',
    'astronomy',
    'anatomy',
    'abstract_algebra',
    'machine_learning',
    'clinical_knowledge',
    'global_facts',
    'management',
    'nutrition',
    'marketing',
    'professional_accounting',
    'high_school_geography',
    'international_law',
    'moral_scenarios',
    'computer_security',
    'high_school_microeconomics',
    'professional_law',
    'medical_genetics',
    'professional_psychology',
    'jurisprudence',
    'world_religions',
    'philosophy',
    'virology',
    'high_school_chemistry',
    'public_relations',
    'high_school_macroeconomics',
    'human_sexuality',
    'elementary_mathematics',
    'high_school_physics',
    'high_school_computer_science',
    'high_school_european_history',
    'business_ethics',
    'moral_disputes',
    'high_school_statistics',
    'miscellaneous',
    'formal_logic',
    'high_school_government_and_politics',
    'prehistory',
    'security_studies',
    'high_school_biology',
    'logical_fallacies',
    'high_school_world_history',
    'professional_medicine',
    'high_school_mathematics',
    'college_medicine',
    'high_school_us_history',
    'sociology',
    'econometrics',
    'high_school_psychology',
    'human_aging',
    'us_foreign_policy',
    'conceptual_physics',
]

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

key_map = {
    "math": "problem",
    "mmlu": "prompt",
    "gsm8k": "question",
    "bbh": "prompt",
    "arc_c": "prompt",
    "hellaswag": "prompt",
    "aime": "problem",
    "gaokao2024": "question",
    "gaokao2023en": "question",
    "gaokao_math_cloze": "question",
    "gaokao_math_qa": "question",
    "amc23": "problem",
    "olympiadbench": "question",
    "college_math": "question",
    "cmath": "question",
    "gpqa_diamond": "Question",
    "omni_math": "problem",
    "data_from_train_math": "query",
}

"""Read data"""
def read_data(file_path):
    """read jsonl"""
    data = []
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            data.append(line)
    return data


def read_data_2(file_path):
    data = []
    with open(file_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data
 

def read_data_math_2(path):
    data = json.load(open(path))
    raw_data = []
    for i, item in enumerate(data):
        try:
            promblem = item['problem'],
        except Exception as e:
            print(i)
        raw_data.append({
            'problem':
             promblem,
            'solution':
            extract_boxed_answer(item['solution'])
        })
    return raw_data


def read_csv_mmlu(path):
    raw_data = []
    option_map = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D' 
    }
    with open(path, encoding='utf-8') as f:
        for item in f:
            item = json.loads(item)
            sub_type = item['subject']
            _hint = f'There is a single choice question about {sub_type.replace("_", " ")}:'
            raw_data.append({
                'input': item['question'],
                'A': item['choices'][0],
                'B': item['choices'][1],
                'C': item['choices'][2],
                'D': item['choices'][3],
                'target': option_map[item['answer']],
                'hint': _hint,
                'type': sub_type
            })
    return raw_data


def read_arc(path):
    with open(path, 'r', errors='ignore') as in_f:
        rows = []
        for line in in_f:
            item = json.loads(line.strip())
            question = item['question']
            choices = item['choices']
            # if len(question['choices']) != 4:
            #     continue
            labels = [c for c in choices['label']]
            answerKey = 'ABCDE'[labels.index(item['answerKey'])]
            if len(labels) == 3:
                rows.append({
                    'question': question,
                    'answerKey': answerKey,
                    'textA': choices['text'][0],
                    'textB': choices['text'][1],
                    'textC': choices['text'][2],
                })
            elif len(labels) == 4:
                rows.append({
                    'question': question,
                    'answerKey': answerKey,
                    'textA': choices['text'][0],
                    'textB': choices['text'][1],
                    'textC': choices['text'][2],
                    'textD': choices['text'][3],
                })
            elif len(labels) == 5:
                rows.append({
                    'question': question,
                    'answerKey': answerKey,
                    'textA': choices['text'][0],
                    'textB': choices['text'][1],
                    'textC': choices['text'][2],
                    'textD': choices['text'][3],
                    'textE': choices['text'][4],
                })
    return rows


def read_data_2_hellaswag(path):
    dataset_list = []
    answer_key = {
        0: "A",
        1: "B",
        2: "C",
        3: "D"
    }
    with open(path, 'r') as f:
        for line in f:
            line = json.loads(line)
            item = {
                'A': line['endings'][0],
                'B': line['endings'][1],
                'C': line['endings'][2],
                'D': line['endings'][3],
                'question': line['ctx'],
                'answer': answer_key[line['label']],
            }
            dataset_list.append(item)
    return dataset_list
############################################################################


"""Get related benchmark data"""
def get_question(args):
    base_dir = os.getcwd()
    datas = read_data(f"{base_dir}/benchmark_data/{args.dataset}.jsonl")
    inputs = []
    for idx, data in enumerate(datas):
        if idx % args.worker_num != args.worker_id:
            continue
        inputs.append(data)
    return inputs

def get_question_math(args):
    base_dir = os.getcwd()
    datas = read_data_math_2(f"{base_dir}/benchmark_data/{args.dataset}.json")
    inputs = []
    for idx, data in enumerate(datas):
        if idx % args.worker_num != args.worker_id:
            continue
        inputs.append(data)
    return inputs

def get_question_mmlu(args):
    inputs = []
    base_dir = os.getcwd()
    datas = read_csv_mmlu(f"{base_dir}/benchmark_data/{args.dataset}.jsonl")
    for idx, data in enumerate(datas):
        if idx % args.worker_num != args.worker_id:
            continue
        prompt= '{}\n{}\nA. {}\nB. {}\nC. {}\nD. {}'.format(data['hint'], data['input'], data['A'], data['B'], data['C'], data['D'])
        data['prompt'] = prompt
        inputs.append({
            'prompt': prompt,
            'A':data['A'], 
            'B': data['B'],
            'C': data['C'],
            'D': data['D'],
            'target': data['target']
        })
    return inputs

def get_question_arc_c(args):
    inputs = []
    base_dir = os.getcwd()
    datas = read_arc(f"{base_dir}/benchmark_data/{args.dataset}.jsonl")
    for idx, data in enumerate(datas):
        if idx % args.worker_num != args.worker_id:
            continue

        if len(data) == 6:
            prompt= '{}\nA. {}\nB. {}\nC. {}\nD. {}'.format(data['question'], data['textA'], data['textB'], data['textC'], data['textD'])
            data['prompt'] = prompt
            inputs.append({
                'prompt': prompt,
                'textA': data['textA'],
                'textB': data['textB'],
                'textC': data['textC'],
                'textD': data['textD'],
                'answerKey': data['answerKey']
            })
        elif len(data) == 5:
            prompt= '{}\nA. {}\nB. {}\nC. {}'.format(data['question'], data['textA'], data['textB'], data['textC'])
            data['prompt'] = prompt
            inputs.append({
                    'prompt': prompt,
                    'textA': data['textA'],
                    'textB': data['textB'],
                    'textC': data['textC'],
                    'answerKey': data['answerKey']
            })
        elif len(data) == 7:
            prompt= '{}\nA. {}\nB. {}\nC. {}\nD. {}\nE. {}'.format(data['question'], data['textA'], data['textB'], data['textC'], data['textD'], data['textE'])
            data['prompt'] = prompt
            inputs.append({
                'prompt': prompt,
                'textA': data['textA'],
                'textB': data['textB'],
                'textC': data['textC'],
                'textD': data['textD'],
                'textE': data['textE'],
                'answerKey': data['answerKey']
            })
    return inputs

def get_question_bbh(args):
    inputs = []
    base_dir = os.getcwd()
    for name in bbh_multiple_choice_sets:
        with open(osp.join(f"{base_dir}/benchmark_data/bbh", f'{name}.json'), 'r') as f:
            datas = json.load(f)['examples']
            for idx, data in enumerate(datas):
                if idx % args.worker_num != args.worker_id:
                    continue
                # prompt= "Answer the question.\n\nQ: {}".format(data["input"])
                data['prompt'] = data["input"]
                data['type'] = name
                inputs.append(data)
    for name in bbh_free_form_sets:
        with open(osp.join(f"{base_dir}/benchmark_data/bbh", f'{name}.json'), 'r') as f:
            datas = json.load(f)['examples']
            for idx, data in enumerate(datas):
                if idx % args.worker_num != args.worker_id:
                    continue
                # prompt= "Answer the question.\n\nQ: {}".format(data["input"])
                data['prompt'] = data["input"]
                data['type'] = name
                inputs.append(data)
    return inputs

def get_question_hellaswag(args=None):
    inputs = []
    base_dir = os.getcwd()
    datas = read_data_2_hellaswag(f"{base_dir}/benchmark_data/{args.dataset}.jsonl")
    for idx, data in enumerate(datas):
        if idx % args.worker_num != args.worker_id:
            continue
        prompt = '{}\nA. {}\nB. {}\nC. {}\nD. {}'.format(data['question'], data['A'], data['B'], data['C'], data['D'])
        data['prompt'] = prompt
        inputs.append(data)
    return inputs

def get_question_gaokao_math_qa(args=None):
    inputs = []
    base_dir = os.getcwd()
    datas = read_data(f"{base_dir}/benchmark_data/{args.dataset}.jsonl")
    for idx, data in enumerate(datas):
        if idx % args.worker_num != args.worker_id:
            continue
        prompt = '{}\nA. {}\nB. {}\nC. {}\nD. {}'.format(data['question'], data["options"]["A"], data["options"]["B"], data["options"]["C"], data["options"]["D"])
        data['question'] = prompt
        inputs.append(data)
    return inputs


def save_jsonl(data, filename):
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
############################################################################


"""Process String"""
def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None
    

def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer


def str2bool(s):
    s = s.lower()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError('invalid value: {}, must be true or false'.format(s))
    

# To accelerate the process of adding chat template
