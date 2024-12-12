import os
import json

"""
根据输入的dataset来加载对应的prompt和response
"""

# 参数准备区
KEY_MAP = {
    "gsm8k": "question",
    "bbh": "prompt",
    "math": "problem",
    "gaokao2024_I": "question",
    "gaokao2024_II": "question",
    "gaokao2023en": "question",
    "amc23": "problem",
    "olympiadbench": "question",
    "college_math": "question",
    "minerva_math": "problem",
    "cmath": "question",
    "gaokao_math_cloze": "question",
    "gaokao_math_qa": "question",
}
SAMPLE_NUM = 64

BBH_MULTIPLE_CHOICE_SETS = [
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

BBH_FREE_FROM_SETS = [
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

# 加载文件
def read_data(file_path):
    data = []
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line = json.loads(line)
            data.append(line)
    return data

def merge_data(file_path):
    datas = []
    files = os.listdir(file_path) # 把文件名都列出来
    for file in files:
        if file.endswith('.json') and "zs" not in file:
            data = read_data(os.path.join(file_path,file))
            datas.extend(data)
    return datas

def get_data_gsm8k(datas, dataset):
    """
    提取出prompt和response
    """
    prompts = []
    responses = {}

    for data in datas:
        prompts.append(data[KEY_MAP[dataset]])
        responses[data[KEY_MAP[dataset]]] = []

        for i in range(SAMPLE_NUM):
            response = data["response_"+str(i)]
            responses[data[KEY_MAP[dataset]]].append(response)
    return responses

def get_data_math(datas, dataset):
    """
    提取出prompt和response
    """
    prompts = []
    responses = {}

    for data in datas:
        prompts.append(data[KEY_MAP[dataset]])
        responses[data[KEY_MAP[dataset]][0]] = []
        for i in range(SAMPLE_NUM):
            response = data["response_"+str(i)]

            responses[data[KEY_MAP[dataset]][0]].append(response)
    return responses

def get_data_bbh(datas, dataset):
    
    data_dict = {}
    # 先聚类
    for data in datas:
        if data['type'] not in data_dict.keys():
            data_dict[data['type']] = [data]
        else:
            data_dict[data['type']].append(data)
    
    prompts = []
    responses = {}
    for name, data in data_dict.items():
        if name in BBH_MULTIPLE_CHOICE_SETS:

            for item in data:
                
                prompts.append(item['prompt'])
                responses[item[KEY_MAP[dataset]]] = []

                for i in range(SAMPLE_NUM):
                    response = item["response_"+str(i)]
                    responses[item[KEY_MAP[dataset]]].append(response)

        if name in BBH_FREE_FROM_SETS:
            for item in data:
                prompts.append(item['prompt'])
                responses[item[KEY_MAP[dataset]]] = []

                for i in range(SAMPLE_NUM):
                    response = item["response_"+str(i)]
                    responses[item[KEY_MAP[dataset]]].append(response)
    return responses


def data_process(datasets):
    models = [('OpenO1-Qwen-7B-v0.1-blend', 'zs')]
    base_path = os.getcwd()

    for (model ,task) in models:
        for dataset in datasets:
            print(f'------{dataset}-{model}-{task}--------')
            # path = "/map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark/infer_result/blend_result/Bo64/math/OpenO1-Qwen-7B-v0.1-blend/zs"
            path = os.path.join(base_path, 'infer_result', 'blend_result', 'Bo64', dataset, model, task)
            datas = merge_data(path)
            if dataset == 'math':
                responses = get_data_math(datas, dataset) # responses的key是prompt
            elif dataset == 'gsm8k':
                responses = get_data_gsm8k(datas, dataset)
            elif dataset == 'bbh':
                responses = get_data_bbh(datas, dataset)
    return responses 


