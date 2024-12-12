import json

# 假设您要读取的 JSONL 文件路径
jsonl_file_path = '/map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark/infer_result_release/sft_result/openo1-qwen2.5-7B-sft-fix-template-ckpt1600/Bo1/gaokao2023en/zs_eval_details_1_2.jsonl'
# 新的 JSONL 文件路径
output_jsonl_file_path = '/map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark/infer_result_release/sft_result/openo1-qwen2.5-7B-sft-fix-template-ckpt1600/Bo1/gaokao2023en/zs_eval_details_1_review2.jsonl'

# 存储 correct 为 false 的数据
false_correct_entries = []

# 读取 JSONL 文件
with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line.strip())  # 解析每一行
        if not entry['correct']:  # 检查 correct 是否为 false
            false_correct_entries.append(entry)  # 添加到列表中

# 将结果保存到新的 JSONL 文件
with open(output_jsonl_file_path, 'w') as output_file:
    for entry in false_correct_entries:
        output_file.write(json.dumps(entry) + '\n')  # 按条写入 JSONL 格式