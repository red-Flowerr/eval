## 配置环境
1. python=3.10 torch=2.41
2. 安装requirements.txt
3. 单独安装latex
```
cd openo1_eval/src/evaluations/latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers=4.42.3
```

## 使用case
```
bash /map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark/scripts/infer_scripts/infer_benchmark_llama_multidataset.sh "gsm8k,math,aime,amc23,college_math,gpqa_diamond,gaokao2023en,cmath,gaokao_math_qa,gaokao_math_cloze,data_from_train_math,gaokao2024,olympiadbench,omni_math" expv5_step40_greedy /map-vepfs/yizhi/OpenRLHF/checkpoint/open-o1-llama3.1-8B-sft-v0.2-1000-refinforce-v0.1_llama-Bo64-difficulty-ratio-v5-balanced-5k_64gpu_Bo32-Ep4-len-8k_global_step40 "greedy" 1
```
