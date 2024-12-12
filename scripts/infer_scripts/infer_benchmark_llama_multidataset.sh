#!/bin/bash
{
    source /map-vepfs/miniconda3/bin/activate eval_benchmark
    export PYTHONPATH=/map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark
    cd /map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark

    # bash /map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark/scripts/infer_scripts/infer_benchmark_llama_multidataset.sh "gsm8k,math,aime,amc23,cmath,college_math,gpqa_diamond,gaokao2024,olympiadbench,omni_math" llama-v0.2_reinforce-v0.1_mix-data-v4_step10 /map-vepfs/yizhi/OpenRLHF/checkpoint/open-o1-llama3.1-8B-sft-v0.2-1000-refinforce-v0.1_Bo64-difficulty-mixture-v4-hard-no-numina-10k_64gpu_Bo32-Ep1-len-16k_global_step10

    datasets=${1:-'math,aime'}
    save_name=$2
    model_name_or_path=$3
    greedy_or_topk=$4
    sample_num=$5
    worker_num=${6:-${MLP_WORKER_NUM:-1}} # machine number, eg $2 if exits. Default use the platform env, and then 4 machines with 8 gpu, do not exceed 40 (amc23)!
    worker_id=${7:-${MLP_ROLE_INDEX:-0}} # task_id, ranging from 0 to worker_num-1
  
    cuda_num=2
    tensor_parallel=2
    inference_batch=-1
    max_tokens=32768
    # 换成自己要infer的model

    # save_name=llama3.1-8B-sft-v0.2_refinforce-v0.1_data-mix-v3-5k_eval_debug
    # model_name_or_path=/map-vepfs/yizhi/OpenRLHF/checkpoint/open-o1-llama3.1-8B-sft-v0.2-1000-refinforce-v0.1_Bo64-difficulty-mixture-v3-all-medium-no-numina-5k_32gpu_Bo32-Ep1-len-8k

    # datasets=('aime' 'cmath' 'gaokao2024' 'gaokao_math_qa' 'gaokao_math_cloze' 'amc23' 'olympiadbench' 'gaokao2023en' 'college_math' 'omni_math' 'gpqa_diamond' 'math' 'gsm8k' 'data_from_train_math' )
    # datasets=('gaokao_math_cloze' 'gaokao2023en' 'gaokao2024' 'omni_math')


    # datasets='gaokao_math_cloze,gaokao2023en,gaokao2024,omni_math'

    task=zs
    base_dir=$(pwd)

    save_file=zs
    save_dir="${base_dir}/infer_result_release/reinforce_result/${save_name}/Bo${sample_num}/--dataset_placeholder--/${save_file}"
    log_dir="${base_dir}/infer_result_release/reinforce_result/${save_name}/Bo${sample_num}/${datasets}/${save_file}"

    echo "save to ${save_dir}"

    mkdir -p "${save_dir}"
    mkdir -p "${log_dir}"

    ind=0
    cd=0
    # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
    python3 src/infer/infer_benchmark_multi_dataset.py --multi_dataset --dataset ${datasets} --inference_batch ${inference_batch} --sample_num ${sample_num} --worker_num ${worker_num} --worker_id ${worker_id} --task ${task} --model ${model_name_or_path} --tensor_parallel ${tensor_parallel} --cuda_start $((cd * tensor_parallel)) --cuda_ind ${ind} --cuda_num ${cuda_num} --max_tokens ${max_tokens} --greedy_or_topk ${greedy_or_topk}  --save_dir ${save_dir} 2>&1 | tee "${log_dir}/${worker_id}-${ind}.log" 

    exit
}

# for step in 10 20 30 40 50 60; do
# bash /map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark/scripts/infer_scripts/infer_benchmark_llama_multidataset.sh "gsm8k,math,aime,amc23,cmath,college_math,gpqa_diamond,gaokao2024,olympiadbench,omni_math" llama-v0.2_reinforce-v0.1_mix-data-v4_step${step} /map-vepfs/yizhi/OpenRLHF/checkpoint/open-o1-llama3.1-8B-sft-v0.2-1000-refinforce-v0.1_Bo64-difficulty-mixture-v4-hard-no-numina-10k_64gpu_Bo32-Ep1-len-16k_global_step${step}
# done