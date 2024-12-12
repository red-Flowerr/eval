#!/bin/bash
{
source /map-vepfs/miniconda3/bin/activate eval_benchmark
export PYTHONPATH=/map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark
cd /map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark

# 对应自己的机器进行修改
worker_num=$MLP_WORKER_NUM # 8
worker_id=$MLP_ROLE_INDEX

# worker_num=1
# worker_id=0


 
cuda_num=8 # 4卡就写4，8卡写8，在下面的 for cd in $(seq 0 $((cuda_num/4 - 1))); 里也要做相应的修改
tensor_parallel=1
sample_num=1
max_tokens=32768
# 换成自己要infer的model

# name要带有qwen/llama字样即可，如果是70b或72b要写在name里
# save_name=openo1-qwen2.5-7B-sft-fix-template
# pt_path=/map-vepfs/huggingface/models/openo1-qwen2.5-7B-sft-fix-template/checkpoint-1600

save_name=llama-v0.2_reinforce-v0.1_mix-data-v4_step60_vol
model_name_or_path=/map-vepfs/yizhi/OpenRLHF/checkpoint/open-o1-llama3.1-8B-sft-v0.2-1000-refinforce-v0.1_Bo64-difficulty-mixture-v4-hard-no-numina-10k_64gpu_Bo32-Ep1-len-16k_global_step60

# datasets=('mmlu')
# datasets=('aime')
datasets=('aime' 'gsm8k' 'math' 'amc23' 'olympiadbench' 'college_math' 'omni_math' 'gpqa_diamond' 'gaokao2024' 'gaokao2023en' 'gaokao_math_cloze' 'gaokao_math_qa') # Add more datasets here

task=zs
base_dir=$(pwd)
# Loop through each dataset
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}

    save_file=zs
    save_dir="${base_dir}/infer_result_release/reinforce_result/${save_name}/Bo${sample_num}/${dataset}/${save_file}"
    log_dir="${base_dir}/infer_result_release/reinforce_result/${save_name}/Bo${sample_num}/${dataset}/${save_file}"
    
    mkdir -p "${save_dir}"
    mkdir -p "${log_dir}"

    for id in $(seq 0 $((worker_num - 1))); do
        if [[ $id -eq $worker_id ]]; then
            echo "$id"
            for cd in $(seq 0 $((cuda_num/8 - 1))); do
                echo "cd $cd"
                for ind in $(seq 0 $tensor_parallel $((cuda_num - 1))); do
                    gpu_indices=$(seq -s ',' $ind $((ind + tensor_parallel - 1)))
                    echo "ind ${ind} gpu_indices ${gpu_indices}"
                    CUDA_VISIBLE_DEVICES=$gpu_indices python3 src/infer/infer_benchmark.py --dataset ${dataset} --sample_num ${sample_num} --worker_num ${worker_num} --worker_id ${worker_id} --task ${task} --model ${model_name_or_path} --tensor_parallel ${tensor_parallel} --cuda_start $((cd * tensor_parallel)) --cuda_ind ${ind} --cuda_num ${cuda_num} --max_tokens ${max_tokens} --save_dir ${save_dir} 2>&1 > "${log_dir}/${worker_id}-${ind}.log" &
                    echo "Start..."
                done
            done
            
            # Wait for GPUs to be free
            memory_threshold=200 # Memory threshold in MiB
            sleep 10 # Initial sleep for 3 seconds

            while true; do
                all_gpus_free=true

                for gpu_id in $(seq 0 $((cuda_num/tensor_parallel - 1))); do
                    memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)

                    if [ "$memory_used" -le "$memory_threshold" ]; then
                        echo "GPU $gpu_id is free."
                    else
                        echo "GPU $gpu_id is still in use. Memory used: $memory_used MiB"
                        all_gpus_free=false
                        break
                    fi
                done

                if $all_gpus_free; then
                    echo "All GPUs are free. Proceeding to next process..."
                    break
                else
                    echo "Waiting for all GPUs to be free..."
                    sleep 10 # Wait for 10 seconds before checking again
                fi
            done

            sleep 10 # Additional sleep after all GPUs are free
        fi
    done
done

}