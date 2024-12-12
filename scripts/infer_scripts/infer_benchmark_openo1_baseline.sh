#!/bin/bash
{
source /map-vepfs/miniconda3/bin/activate eval_benchmark
export PYTHONPATH=/map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark
cd /map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark

# 对应自己的机器进行修改
worker_num=$MLP_WORKER_NUM # 8
worker_id=$MLP_ROLE_INDEX



cuda_num=8
tensor_parallel=2
sample_num=1
max_tokens=32768
# 换成自己要infer的model

# save_name=openo1-qwen2.5-7B-sft-fix-template-ckpt1600
# model_name_or_path=/map-vepfs/huggingface/models/openo1-qwen2.5-7B-sft-fix-template/checkpoint-1600

# save_name=OpenO1-Qwen-7B-v0.1-checkpoint-1000
# model_name_or_path=/map-vepfs/huggingface/models/OpenO1/OpenO1-Qwen-7B-v0.1/checkpoint-1000

save_name=OpenO1-Qwen-7B-v0.2-fix-template-1600-ppo-v0.5_Bo64-difficulty-mixture-v1-5k_64gpu_Bo32-Ep1-len-8k
model_name_or_path=/map-vepfs/yizhi/OpenRLHF/checkpoint/OpenO1-Qwen-7B-v0.2-fix-template-1600-ppo-v0.5_Bo64-difficulty-mixture-v1-5k_64gpu_Bo32-Ep1-len-8k

# datasets=('aime') # 先跑aime，再跑下面的
# datasets=('gsm8k' 'math' 'bbh' 'mmlu' 'arc_c')
datasets=('math')

task=zs
base_dir=$(pwd)
# Loop through each dataset
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}

    save_file=zs
    save_dir="${base_dir}/infer_result_release/ppo_result/${save_name}/Bo${sample_num}/${dataset}/${save_file}"
    log_dir="${base_dir}/infer_result_release/ppo_result/${save_name}/Bo${sample_num}/${dataset}/${save_file}"
    # save_dir="${base_dir}/infer_result_release/sft_result/${save_name}/Bo${sample_num}/${dataset}/${save_file}"
    # log_dir="${base_dir}/infer_result_release/sft_result/${save_name}/Bo${sample_num}/${dataset}/${save_file}"

    mkdir -p "${save_dir}"
    mkdir -p "${log_dir}"

    echo "save to ${save_dir}"
    # exit()
    for id in $(seq 0 $((worker_num - 1))); do
        if [[ $id -eq $worker_id ]]; then
            
            for cd in $(seq 0 $((cuda_num/8 - 1))); do
                for ind in $(seq 0 $tensor_parallel $((cuda_num - 1))); do
                    gpu_indices=$(seq -s ',' $ind $((ind + tensor_parallel - 1)))
                    CUDA_VISIBLE_DEVICES=$gpu_indices python3 src/infer/infer_benchmark.py --dataset ${dataset} --sample_num ${sample_num} --worker_num ${worker_num} --worker_id ${worker_id} --task ${task} --model ${model_name_or_path} --tensor_parallel ${tensor_parallel} --cuda_start $((cd * tensor_parallel)) --cuda_ind ${ind} --cuda_num ${cuda_num} --max_tokens ${max_tokens} --save_dir ${save_dir} 2>&1 > "${log_dir}/${worker_id}-${ind}.log" &
                    echo "Start..."
                done
            done
            # start_gpu_id=4 # 设置起始gpu id，从4-7卡
            # for cd in $(seq -$start_gpu_id -$start_gpu_id); do
            #     for ind in $(seq 4 1 7); do 
            #         gpu_indices=$ind
            #         CUDA_VISIBLE_DEVICES=$gpu_indices python3 src/infer/infer_benchmark.py --dataset ${dataset} --sample_num ${sample_num} --worker_num ${worker_num} --worker_id ${worker_id} --task ${task} --model ${model_name_or_path} --tensor_parallel ${tensor_parallel} --cuda_start $((cd * tensor_parallel)) --cuda_ind ${ind} --cuda_num ${cuda_num} --max_tokens ${max_tokens} --save_dir ${save_dir} 2>&1 > "${log_dir}/${worker_id}-${ind}.log" &
            #         echo "Start on GPU ${ind}..."
            #     done
            # done

            # Wait for GPUs to be free
            memory_threshold=500 # Memory threshold in MiB
            sleep 10 # Initial sleep for 3 seconds

            while true; do
                all_gpus_free=true

                for gpu_id in $(seq ${start_gpu_id} $((${start_gpu_id} + cuda_num/tensor_parallel - 1))); do
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