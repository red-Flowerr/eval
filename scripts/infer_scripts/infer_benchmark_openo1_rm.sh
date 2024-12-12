#!/bin/bash

# 对应自己的环境进行修改
source /map-vepfs/miniconda3/bin/activate openrlhf
export PYTHONPATH=/map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark
cd /map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark

# 对应自己的机器进行修改
worker_num=$MLP_WORKER_NUM # 8
worker_id=$MLP_ROLE_INDEX

cuda_num=8
tensor_parallel=4
save_name=Qwen2.5-math-rm-72B
# 需修改成自己的model路径
model_name=/map-vepfs/huggingface/models/Qwen/Qwen2.5-Math-RM-72B

datasets=('math')
base_dir=$(pwd)
for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    
    save_file=rm_bon
    save_dir="${base_dir}/infer_result/BoN_rm_result/${dataset}/${save_name}/${save_file}"
    log_dir="${base_dir}/infer_result/BoN_rm_result/${dataset}/${save_name}/${save_file}"

    mkdir -p "${save_dir}"
    mkdir -p "${log_dir}"

    for id in $(seq 0 $((worker_num - 1))); do
        if [[ $id -eq $worker_id ]]; then

            for cd in $(seq 0 $((cuda_num/8 - 1))); do
                for ind in $(seq 0 $tensor_parallel $((cuda_num - 1))); do
                    gpu_indices=$(seq -s ',' $ind $((ind + tensor_parallel - 1)))
                    echo "Starting process on GPU $gpu_indices for dataset ${dataset}"
                    CUDA_VISIBLE_DEVICES=$gpu_indices python3 src/infer/infer_rm_benchmark.py --dataset ${datasets} --worker_num ${worker_num} --worker_id ${worker_id} --model_name ${model_name} --tensor_parallel ${tensor_parallel} --cuda_start $((cd * tensor_parallel)) --cuda_ind ${ind} --cuda_num ${cuda_num} --save_dir ${save_dir} 2>&1 > "${log_dir}/${worker_id}-${ind}.log" &
                    echo "Start..."
                done
            done
            
            # Wait for GPUs to be free
            memory_threshold=500 # Memory threshold in MiB
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
