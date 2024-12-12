#!/bin/bash
{
source /map-vepfs/miniconda3/bin/activate eval_benchmark
export PYTHONPATH=/map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark
cd /map-vepfs/openo1/OpenO1-Evaluation/eval_benchmark

# 对应自己的机器进行修改
worker_num=1 # 8
worker_id=0

cuda_num=8
tensor_parallel=4
sample_num=1
max_tokens=32768

# 换成自己要infer的model
save_name=openo1-llama-8B-sft-v0.2
model_name_or_path=/map-vepfs/huggingface/models/openo1-llama3.1-8B-sft-v0.2/checkpoint-1000

datasets=('gaokao2024' 'cmath')
task=zs
base_dir=$(pwd)

# 定义显存阈值和初始休眠时间
memory_threshold=5000 # Memory threshold in MiB (5 GB)
sleep 3 # Initial sleep for 3 seconds

for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}

    save_file=zs
    save_dir="${base_dir}/infer_result_release/sft_result/${save_name}/Bo${sample_num}/${dataset}/${save_file}"
    log_dir="${base_dir}/infer_result_release/sft_result/${save_name}/Bo${sample_num}/${dataset}/${save_file}"

    mkdir -p "${save_dir}"
    mkdir -p "${log_dir}"

    echo "save to ${save_dir}"

    for id in $(seq 0 $((worker_num - 1))); do
        if [[ $id -eq $worker_id ]]; then
            
            # 遍历所有GPU，寻找空闲的GPU
            while true; do
                any_gpu_free=false
                free_gpu_ids=()  # 用于存储空闲的GPU编号

                for gpu_id in $(seq 0 $((${cuda_num} - 1))); do
                    memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)

                    if [ "$memory_used" -le "$memory_threshold" ]; then
                        echo "GPU $gpu_id is free."
                        any_gpu_free=true
                        free_gpu_ids+=($gpu_id)  # 记录空闲的GPU编号
                    else
                        echo "GPU $gpu_id is still in use. Memory used: $memory_used MiB"
                    fi
                done

                # 如果有空闲的GPU，启动任务
                if $any_gpu_free; then
                    for gpu_id in "${free_gpu_ids[@]}"; do
                        # 启动任务
                        CUDA_VISIBLE_DEVICES=$gpu_id python3 src/infer/infer_benchmark.py \
                            --dataset ${dataset} \
                            --sample_num ${sample_num} \
                            --worker_num ${worker_num} \
                            --worker_id ${worker_id} \
                            --task ${task} \
                            --model ${model_name_or_path} \
                            --tensor_parallel ${tensor_parallel} \
                            --cuda_start $gpu_id \
                            --cuda_ind $gpu_id \
                            --cuda_num ${cuda_num} \
                            --max_tokens ${max_tokens} \
                            --save_dir ${save_dir} \
                            2>&1 > "${log_dir}/${worker_id}-${gpu_id}.log" &
                        
                        echo "Start on GPU ${gpu_id}..."
                        
                        # 为了避免频繁启动任务，可以在这里添加一个小的延迟
                        sleep 1
                    done
                    
                    # 任务启动后，继续监控其他GPU
                    continue
                else
                    echo "Waiting for any GPU to be free..."
                    sleep 3 # Wait for 3 seconds before checking again
                fi
            done
        fi
    done
done

}