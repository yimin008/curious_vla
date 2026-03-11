#!/bin/bash
set -euo pipefail
source /mnt/data/miniconda3/etc/profile.d/conda.sh
conda activate lf

sleep 5
: "${model_name_or_path:="YOUR_MODEL_PATH"}"
: "${template:=qwen3_vl}"

: "${max_num_seqs:=256}"
: "${max_num_batched_tokens:=98304}"

: "${min_pixels:=3136}"  # ATTENTION!!! Don't delete!!!
: "${max_pixels:=262144}"
mm_processor_kwargs="{\"min_pixels\": ${min_pixels}, \"max_pixels\": ${max_pixels}}"

num_instances=$1
start_port=8192

# 用法: ./vllm_serve.sh <num_instances>
# 例如: ./vllm_serve.sh 4   # 启动4个实例
#      ./vllm_serve.sh 8   # 启动8个实例
#      ./vllm_serve.sh 16  # 启动16个实例

if [ -z "$num_instances" ]; then
  echo "please provide num_instances: $0 4"
  exit 1
fi

cards_per_instance=$((8 / num_instances))

for i in $(seq 0 $((num_instances-1))); do
  start_card=$((i * cards_per_instance))
  end_card=$((start_card + cards_per_instance - 1))
  port=$((start_port + i))

  devices=$(seq -s, $start_card $end_card)

  echo "Start Instance $i: Use Devices: $devices, Port: $port"

  CUDA_VISIBLE_DEVICES=$devices \
  vllm serve $model_name_or_path \
    --limit-mm-per-prompt.video 0 \
    --mm-processor-kwargs "${mm_processor_kwargs}" \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port $port \
    --tensor-parallel-size $cards_per_instance \
    --enable-prefix-caching \
    --max-num-seqs $max_num_seqs \
    --max-num-batched-tokens $max_num_batched_tokens \
    --disable-uvicorn-access-log \
    --disable-frontend-multiprocessing \
    --async-scheduling \
    &
done
