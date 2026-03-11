#!/bin/bash
source /mnt/data/miniconda3/etc/profile.d/conda.sh
conda activate lf

# pkill -9 VLLM
# pkill -9 llamafactory-cl
# pkilk -9 llamafact
# pkill -9 python3.10
# pkill -9 python
clear
sleep 5

: "${model_name_or_path=YOUR_MODEL_PATH}"
: "${template:=qwen2_vl}"

# 用法: ./lf_serve_cot.sh <num_instances>
# 例如: ./lf_serve_cot.sh 4   # 启动4个实例
#      ./lf_serve_cot.sh 8   # 启动8个实例
#      ./lf_serve_cot.sh 16  # 启动16个实例

num_instances=$1
start_port=8192

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

  CUDA_VISIBLE_DEVICES=$devices API_VERBOSE=0 \
  API_PORT=$port \
  llamafactory-cli api \
    --model_name_or_path $model_name_or_path \
    --template $template \
    --infer_backend vllm \
    --image_max_pixels 262144 \
    --vllm_maxlen 65536 \
    --trust_remote_code true &
done

# --vllm_maxlen 16384
# 控制vllm kvcache最大token数（输入+输出），目前分辨率下，1920*1080 smart resize后大约1248token
# 需要控制好prompt长度，否则可能要开TP解决OOM问题。如果有80G显存可忽略。
