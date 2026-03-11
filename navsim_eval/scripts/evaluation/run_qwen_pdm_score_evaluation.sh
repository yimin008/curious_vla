export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="ABS_PATH_TO_NAVSIM_EVAL" # /root_dir/navsim_eval
export OPENSCENE_DATA_ROOT="ABS_PATH_TO_OPENSCENE_DATA" # see /root_dir/navsim_eval/docs/install.md
export NAVSIM_EXP_ROOT="EXP_ROOT_PATH" # any path you want to store the evaluation results and metric cache
export NUPLAN_MAPS_ROOT="ABS_PATH_TO_NUPLAN_MAPS" # see /root_dir/navsim_eval/docs/install.md
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navtest
export STATS_PATH="ABS_PATH_TO_TRAJECTORY_STATS" # /root_dir/stats/trajectory_stats_train.json


TRAIN_TEST_SPLIT=navtest


: "${AGENT_NAME:=navsim_qwen_norm_cot_baseline_agent}"
: "${EXPERIMENT_NAME:=demo_test}"
: "${MODEL_NAME_OR_PATH:=YOUR_MODEL_PATH}" # if use lf api serve, can be none; if use vllm serve, should fill the actual path.

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
train_test_split=$TRAIN_TEST_SPLIT \
experiment_name=$EXPERIMENT_NAME \
agent=$AGENT_NAME \
agent.config.model_name_or_path=$MODEL_NAME_OR_PATH \
metric_cache_path=$CACHE_PATH \
worker=single_machine_thread_pool \
worker.use_process_pool=True
