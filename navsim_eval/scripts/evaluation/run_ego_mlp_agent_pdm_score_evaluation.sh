export NAVSIM_DEVKIT_ROOT="ABS_PATH_TO_NAVSIM_EVAL"
export OPENSCENE_DATA_ROOT="ABS_PATH_TO_OPENSCENE_DATA"
export NAVSIM_EXP_ROOT="EXP_ROOT_PATH"
export NUPLAN_MAPS_ROOT="ABS_PATH_TO_NUPLAN_MAPS"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_navtest
TRAIN_TEST_SPLIT=navtest
CHECKPOINT=$NAVSIM_EXP_ROOT/checkpoints/ego_status_mlp_seed_0.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=ego_status_mlp_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=ego_mlp_agent \
metric_cache_path=$CACHE_PATH \
