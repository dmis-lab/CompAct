CUDA_VISIBLE_DEVICES=0,1,2,3
n_processes=4

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes $n_processes scripts/run_sft.py recipes/mistral-7b-instruct-v0.2/sft/config_full.yaml
