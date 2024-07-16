CUDA_VISIBLE_DEVICES=0

PRE_DIR="[your repository path]"

ret=contriever-msmarco

comp_name=cwyoon99/CompAct-7b

# reader
model_name=meta-llama/Meta-Llama-3-8B
cache_dir="[your caching paths]"

# Inference
task=HotpotQA
split=dev

iter=6
segment_size=5

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_prompt.py \
    --task $task \
    --data_path $PRE_DIR/data/data/retrieval/$ret"_"$task/$split.json \
    --fshot \
    --fshot_path $PRE_DIR/data/demos/fshot_$task.json \
    --compress_output_dir $PRE_DIR/data/experiments/compress/$ret"_"$task/$split \
    --read_output_dir $PRE_DIR/data/experiments/test/$ret"_"$task/$split \
    --compressor_name_or_path $comp_name \
    --model_name_or_path $model_name \
    --cache_dir $cache_dir \
    --batch_decoding \
    --batch_size 20 \
    --read_wo_prev_eval \
    --segment_size $segment_size \
    --max_iteration $iter \
    --results_file_name results \