
python -m train.reward_trainer \
    --do_eval \
    --project ko-flan \
    --run_name koflan-0731 \
    --model_name_or_path checkpoint/test/epoch-9 \
    --dataset iknow-lab/koflan-0731 \
    --logging_steps 500 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --output_dir ./checkpoint