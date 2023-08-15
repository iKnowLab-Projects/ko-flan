
accelerate launch -m train.reward_trainer \
    --do_train --do_eval \
    --project ko-flan \
    --run_name koflan-0813-data0731 \
    --model_name_or_path monologg/koelectra-base-v3-discriminator \
    --dataset iknow-lab/koflan-0731 \
    --logging_steps 100 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 100 \
    --save_epochs 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --output_dir ./checkpoint