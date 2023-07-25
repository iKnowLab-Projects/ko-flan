
python -m train.reward_trainer \
    --do_train --do_eval \
    --project ko-flan \
    --run_name test \
    --model_name_or_path monologg/koelectra-small-v3-discriminator \
    --dataset iknow-lab/koflan-test-12t-93k-0725 \
    --logging_steps 500 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --output_dir ./checkpoint