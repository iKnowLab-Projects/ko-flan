wandb online

# 하이퍼파라미터 수정
# 모델 평가(evaluation_strategy)
# 모델 저장

accelerate launch -m train.reward_trainer \
    --do_train --do_eval \
    --project ko-flan \
    --run_name koflan-1101-data0725 \
    --model_name_or_path monologg/koelectra-base-v3-discriminator \
    --dataset iknow-lab/koflan-test-12t-93k-0725 \
    --logging_steps 10 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --save_epochs 1 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --output_dir ./checkpoint