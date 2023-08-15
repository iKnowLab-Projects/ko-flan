
python -m eval.eval_dataset \
    --model_name_or_path checkpoint/koelectra-base-0813-data0731/epoch-10 \
    --batch_size 32 
    
# python -m train.reward_trainer \
#     --do_eval \
#     --project ko-flan \
#     --run_name koflan-0731 \
#     --model_name_or_path checkpoint/koelectra-base-0813-data0731/epoch-10 \
#     --dataset iknow-lab/koflan-0731 \
#     --logging_steps 500 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 32 \
#     --num_train_epochs 10 \
#     --save_strategy epoch \
#     --evaluation_strategy epoch \
#     --output_dir ./checkpoint