
accelerate launch -m train.s2s_trainer \
    --do_train --do_eval \
    --project ko-flan \
    --run_name ke-t5-large-0904 \
    --model_name_or_path KETI-AIR/ke-t5-large \
    --model_type seq2seq \
    --dataset iknow-lab/koflan-0904 \
    --logging_steps 100 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 20 \
    --save_epochs 5 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --output_dir ./checkpoint