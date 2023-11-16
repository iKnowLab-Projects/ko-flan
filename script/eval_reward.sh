
test() {
    model=$1
    revision=$2
    output=$3
    python -m eval.eval_dataset \
        --model_name_or_path "$model" \
        --batch_size 32 \
        --task "aihub_complaints_topic,ko_relation_fields,aihub_dialog_topic,csatqa-*" \
        --revision $revision \
        --output "eval-results/$output"
}

# model="iknow-lab/ko-flan-zero-v0-0731"
# test $model "main" "ko-flan-zero-v0-0731.csv"

model="iknow-lab/azou"
test $model "epoch-5" "azou-5epoch.csv"
test $model "epoch-4" "azou-4epoch.csv"
test $model "epoch-3" "azou-3epoch.csv"
test $model "epoch-2" "azou-2epoch.csv"
test $model "epoch-1" "azou-1epoch.csv"
