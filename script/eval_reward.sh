
test() {
    model=$1
    revision=$2
    output=$3
    python -m eval.eval_dataset \
        --model_name_or_path "$model" \
        --batch_size 32 \
        --revision $revision \
        --output "eval-results/$output"
}

model="heegyu/rm-1031-roberta-large-5e-5"
test $model "epoch-1" "1031-large-1epoch.csv"
test $model "epoch-2" "1031-large-2epoch.csv"
test $model "epoch-3" "1031-large-3epoch.csv"
test $model "epoch-4" "1031-large-4epoch.csv"
test $model "epoch-5" "1031-large-5epoch.csv"
