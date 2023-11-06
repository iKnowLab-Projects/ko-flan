
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


test "heegyu/rm-koflan-1031-roberta-base-5e-5" "epoch-1" "1031-base-1epoch.csv"
# test "heegyu/rm-koflan-1031-roberta-base-5e-5" "epoch-2" "1031-base-2epoch.csv"
# test "heegyu/rm-koflan-1031-roberta-base-5e-5" "epoch-3" "1031-base-3epoch.csv"
# test "heegyu/rm-koflan-1031-roberta-base-5e-5" "epoch-4" "1031-base-4epoch.csv"
# test "heegyu/rm-koflan-1031-roberta-base-5e-5" "epoch-5" "1031-base-5epoch.csv"
