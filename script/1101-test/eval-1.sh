
# 평가
# 한국어 일반상식(kobest)
# 혐오표현(apeach)
# 주제 분류(klue, nsmc)

python -m eval.eval_dataset \
    --model_name_or_path "iknow-lab/ko-flan-zero-v0-0731" \
    --batch_size 32 \
    --output "eval-result.csv"