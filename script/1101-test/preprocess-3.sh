# 제로샷 분류 모델 학습 > 데이터 전처리 
# ray를 이용한 분산 처리 --use-ray/--no-ray
# task 당 개수 제한(-1 이면 제한 없음)
# --max_instance_per_task 
# 특수 포멧 허용
# --require_negative/--allow_no_negative
# --require_input/--allow_no_input

python -m task.run --splits train \
    --tasks "korquad_v1.0,klue_mrc,klue_nli,klue_ynat,kornlu_nli,klue_re,kobest_copa,kobest_hellaswag,kobest_boolq,kobest_wic,niklex,nikl_absa,ko_nia_normal,mm_2021ner,mm_2022ner,mm_dialog,aihub_dialogSummarySummary,aihub_dialogSummaryTopic,aihub_columnDocumnetSummary,aihub_2020newsArticleSummary,aihub_bookSummary,tl_text_entailment,kor_nli" \
    --allow_no_negative \
    --allow_no_input \
    --use-ray \
    --max_instance_per_task 5000
