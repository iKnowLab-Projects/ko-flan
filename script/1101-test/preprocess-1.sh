# 제로샷 분류 모델 학습 > 데이터 전처리 > 선택적 데이터 전처리 및 저장

# unsmile,klue_*,kobest_*,korquad_v1.0,kornlu_nli,kor_nlu,tl_text_entailment,nikl_cb*,ko_nia_normal,aihub_dialogSummaryTopic,niklex,nikl_absa
python -m task.run --splits train \
    --tasks "nsmc,apeach" \
    --max_instance_per_task 5000

# for test, only klue and kobest, all items
python -m task.run --splits test --tasks "apeach,klue_*,kobest_*,haerae_csatqa" --max_instance_per_task -1

