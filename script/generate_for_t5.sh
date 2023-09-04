# every task, max 20k
python -m task.run --splits train \
    --tasks "nsmc,apeach,korquad_v1.0,klue_mrc,klue_nli,klue_ynat,kornlu_nli,klue_re,kobest_copa,kobest_hellaswag,kobest_boolq,kobest_wic,niklex,nikl_absa,ko_nia_normal,mm_2021ner,mm_2022ner,mm_dialog,aihub_dialogSummarySummary,aihub_dialogSummaryTopic,aihub_columnDocumnetSummary,aihub_2020newsArticleSummary,aihub_bookSummary,tl_text_entailment,kor_nli" \
    --allow_no_negative \
    --use-ray \
    --max_instance_per_task 20000

# for test, only klue and kobest, all items
python -m task.run --splits test --tasks "nsmc,apeach,klue_*,kobest_*,haerae_csatqa" --max_instance_per_task -1

python -m task.push_to_hub iknow-lab/koflan-0904