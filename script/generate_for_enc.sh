# every task, max 20k

# 1101
# "nsmc,apeach,unsmile,klue_*,kobest_*,korquad_v1.0,kornlu_nli,kor_nlu,tl_text_entailment,niklex,nikl_absa,ko_nia_normal,aihub_dialogSummaryTopic,mm_2022dialogTopic,mm_2022newsTopic,mm_2022chatTopic"
# 1101 + kowow + aihub compaints
# "nsmc,apeach,unsmile,klue_*,kobest_*,korquad_v1.0,kornlu_nli,kor_nlu,tl_text_entailment,niklex,nikl_absa,ko_nia_normal,aihub_dialogSummaryTopic,mm_2022dialogTopic,mm_2022newsTopic,mm_2022chatTopic,kowow_dialog_topic,kowow_sentence_topic,aihub_complaints_*"
# 1031 + aihub complaints
# "nsmc,apeach,unsmile,klue_*,kobest_*,korquad_v1.0,kornlu_nli,kor_nlu,tl_text_entailment,niklex,nikl_absa,ko_nia_normal,aihub_complaints_*"


python -m task.run --splits train \
    --tasks "nsmc,apeach,unsmile,klue_*,kobest_*,korquad_v1.0,kornlu_nli,kor_nlu,tl_text_entailment,niklex,nikl_absa,ko_nia_normal,aihub_complaints_*" \
    --max_instance_per_task 5000 \
    --use-ray

# for test, only klue and kobest, all items
python -m task.run --splits test --tasks "apeach,klue_*,kobest_*" --max_instance_per_task -1 --use-ray

# python -m task.push_to_hub iknow-lab/koflan-1101-zero-v3