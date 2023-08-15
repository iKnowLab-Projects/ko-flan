
# tasks="nsmc,korquad,korunsmile,kohatespeech,kobest_*,klue_*"
tasks="kobest_*"

python -m eval.eval_harness_main \
    --model hf-seq2seq \
    --model_args "pretrained=checkpoint/ke-t5-small-0813/epoch-5,max_length=512" \
    --tasks "$tasks" \
    --device cuda:0