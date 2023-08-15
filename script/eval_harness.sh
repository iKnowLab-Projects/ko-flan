
lm-eval --model hf-seq2seq \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000 \
    --tasks nsmc,korquad,korunsmile,kohatespeech,kobest_*, klue_*\
    --device cuda:0