from task.base import BaseGenerator
from datasets import load_dataset
from itertools import chain
from typing import List



def check_label(negative: str, positives: List[str]):
    for pos in positives:
        if negative in pos or pos in negative:
            return False
        
    return True

class KorQuADv1Generator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, split: str):
        if split == "test":
            split = "dev"

        dataset = load_dataset("KETI-AIR/korquad", "v1.0", split=split)
        data = dataset.to_pandas()

        for item in dataset:
            instruction = item["question"]
            text = item["context"]
            label = item["answers"]["text"]
            negative_labels = data[(data.context == text) & (data.question != instruction)]
            negative_labels = negative_labels.answers.values
            # 중복 제거
            negative_labels = set(chain(*[x['text'].tolist() for x in negative_labels]))
            # 정답 라벨과 동일하거나 정답 라벨과 겹치는 negative 라벨은 제거한다.
            negative_labels = [x for x in negative_labels if check_label(x, label)]
            
            yield {
                "instruction": instruction,
                "input": text,
                "positives": label,
                "negatives": negative_labels,
            }
