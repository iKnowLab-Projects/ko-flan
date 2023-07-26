from task.base import BaseGenerator
from datasets import load_dataset


class KobestBoolqGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.positives = ["맞습니다", "네", "그렇다", "그렇습니다"]
        self.negatives = ["틀립니다", "아니요", "아니다", "아닙니다"]

    def generate(self, split: str):

        dataset = load_dataset("skt/kobest_v1", "boolq", split=split)

        for item in dataset:
            instruction = item["question"]
            input = item["paragraph"]
            
            if item["label"] == 1: # true
                pos, neg = self.positives, self.negatives
            else:
                pos, neg = self.negatives, self.positives
                
            yield {
                "instruction": instruction,
                "input": input,
                "positives": pos,
                "negatives": neg,
            }
