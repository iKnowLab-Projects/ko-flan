from task.base import BaseGenerator
from datasets import load_dataset
import random


class CB2021Generator(BaseGenerator):
    # 국립국어원 추론_확신성 분석 말뭉치 2021
    def __init__(self) -> None:
        super().__init__()

        self.labels = {
            "함의": [
                "함의",
                "함의관계",
                "함의하고 있다",
                "수반",
                "수반관계",
                "동반관계",
                "동시에 일어나다",
                "함께하는 관계",
                "포함하는 문장",
            ],
            "중립": ["중립", "무관계", "상관없음", "관련없음", "상관관계가 없다", "아무런 관련이 없다"],
            "모순": ["모순", "모순관계", "상충관계", "대립하는 관계", "모순되는 관계", "반대되는 관계", "충돌하는 관계"],
        }
        self.label_keys = set(self.labels.keys())

    def generate(self, split: str):
        dataset = load_dataset(
            "iknow-lab/nikl_cb_2021_v1.1", split="train"
        ).train_test_split(0.1, seed=42)[split]

        for item in dataset:
            instruction = item["question"] + " " + item["prompt"]
            label = item["relation"]
            text = item["context+target"]

            pos = self.labels[label]
            neg = self.label_keys - set([label])
            neg = [x for n in neg for x in self.labels[n]]

            assert len(pos) > 0
            assert len(neg) > 0

            yield {
                "instruction": instruction,
                "input": text,
                "positives": pos,
                "negatives": neg,
            }
