from task.base import BaseGenerator

from datasets import load_dataset
import random


class NSMCGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.positives = ["긍정적인 리뷰", "긍정", "좋아한다", "선호"]
        self.negatives = ["부정적인 리뷰", "부정", "싫어한다", "불호"]

        self.instructions = [
            "이 문장의 감정을 분석하세요",
            "텍스트의 감정 분석을 수행하세요",
            "이 문장의 긍정 혹은 부정으로 분류하세요",
            "이 내용을 긍정/부정으로 분류하세요",
            "글을 읽고 긍정/부정으로 분류하세요",
            "이 글을 읽고 긍정적인지 부정적인지 분류해보세요.",
            "이 문장의 긍정 혹은 부정을 판별해주세요.",
            "이 댓글을 읽고 작성자의 감정을 파악해보세요.",
        ]

    def generate(self, split: str):
        dataset = load_dataset("nsmc", split=split)

        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            text = item["document"]
            label = item["label"]

            if label == 1:  # positive
                pos, neg = self.positives, self.negatives
            else:
                pos, neg = self.negatives, self.positives

            yield {
                "instruction": instruction,
                "input": text,
                "positives": [random.choice(pos)],
                "negatives": [random.choice(neg)],
            }
