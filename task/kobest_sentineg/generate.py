from task.base import BaseGenerator
from datasets import load_dataset
import random


class KobestBoolqGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.positives = [
            "좋은 평가입니다.",
            "긍정적인 내용입니다.",
            "좋습니다.",
            "긍정문입니다.",
            "우수입니다.",
            "긍정적인 표현입니다.",
            "만족스럽습니다.",
        ]
        self.negatives = [
            "나쁜 평가입니다.",
            "부정적인 내용입니다.",
            "별로입니다.",
            "부정문입니다.",
            "미흡합니다.",
            "부정적인 표현입니다.",
            "불만족스럽스니다.",
        ]
        self.instruction = [
            "다음 문장이 긍정적인지 부정적인지 분류하세요.",
            "다음 평가가 좋은지 나쁜지 판단해주세요.",
            "평가가 좋은지 나쁜지 판별해주세요.",
            "이 문장이 긍정적인지 부정적인지 분류해주세요."
            "이 평가가 양호한지 불량한지 판단해주세요."
            "평가가 긍정인지 부정인지 판별해주세요."
            "이 문장이 긍정적인 표현인지 부정적인 표현인지 구별해주세요."
            "다음 평가 결과가 좋다고 볼 수 있는지 나쁘다고 볼 수 있는지 확인해주세요."
            "평가 결과가 만족스러운지 아닌지 판단해주세요."
            "이 문장이 긍정적인 감정을 표현하는지 부정적인 감정을 표현하는지 확인해주세요."
            "이 평가가 긍정적인 결과를 나타내는지 부정적인 결과를 나타내는지 판단해주세요."
            "평가 결과가 우수한지 미흡한지 판별해주세요."
            "다음 문장이 긍정적인 표현을 사용하고 있는지 부정적인 표현을 사용하고 있는지 분류해주세요."
            "이 평가가 성공적인지 실패적인지 판단해주세요."
            "평가 결과가 긍정적인지 부정적인지 판별해주세요.",
        ]

    def generate(self, split: str):
        dataset = load_dataset("skt/kobest_v1", "sentineg", split=split)

        for item in dataset:
            input = item["sentence"]

            if item["label"] == 1:  # true
                pos, neg = self.positives, self.negatives
            else:
                pos, neg = self.negatives, self.positives
            instruction = random.choice(self.instructions)

            yield {
                "instruction": instruction,
                "input": input,
                "positives": pos,
                "negatives": neg,
            }
