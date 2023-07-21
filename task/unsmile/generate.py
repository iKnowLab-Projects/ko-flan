from task.base import BaseGenerator
from datasets import load_dataset
import random


class UnSmileGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.positives = ["있습니다", "예", "네", "그렇습니다", "포함합니다", "혐오 표현입니다"]
        self.negatives = ["없습니다", "아니요", "아닙니다", "포함하지 않습니다", "혐오 표현이 아닙니다"]
        self.instructions = [
            "이 문장에 비난적인 표현이 포함되어 있나요?",
            "비판적인 표현이 이 문장에 포함되어 있습니까?",
            "이 문장은 의도적인 공격을 포함하고 있습니까?",
            "모욕적인 언어가 이 문장에 담겨져 있습니까?",
            "이 문장에는 차별적인 발언이 포함되어 있나요?",
            "비난, 비판적인 표현이 이 문장에 녹아 있습니까?",
            "이 문장은 의도적인 공격적인 표현을 담고 있습니까?",
            "모욕적인 언어가 이 문장에 들어있나요?",
            "이 문장에는 차별적인 발언이 있습니까?",
            "비난, 비판적인 표현이 이 문장에 있습니까?",
            "혐오적인 내용이 포함되어 있나요?",
            "혐오 표현이 문장에 있나요?",
        ]

    def generate(self, split: str):
        if split == "test":
            split = "valid"

        dataset = load_dataset('smilegate-ai/kor_unsmile', split=split)

        for item in dataset:
            instruction = random.choice(self.instructions)
            text = item["문장"]
            
            if item["clean"] == 1: # clean-speech 
                pos, neg = self.negatives, self.positives
            else:
                pos, neg = self.positives, self.negatives
                
            yield {
                "instruction": instruction,
                "input": text,
                "positives": pos,
                "negatives": neg,
            }
