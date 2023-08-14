from task.base import BaseGenerator
from datasets import load_dataset
import random


class UnSmileGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.labels = [
            "여성/가족",
            "남성",
            "성소수자",
            "인종/국적",
            "연령",
            "지역",
            "종교",
            "기타 혐오",
            "악플/욕설",
            "없다",
        ]
        self.instructions = [
            "이 문장에 무엇에 대한 비난적인 표현이 포함되어 있나요?",
            "어떠한 비판적인 표현이 이 문장에 포함되어 있습니까?",
            "이 문장은 누구에 대한 의도적인 공격을 포함하고 있습니까?",
            "누구를 대상으로 한 모욕적인 언어가 이 문장에 담겨져 있습니까?",
            "이 문장에는 어떠한 차별적인 발언이 포함되어 있나요?",
            "어떤 비난, 비판적인 표현이 이 문장에 녹아 있습니까?",
            "이 문장은 특정 집단에 대한 의도적인 공격적인 표현을 담고 있습니까?",
            "누구에게 모욕적인 언어가 이 문장에 들어있나요?",
            "이 문장에는 어떠한 집단에 대한 차별적인 발언이 있습니까?",
            "어떤 종류의 비난, 비판적인 표현이 이 문장에 있습니까?",
            "어떤 집단에 대한 혐오적인 내용이 포함되어 있나요?",
            "특정 그룹, 대상에 대한 혐오 표현이 문장에 있나요?",
        ]

    def generate(self, split: str):
        if split == "test":
            split = "valid"

        dataset = load_dataset("smilegate-ai/kor_unsmile", split=split)

        for item in dataset:
            instruction = random.choice(self.instructions)
            text = item["문장"]
            pos_labels = []
            neg_labels = []

            for index, i in enumerate(item["labels"]):
                if i == 1:
                    pos_labels.append(self.labels[index])
                else:
                    neg_labels.append(self.labels[index])

            if len(pos_labels) == 0 or len(neg_labels) == 0:
                continue

            yield {
                "instruction": instruction,
                "input": text,
                "positives": pos_labels,
                "negatives": neg_labels,
            }
