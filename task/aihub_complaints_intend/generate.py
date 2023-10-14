from task.base import BaseGenerator
from datasets import load_dataset
import random
# import jsonlines


class aihubComplaintsIntendGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.topicList = []
        self.instructions = [
            "민원의 의도를 파악하여 적절한 카테고리로 분류하세요.",
            "이 민원이 어떤 의도를 가지고 있는지 파악하고 해당 의도에 따라 분류하세요.",
            "아래 민원 내용을 읽고, 민원 제기자의 의도를 정확하게 파악해주세요.",
            "민원의 주요 의도를 이해하고, 그에 맞게 분류하세요.",
            "다음 내용을 검토하고, 민원 제기자가 어떤 의도를 가지고 있는지 파악하세요.",
            "민원 제기자의 의도를 확인하고, 이를 반영하여 적절한 카테고리로 분류하세요.",
            "이 민원의 목적을 분명하게 이해하고, 해당 의도에 따라 분류해주세요.",
            "민원의 주된 의도를 찾아내고, 그에 따라 어떤 분류에 속할지 결정하세요.",
            "민원 제기자가 어떤 의도로 이 민원을 제출했는지 파악해주세요.",
            "민원의 핵심 의도를 이해하고, 그에 따라 적절한 카테고리로 분류하세요.",
            "민원 제기자의 주요 의도를 이해하고, 그에 따라 어떤 분류에 속할지 결정하세요.",
            "민원 제기자의 목적을 파악하고, 그에 따라 이 민원을 어떻게 분류할 것인가요?",
            "민원을 검토하고, 민원 제기자의 의도에 따라 어떤 분류에 속하는지 알려주세요.",
            "민원 제기자의 의도를 이해하고, 이를 반영하여 어떤 카테고리로 분류하세요.",
            "민원의 목적을 파악하고, 그에 따라 어떤 범주로 분류하면 좋을까요?",
            "민원 내용을 읽고, 민원 제기자의 의도를 정확하게 이해해주세요.",
            "민원 제기자가 어떤 목적을 가지고 있는지 파악하고, 그에 맞게 분류하세요.",
            "다음 민원을 보고, 민원 제기자의 의도를 정확하게 이해하고 분류하세요.",
            "민원의 주된 의도를 이해하고, 그에 따라 적절한 카테고리로 분류하세요.",

        ]

    def generate(self, split: str):
        dataset = load_dataset(
            "iknow-lab/aihub_complaints", split=split, use_auth_token=True
        ).shuffle(seed=42)
        self.topicList = list(set(dataset['predication']))
        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            text = item["text"]
            pos = item["predication"]
            neg = [x for x in self.topicList if x != pos]
            if len(neg) > 10:
                neg = random.sample(neg, k=10)
            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": neg,
            }
