from task.base import BaseGenerator
from datasets import load_dataset
import random
# import jsonlines


class aihubComplaintsTopicGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.topicList = []
        self.instructions = [
            "다음 민원을 보고, 카테고리를 분류해보세요.",
            "이 민원 내용을 확인하고, 제일 밀접한 분야로 분류하시오",
            "이 민원이 어떤 카테고리에 속하는지 판단해보세요.",
            "아래 민원 내용을 읽고, 어떤 분류에 해당하는지 알려주세요.",
            "다음 내용을 확인하고, 이것을 어떤 그룹으로 묶아야 할지 선택하시오",
            "민원 내용을 검토하고, 어떤 카테고리에 해당하는지 알려주세요.",
            "이 민원을 어떤 유형으로 분류하면 좋을까요?",
            "아래 민원을 읽고, 이를 어떤 분야에 속한다고 생각하십니까?",
            "이 민원이 어떤 분류에 해당하는지 알려주세요.",
            "민원 내용을 확인하고, 어떤 카테고리로 분류해야 합니까?",
            "다음 민원을 보고, 이것을 어떤 범주로 분류하시겠어요?",
            "이 민원이 어떤 카테고리에 속할 것 같나요?",
            "아래의 민원 내용을 검토하고, 어떤 분류에 속하는지 알려주세요.",
            "민원을 확인하고, 이것을 어떤 유형으로 구분하시겠어요?",
            "이 민원을 어떤 분야로 분류하면 좋을까요?",
            "아래 민원 내용을 읽고, 이를 어떤 범주에 속한다고 생각하십니까?",
            "이 민원을 보고, 어떤 그룹으로 묶어야 할 것 같나요?",
            "다음 민원을 보고, 이것을 어떤 카테고리로 분류하시겠어요?",
            "이 민원이 어떤 분류에 속할 것 같아요?",
            "아래의 민원 내용을 검토하고, 어떤 카테고리에 속하는지 알려주세요.",
            "민원을 확인하고, 이것을 어떤 유형으로 판단하시겠어요?",
            "이 민원을 어떤 분야로 분류하면 좋을까요?",
            "아래 민원을 읽고, 이를 어떤 범주에 속한다고 생각하십니까?",
            "이 민원을 보고, 어떤 그룹으로 묶어야 할 것 같나요?",
            "다음 민원을 보고, 이것을 어떤 카테고리로 분류하시겠어요?",
            "이 민원이 어떤 분류에 속할 것 같아요?",
            "아래의 민원 내용을 검토하고, 어떤 카테고리에 속하는지 알려주세요.",
            "민원을 확인하고, 이것을 어떤 유형으로 판단하시겠어요?",
            "이 민원을 어떤 분야로 분류하면 좋을까요?",
            "아래 민원을 읽고, 이를 어떤 범주에 속한다고 생각하십니까?",
        ]

    def generate(self, split: str):
        dataset = load_dataset(
            "iknow-lab/aihub_complaints", split=split, use_auth_token=True
        ).shuffle(seed=42)
        self.topicList = list(set(dataset['category']))
        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            text = item["text"]
            pos = item["category"]
            neg = [x for x in self.topicList if x != pos]
            if len(neg) > 10:
                neg = random.sample(neg, k=10)
            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": neg,
            }
