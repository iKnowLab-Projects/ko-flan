from task.base import BaseGenerator
from datasets import load_dataset
import random
import itertools

# import jsonlines


class aihubComplaintsKeywordGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.topicList = []
        self.instructions = [
            "이 민원의 핵심 키워드를 찾아보세요.",
            "민원 내용에서 주요 키워드를 찾아내고 알려주세요.",
            "아래 민원을 검토하고, 그 안의 주요 키워드를 식별해보세요.",
            "민원의 핵심 키워드를 확인하고 알려주세요.",
            "다음 내용을 보고, 그 안의 주요 키워드를 파악해보세요.",
            "민원 내용을 검토하고, 어떤 키워드가 중요한지 알려주세요.",
            "민원 내용을 확인하고, 어떤 키워드가 주요한지 확인하세요.",
            "민원에서 어떤 키워드가 중요한지 식별하고 알려주세요.",
            "다음 민원을 보고, 그 안의 핵심 키워드를 찾아내세요.",
            "민원 내용을 검토하고, 어떤 키워드가 핵심인지 알려주세요.",
            "민원 내용을 읽고, 어떤 키워드가 중요한지 파악해주세요.",
            "다음 민원을 보고, 그 안의 주요 키워드를 찾아보세요.",
            "민원 내용을 검토하고, 어떤 키워드가 주요한지 확인하세요.",
            "다음 민원을 보고, 그 안의 핵심 키워드를 찾아내고 단어를 말해주세요.",
            "민원 내용을 검토하고, 어떤 키워드가 중요한지 파악해보세요.",
            "민원의 핵심 키워드를 확인하고, 그에 따라 단어를 제공하세요.",
            "민원 내용을 읽고, 어떤 키워드가 중요한지 확인해주세요.",
            "다음 민원을 보고, 그 안의 주요 키워드를 찾아내고 알려주세요.",
            "민원의 핵심 키워드를 확인하고, 그에 따라 이 단어를 파악하세요.",
            "다음 민원을 보고, 그 안의 핵심 키워드를 찾아내고 알려주세요.",
            "민원 내용을 검토하고, 어떤 키워드가 중요한지 확인하세요.",
            "민원 내용을 읽고, 어떤 키워드가 중요한지 파악해주세요.",

        ]

    def generate(self, split: str):
        dataset = load_dataset(
            "iknow-lab/aihub_complaints", split=split, use_auth_token=True
        ).shuffle(seed=42)

        self.topicList = list(set(itertools.chain(*(dataset['keyword']))))
        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            text = item["text"]
            pos = item["keyword"]
            neg = [x for x in self.topicList if x not in pos]
            if len(neg) > 10:
                neg = random.sample(neg, k=10)
            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": neg,
            }
