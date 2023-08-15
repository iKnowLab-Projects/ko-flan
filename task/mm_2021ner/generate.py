from task.base import BaseGenerator
from datasets import load_dataset
import random


class mm2021NERGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.labelList = []
        self.instructions = [
            "다음 문장에서 개체와의 관계를 분류하세요.",
            "문장 내에 있는 개체를 적절한 관계로 분류해보세요.",
            "문장과 개체의 관계를 파악해보세요.",
            "다음 문장과 개체를 이해하고 어떠한 관계를 떠올릴 수 있는지 생각해보세요.",
            "주어진 문장과 개체를 읽고 적절한 관계를 정해보세요",
            "개체와 문장을 분석하여 어떻게 관련이 있는지 판단해보세요.",
            "주어진 개체가 문장에서 어떻게 쓰이는지 결정해보세요.",
            "문장내에 있는 개체의 쓰임에 대해 파악해보세요.",
            "주어진 문장과 해석하고 개체가 관련된 분야나 주제를 찾아보세요.",
            "주어진 내용을 분석하여 문장 내 개체의 역할을 도출해보세요.",
            "다음 문장을 해석하고 개체가 어떤 분야에 대한 단어인지 유추해보세요.",
            "주어진 문장과 개체를 읽고 어떤 분야나 주제로 연관되어 있는지 파악해보세요.",
            "다음 문장과 개체를 해석하고 관련된 주제를 분류해보세요.",
            "문장 내 주어진 개체의 적절한 분야나 카테고리를 판단해보세요.",
            "다음 개체를 문장과 함께 이해하고 어떤 분야로 관련되어 있는지 생각해보세요.",
            "주어진 문장과 개체가 관련된 분야나 주제를 결정해보세요.",
            "주어진 문장을 읽고 개체가 의미하는 카테고리를 파악해보세요.",
            "다음 문장을 읽고 주요 내용을 정리하여 개체가 문장에서 어떻게 관련이 있는지 찾아보세요.",
            "주어진 문장을 해석하고 개체가 속하는 분야를 찾아보세요.",
            "주어진 문장을 읽고 개체의 역할에 대해 추론해보세요.",
            "다음 내용을 분석하여 개체가 어떤 주제로 관련이 있는지 판단해보세요.",
            "주어진 문장을 해석하고 개체가 어떤 분야나 주제에 속하는지 판단해보세요.",
            "다음 문장 내 개체가 어떤 분야에 속하는지 유추해보세요.",
            "다음 문장을 이해하고 어떤 분야와 관련된 개체인지 생각해보세요.",
            "주어진 텍스트로 이루어진 문장에서 개체가 속한 주제를 찾아보세요.",
            "다음 문장에서는 개체가 어떤 주제와 관련이 있는지 결정해보세요.",
        ]

    def generate(self, split: str):
        dataset = load_dataset(
            "iknow-lab/mm_2021NER", split="train", use_auth_token=True
        ).shuffle(seed=42)
        dataset = dataset.train_test_split(test_size=0.1)[split]
        self.labelList = list(set([x["label"] for x in dataset]))
        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            text = f"문장: \"{item['sentence']}\", 개체: {item['title']}"
            pos = item["label"]
            neg = [x for x in self.labelList if x != pos]

            if len(neg) > 10:
                neg = random.sample(neg, k=10)

            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": neg,
            }
