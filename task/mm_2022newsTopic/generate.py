from task.base import BaseGenerator
from datasets import load_dataset
import random


class mm2022NewsTopicGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.topicList = []
        self.instructions = [
            "기사 내용을 읽고 대화의 분야를 분류하세요.",
            "기사 내용에 적절한 토픽으로 분류해보세요.",
            "이 기사와 관련된 분야를 파악해보세요.",
            "다음 기사를 이해하고 어떠한 주제를 떠올릴 수 있는지 생각해보세요.",
            "주어진 기사문을 읽고 적절한 분야를 정해보세요",
            "다음 기사문을 분석하여 어떠한 주제와 관련이 있는지 판단해보세요.",
            "주어진 기사가 어떤 주제와 일치하는지 결정해보세요.",
            "다음 기사를 읽고 어떤 주제에 대해 이야기하는지 파악해보세요.",
            "주어진 기사문을 해석하고 관련된 분야나 주제를 찾아보세요.",
            "주어진 기사 내용을 분석하여 주제를 도출해보세요.",
            "다음 기사를 해석하고 어떤 분야에 대한 내용인지 유추해보세요.",
            "주어진 기사문을 읽고 어떤 분야나 주제로 연관되어 있는지 파악해보세요.",
            "다음 기사 문장들을 해석하고 이와 관련된 주제를 분류해보세요.",
            "주어진 뉴스의 적절한 분야나 카테고리를 판단해보세요.",
            "다음 뉴스를 이해하고 어떤 주제와 관련되어 있는지 생각해보세요.",
            "주어진 뉴스 내용을 읽고 이와 관련된 분야나 주제를 결정해보세요.",
            "주어진 뉴스를 읽고 기자가 이야기하는 주제를 파악해보세요.",
            "다음 뉴스를 읽고 주요 내용을 정리하여 기사가 어떤 주제와 관련이 있는지 찾아보세요.",
            "주어진 뉴스를 해석하고 뉴스와 관련된 분야를 찾아보세요.",
            "주어진 기사문을 읽고 기사의 주제를 추론해보세요.",
            "다음 기사 내용을 분석하여 어떤 주제와 관련이 있는지 판단해보세요.",
            "주어진 문장들을 해석하고 어떤 분야나 주제에 대해 이야기하는지 판단해보세요.",
            "다음 뉴스의 토픽이 어떤 분야에 속하는지 유추해보세요.",
            "다음 문장들을 이해하고 어떤 주제와 관련된 뉴스인지 생각해보세요.",
            "주어진 텍스트들로 이루어진 기사가 속한 주제를 찾아보세요.",
            "다음 기사는 어떤 주제와 관련이 있는지 결정해보세요.",
        ]

    def generate(self, split: str):
        dataset = load_dataset(
            "iknow-lab/mm_2022newsTopic", split="train", use_auth_token=True
        ).shuffle(seed=42)
        dataset = dataset.train_test_split(test_size=0.1)[split]
        self.topicList = list(set([x["topic"] for x in dataset]))
        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            text = "".join(x for x in item["paragraph"])
            pos = item["topic"]
            neg = [x for x in self.topicList if x != pos]

            if len(neg) > 10:
                neg = random.sample(neg, k=10)

            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": neg,
            }
