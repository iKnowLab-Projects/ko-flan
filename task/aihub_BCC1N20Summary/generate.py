from task.base import BaseGenerator
from datasets import load_dataset
import random


class aihubBCC1N20SummaryGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.topicList = []
        self.instructions = [
            "대본을 읽고 적절한 요약문을 생성하세요.",
            "다음 대본을 한 문장으로 요약해주세요.",
            "다음 대본을 하나의 문장으로 줄여보세요",
            "다음 대본을 이해하고 요약해보세요.",
            "주어진 대본을 읽고 적절한 요약문을 만들어 주세요.",
            "주어진 내용을 분석하여 대본을 간추려주세요.",
            "다음 대본의 내용을 논리적이고 짧게 정리해주세요.",
            "다음을 읽고 어떤 이야기를 하는지 짧게 나타내보세요.",
            "주어진 대본을 해석하고 요약해보세요.",
            "다음 대본들을 읽고 주요 내용을 정리하여 간단히 표현해주세요.",
        ]
    def generate(self, split: str):
        dataset = load_dataset(
            "iknow-lab/aihub_BCC1N20Summary", split=split, token=True
        ).shuffle(seed=42)
        
        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            text = "".join(x for x in item["passage"])
            pos = item["1_sentence_summary"]
            neg = []

            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": neg,
            }
    
