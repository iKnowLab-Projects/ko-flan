from task.base import BaseGenerator
from datasets import load_dataset
import random


class aihubDialogSummarySummaryGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.topicList = []
        self.instructions = [
            "대화 내용을 읽고 적절한 요약문을 생성하세요.",
            "대화 내용을 요약해주세요.",
            "이 대화를 간단한 문장으로 줄여보세요",
            "다음 대화를 이해하고 요약해보세요.",
            "주어진 대화문을 읽고 적절한 요약문을 만들어 주세요.",
            "다음 대화문을 분석하여 하나의 문장으로 요약해주세요.",
            "주어진 대화 상황을 한 문장으로 정리해주세요.",
            "다음 대화를 읽고 어떤 이야기를 하는지 짧게 한 문장으로 나타내보세요.",
            "주어진 대화문을 해석하고 한 문장으로 요약해보세요.",
            "주어진 대화문을 읽고 이야기를 요약해주세요.",
            "다음 문장들을 읽고 주요 내용을 정리하여 대화를 한 문장으로 표현해주세요.",
            "주어진 문장들을 해석하고 대화를 간단히 한 문장으로 요약해보세요.",
            "주어진 텍스트들을 읽고 대화를 하나의 문장으로 나타내보세요.",
            "주어진 텍스트들로 이루어진 대화를 짧은 문장으로 요약해주세요.",
            "다음 대화를 간단하게 요약해보세요.",
        ]

    def generate(self, split: str):
        dataset = load_dataset(
            "iknow-lab/aihub_dialogSummary", split=split, token=True
        ).shuffle(seed=42)
        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            text = "".join(x for x in item["conversation"])
            pos = item["summary"]
            neg = []

            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": neg,
            }
