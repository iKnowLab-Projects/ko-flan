from task.base import BaseGenerator
from datasets import load_dataset
import random


class aihubThesisSummaryGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.topicList = []
        self.instructions = [
            "내용을 읽고 적절한 요약문을 생성하세요.",
            "다음 텍스트를 요약해주세요.",
            "다음 문장을 간단한 문장으로 줄여보세요",
            "다음 자료를 이해하고 요약해보세요.",
            "주어진 문장을 읽고 적절한 요약문을 만들어 주세요.",
            "주어진 내용을 분석하여 주장을 간추려주세요.",
            "다음 문장의 내용을 논리적으로 정리해주세요.",
            "다음을 읽고 어떤 이야기를 하는지 짧게 나타내보세요.",
            "주어진 지문을 해석하고 요약해보세요.",
            "다음 문장들을 읽고 주요 내용을 정리하여 간단히 표현해주세요.",
        ]
    def twiceLoop(self, dataset, which):
        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            text = "".join(x for x in item[which+"_passage"])
            pos = item[which+"_summary"]
            neg = []

            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": neg,
            }
    def generate(self, split: str):
        dataset = load_dataset(
            "iknow-lab/aihub_dialogSummary", split=split, token=True
        ).shuffle(seed=42)
        twiceLoop(dataset = dataset, which = "section")
        twiceLoop(dataset = dataset, which = "entire")
        
