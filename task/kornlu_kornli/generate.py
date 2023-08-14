from task.base import BaseGenerator

from datasets import load_dataset
from collections import defaultdict
import random


class KorNLUGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.instructions = [
            "다음 문장에 수반되는 내용을 작성하세요.",
            "다음 문장과 일관성 있는 내용을 가진 문장을 작성하세요.",
            "다음 문장과 일관성 있는 내용을 지닌 문장을 작성하세요.",
            "주어진 문장을 기반으로 이어질 수 있는 내용을 가진 문장을 작성해 주세요.",
            "다음 문장과 잘 연결되는 문장을 작성해 주세요.",
            "다음 문장과 자연스럽게 이어질 수 있도록 문장을 작성해 주세요.",
            "제시된 문장과 잘 어울리는 문장을 작성해 보세요.",
            "주어진 문장에 맞는 후속 문장을 작성해 주세요.",
            "주어진 문장에 따라 자연스럽게 이어지는 문장을 만들어 보세요.",
            "주어진 문장과 동일한 주제로 문장을 만들어 보세요.",
            "제시된 문장과 자연스럽게 이어지는 문장을 작성해 보세요.",
        ]

        self.negInstructions = [
            "다음 문장에 수반되지 않는 내용을 작성하세요.",
            "다음 문장과 일관성 없는 내용을 가진 문장을 작성하세요.",
            "다음 문장과 일관성 없는 내용을 지닌 문장을 작성하세요.",
            "주어진 문장을 기반으로 이어질 수 없는 내용을 가진 문장을 작성해 주세요.",
            "다음 문장과 잘 연결되지 않는 문장을 작성해 주세요.",
            "다음 문장 뒤에 올 문장으로 적절하지 않은 문장을 작성해 주세요.",
            "제시된 문장과 어울리지 않는 문장을 작성해 보세요.",
            "주어진 문장에 어울리지 않는 후속 문장을 작성해 주세요.",
            "주어진 문장에 따라 자연스럽게 이어지지 않는 문장을 만들어 보세요.",
            "주어진 문장과 동일하지 않은 주제로 문장을 만들어 보세요.",
            "제시된 문장과 자연스럽게 이어지지 않는 문장을 작성해 보세요.",
        ]

    def generate(self, split: str):
        dataset = load_dataset("kor_nlu", "nli", split=split)
        dataset = dataset.filter(lambda x: x["label"] != 1)

        grouped_dataset = defaultdict(list)
        for item in dataset:
            premise = item["premise"]
            grouped_dataset[premise].append(item)

        for premise in grouped_dataset:
            instruction = random.choice(self.instructions)
            text = premise

            pos = list(filter(lambda x: x["label"] == 0, grouped_dataset[premise]))
            neg = list(filter(lambda x: x["label"] == 2, grouped_dataset[premise]))

            pos_list = [y["hypothesis"] for y in pos]
            neg_list = [y["hypothesis"] for y in neg]

            if len(neg) > 0 and len(pos) > 0:
                yield {
                    "instruction": instruction,
                    "input": text,
                    "positives": pos_list,
                    "negatives": neg_list,
                }

            instruction = random.choice(self.negInstructions)

            if len(neg) > 0 and len(pos) > 0:
                yield {
                    "instruction": instruction,
                    "input": text,
                    "positives": neg_list,
                    "negatives": pos_list,
                }
