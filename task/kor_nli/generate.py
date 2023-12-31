from task.base import BaseGenerator

from datasets import load_dataset
import random


class KorNLIGenerator(BaseGenerator):
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
            "제시된 문장과 자연스럽게 이어지는 문장을 작성해 보세요."
        ]

    def generate(self, split: str):
        if split == "test":
            raise Exception("'test' split doesn't exist in dataset")
        else:
            dataset = load_dataset("kor_nli", "multi_nli", split=split)
            dataset = dataset.filter(lambda x: x["label"] != 1)

            for item in dataset:
                instruction = random.choice(self.instructions)
                text = item["premise"]

                pos = [item["hypothesis"]]
                random_item = random.choice(dataset)
                neg = [random_item["hypothesis"]]

                if len(neg) > 0 and len(pos) > 0 and item["label"] == 0:
                    yield {
                        "instruction": instruction,
                        "input": text,
                        "positives": pos,
                        "negatives": neg,
                    }
