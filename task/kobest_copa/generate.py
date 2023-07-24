from task.base import BaseGenerator
from dataclasses import dataclass
from datasets import load_dataset
import random
from typing import List


class KOBEST_COPAGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.instructions = {
        "원인": [
    "이 문장의 원인을 나타내는 문장을 작성하시오.",
    "이 문장이 발생한 원인을 설명하는 문장을 제시하시오.",
    "이 문장에 대한 원인을 표현하는 문장을 써보세요.",
    "이 문장이 일어난 이유를 설명하는 문장을 만드세요.",
    "이 문장이 왜 일어났는지를 나타내는 문장을 작성하시오.",
    "이 문장이 일어나게된 원인을 설명하는 문장을 작성하시오",
    "이 문장이 생겨난 근본적인 원인을 나타내는 문장을 작성하시오.",
    "이 문장이 왜 발생했는지 그 원인을 상세히 나타내는 문장을 작성하시오.",
    "이 문장이 발생한 주된 원인을 서술하는 문장을 작성하시오.",
    "이 문장이 어떤 상황에서 발생했는지 그 원인을 설명하는 문장을 작성하시오.",
    "이 문장이 어떤 조건 하에서 나타났는지 그 원인을 설명하는 문장을 작성하시오.",
    "이 문장이 만들어진 원인을 상세히 설명하는 문장을 작성하시오."
             ],
        "결과": [
    "이 문장의 결과를 나타내는 문장을 작성하시오.",
    "이 문장이 초래한 결과를 설명하는 문장을 제시하시오.",
    "이 문장에 따른 결과를 표현하는 문장을 써보세요.",
    "이 문장이 어떤 결과를 가져왔는지 설명하는 문장을 만드세요.",
    "이 문장이 끝내 무엇을 초래했는지를 나타내는 문장을 작성하시오.",
    "이 문장이 끝난 후의 상황을 설명하는 문장을 작성하시오.",
    "이 문장이 초래한 결과를 상세하게 서술하시오.",
    "이 문장이 뒤따르는 결과를 서술하는 문장을 써보세요.",
    "이 문장이 만든 결과를 표현하는 문장을 만드세요.",
    "이 문장이 어떤 영향을 미쳤는지를 나타내는 문장을 작성하시오.",
    "이 문장의 결론을 나타내는 문장을 작성하시오.",
    "이 문장이 가져온 결과를 서술하는 문장을 작성하시오.",
    "이 문장의 이후 상황을 표현하는 문장을 작성하시오.",
    "이 문장이 초래한 변경 사항을 설명하는 문장을 작성하시오.",
    "이 문장이 미친 결과를 나타내는 문장을 작성하시오."
            ]
        }

    def generate(self, split: str):
        dataset = load_dataset('skt/kobest_v1','copa', split=split)

        for item in dataset:
            label = item["question"]
            if label =='원인 ':
                label = "원인"
            instruction = random.choice(self.instructions[label])
            text = item["premise"]
            pos = item["alternative_1"]
            neg = item["alternative_2"]

            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": [neg],
            }
