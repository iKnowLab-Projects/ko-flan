from task.base import BaseGenerator
from dataclasses import dataclass
from datasets import load_dataset
import random
from typing import List


class KOBEST_HELLASWAGGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.instructions = [
            "주어진 정보를 분석하고 이에 맞는 결론 문장을 작성하십시오.",
            "텍스트를 주의 깊게 읽고, 가장 어울리는 마지막 문장을 생성해보세요.",
            "텍스트의 내용을 이해한 뒤, 적절한 마무리 문장을 서술해주세요.",
            "문단을 정독한 후, 가장 적절하게 이어질 마지막 문장을 작성하십시오.",
            "제공된 내용을 기반으로 한, 가장 자연스러운 마지막 문장을 써 주시기 바랍니다.",
            "읽은 내용에 따라 이어질 수 있는 가장 적절한 문장을 선택하십시오.",
            "문서의 내용을 바탕으로, 이를 잘 마무리 지을 수 있는 문장을 찾아보세요.",
            "문장의 흐름을 이해하고, 이를 마무리할 수 있는 최선의 문장을 작성해주세요.",
            "제시된 내용을 바탕으로 마지막에 올 수 있는 문장을 제시하십시오.",
            "기존의 내용을 분석하여 그에 맞는 최종 문장을 작성해 주세요.",
            "해당 내용을 철저히 분석한 후, 이를 마무리 지을 적절한 문장을 찾아주세요.",
            "주어진 텍스트의 내용을 바탕으로, 가장 적절한 마지막 문장을 구성해보십시오.",
            "주어진 정보를 잘 이해하고, 이에 알맞는 마지막 문장을 제안해주세요.",
            "텍스트를 주의 깊게 파악한 후, 이를 잘 마무리할 수 있는 문장을 만들어주세요.",
            "주어진 내용을 끝마칠 수 있는 가장 적절한 문장을 찾아주세요.",
            "텍스트의 마지막을 이어갈 수 있는 가장 좋은 문장을 찾아 주시기 바랍니다.",
            "텍스트의 흐름을 이해한 후, 가장 적합한 마지막 문장을 제시해주세요.",
            "문장을 정독하고 이를 완성할 수 있는 최적의 문장을 작성해 주세요.",
            "제시된 텍스트를 꼼꼼히 읽고, 이를 마무리 짓는 가장 적절한 문장을 작성해주세요.",
            "텍스트를 신중히 분석한 후, 가장 잘 어울리는 마지막 문장을 제시해주세요.",
        ]

    def generate(self, split: str):
        dataset = load_dataset("skt/kobest_v1", "hellaswag", split=split)
        for item in dataset:
            label = item["label"]
            text = item["context"]
            pos = list(item.values())[label + 1]
            neg = [
                item
                for index, item in enumerate(list(item.values()))
                if (index != label + 1) and (index != 5) and (index != 0)
            ]
            instruction = random.choice(self.instructions)

            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": neg,
            }
