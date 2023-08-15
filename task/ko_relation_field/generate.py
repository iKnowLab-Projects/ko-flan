from task.base import BaseGenerator
from dataclasses import dataclass
from datasets import load_dataset
import random
from typing import List
import json


class KO_RELATION_FIELDGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.instructions = [
     "주어진 문장을 기반으로 그 내용이 어느 분야에 속하는지 판단해주세요.",
    "이 문장의 주제나 분야는 무엇이라고 생각하시나요?",
    "다음의 문장이 어떤 분야에 대한 내용을 담고 있는지 알려주세요.",
    "주어진 문구를 보고, 가장 관련된 분야나 주제를 선택해주세요.",
    "이 내용이 주로 어떤 분야에서 다루어질까요?",
    "이 문장이 어느 분야의 지식을 필요로 하나요?",
    "주어진 텍스트의 주제나 분야를 짚어보세요.",
    "이 문장이 어떤 분야에 관한 것인지 분석해주세요.",
    "이 문장에 가장 잘 어울리는 분야나 주제는 무엇인가요?",
    "다음의 텍스트가 어떤 분야의 내용을 기반으로 하고 있는지 말해보세요.",
    "주어진 내용이 어느 분야에서 주로 다루어지는 내용인지 알려주세요.",
    "이 문구가 어느 분야와 가장 밀접하게 연관되어 있다고 생각하시나요?",
    "주어진 문장을 읽고, 그 내용이 어떤 분야와 관련이 있는지 파악해주세요.",
    "이 문장을 기반으로 그 주제나 분야를 추측해보세요.",
    "어떤 분야의 전문가가 이 문장을 보면 흥미를 느낄까요?",
    "이 문장이 어떤 분야에 대한 내용을 포함하고 있는지 설명해주세요.",
    "주어진 문장의 내용을 바탕으로 그 분야나 주제를 정확하게 지칭해보세요.",
    "이 내용이 어느 분야에 대한 지식을 반영하고 있는지 말해보세요.",
    "문장이 언급하고 있는 주제나 내용이 어느 분야에 속한다고 생각하시나요?",
    "이 문장을 보고 그것이 어떤 분야에 대한 것인지 즉시 판단해주세요."
        ]

        self.negative_sample =  ['지역_사회', '국제', 'IT_과학', '연예', '문화', '정치', '의약학', '스포츠', '경제',
       '기계공학', '인문학', '사회과학', '전기전자']



    def generate(self, split: str):

        dataset = load_dataset("iknow-lab/korean_relation", split=split)

        for item in dataset:
            text =  item["sentence"]
            pos = item['field']
            neg = random.choices(
                list(filter(lambda x: x != item['field'], self.negative_sample)),k=4)
            instruction = random.choice(self.instructions)

            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": neg,
            }
