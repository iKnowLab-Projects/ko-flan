from task.base import BaseGenerator
from dataclasses import dataclass
from datasets import load_dataset
import random
from typing import List
import json


class KOWOWSENTENCEPASSAGEGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.instructions = [
            "이 문장을 읽고 가장 관련히 깊은 단락을 찾아보세요",
            "문장을 주의 깊게 읽고 관련된 부분을 찾아주세요",
            "이 내용과 가장 밀접하게 연관된 부분을 찾아보세요",
            "이 텍스트를 기반으로 관련된 단락을 찾아주세요",
            "문장을 분석하고 가장 연관성이 높은 부분을 탐색하세요",
            "이 내용을 참고하여 관련된 문단을 찾아보세요",
            "이 문장과 밀접한 관련이 있는 부분을 찾아주세요",
            "텍스트를 꼼꼼히 읽고 연관된 부분을 검색하세요",
            "이 문장을 기준으로 관련성이 높은 단락을 찾아주세요",
            "이 내용과 일치하는 관련된 부분을 탐색하세요",
            "문장을 분석하고 관련성 있는 부분을 찾아보세요",
            "이 텍스트와 밀접한 관계가 있는 단락을 찾아주세요",
            "이 문장을 참고하여 관련된 내용을 찾아보세요",
            "텍스트와 연관된 가장 관련된 부분을 검색하세요",
            "이 문장과 관련이 깊은 부분을 탐색해주세요",
            "텍스트를 기반으로 연관성 있는 부분을 찾아보세요",
            "이 문장에 가장 밀접하게 연결된 부분을 찾아주세요",
            "이 내용을 분석하여 관련된 단락을 찾아보세요",
            "문장을 꼼꼼히 읽고 관련이 있는 부분을 찾아주세요",
            "이 텍스트와 관련된 부분을 탐색하세요",
            "이 문장과 관련이 있는 가장 중요한 부분을 찾아보세요",
            "텍스트를 읽고 그와 관련된 부분을 검색하세요",
            "이 내용과 관련이 깊은 단락을 찾아주세요",
            "문장과 연관된 내용을 찾아보세요",
            "이 텍스트와 밀접하게 연관된 부분을 찾아보세요",
            "이 문장을 기반으로 관련된 내용을 탐색하세요",
            "텍스트와 관련된 부분을 찾아주세요",
            "이 내용을 읽고 연관성 있는 부분을 찾아보세요",
            "문장을 참고하여 관련된 부분을 검색하세요",
            "이 텍스트와 연관성이 높은 단락을 찾아주세요",
        ]

    def generate(self, split: str):
        dataset = load_dataset("iknow-lab/kowow", split=split)

        for item in dataset:
            text = item["context"]
            pos = item["postive_passage"]
            neg = item["negative_passage"]
            instruction = random.choice(self.instructions)

            yield {
                "instruction": instruction,
                "input": text,
                "positives": pos,
                "negatives": neg,
            }
