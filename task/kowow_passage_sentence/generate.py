from task.base import BaseGenerator
from dataclasses import dataclass
from datasets import load_dataset
import random
from typing import List
import json


class KOWOWPASSAGESENTENCEGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.instructions = [
        "A 문장과 가장 관련 있는 문장을 B 단락에서 찾아보세요.",
        "B 단락에서 A 문장과 유사한 내용의 문장을 선택하세요.",
        "A 문장과 같은 주제를 다루는 문장이 B 단락에 어떤 것인지 확인하세요.",
        "A 문장과 연관성이 높은 문장을 B 단락에서 찾아보세요.",
        "B 단락 내에서 A 문장과 동일한 맥락을 가진 문장을 선택하세요.",
        "A 문장과 가장 밀접한 관련을 가진 문장이 B 단락의 어디에 있는지 확인하세요.",
        "B 단락에서 A 문장과 동일한 특징을 가진 문장을 찾아보세요.",
        "A 문장과 동일한 주제나 내용을 다루는 문장을 B 단락에서 찾아보세요.",
        "B 단락 내에서 A 문장과 연결되는 문장을 찾아보세요.",
        "A 문장과 관련된 내용을 가진 문장이 B 단락의 어디에 있는지 확인하세요.",
        "B 단락에서 A 문장과 가장 유사한 문장을 선택하세요.",
        "A 문장과 같은 아이디어나 내용을 가진 문장을 B 단락에서 찾아보세요.",
        "B 단락 내에서 A 문장과 동일한 주제를 가진 문장을 선택하세요.",
        "A 문장과 연결되는 아이디어나 내용을 가진 문장을 B 단락에서 찾아보세요.",
        "B 단락에서 A 문장과 관련된 특징을 가진 문장을 찾아보세요.",
        "A 문장과 비슷한 주제나 내용을 다루는 문장이 B 단락에 어떤 것인지 확인하세요.",
        "B 단락 내에서 A 문장과 동일한 정보나 아이디어를 가진 문장을 찾아보세요.",
        "A 문장과 같은 내용을 가진 문장을 B 단락에서 찾아보세요.",
        "B 단락에서 A 문장과 연관된 아이디어나 주제를 가진 문장을 찾아보세요.",
        "A 문장과 연결된 특징을 가진 문장이 B 단락의 어디에 있는지 확인하세요.",
        "B 단락 내에서 A 문장과 유사한 특징이나 주제를 가진 문장을 선택하세요.",
        "A 문장과 동일한 아이디어나 내용을 다루는 문장을 B 단락에서 찾아보세요.",
        "B 단락에서 A 문장과 관련성이 높은 문장을 찾아보세요.",
        "A 문장과 같은 정보나 주제를 가진 문장이 B 단락에 어떤 것인지 확인하세요.",
        "B 단락 내에서 A 문장과 동일한 특징이나 내용을 가진 문장을 찾아보세요.",
        "A 문장과 관련된 아이디어나 내용을 가진 문장을 B 단락에서 찾아보세요."

        ]

    def generate(self, split: str):

        dataset = load_dataset("iknow-lab/kowow", split=split)

        for item in dataset:
            text = 'A :'+ item['context'] +" "+ 'B :'+ item["postive_passage"][0]
            pos = item['postive_sentence'][0]
            neg = item['negative_sentence'][0]
            instruction = random.choice(self.instructions)


            yield {
                "instruction": instruction,
                "input": text,
                "positives": pos,
                "negatives": neg,
            }
