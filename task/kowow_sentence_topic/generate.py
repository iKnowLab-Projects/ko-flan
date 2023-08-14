from task.base import BaseGenerator
from dataclasses import dataclass
from datasets import load_dataset
import random
from typing import List
import json


class KOWOWSENTENCETOPICGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.instructions = [
            "주어진 문장을 분석하여 관련 주제를 찾으세요.",
            "문장의 내용을 읽고 중심 주제를 결정하십시오.",
            "문장의 내용을 기반으로 적절한 토픽을 선정하시오.",
            "주어진 텍스트를 읽고 관련된 주제를 작성하십시오.",
            "문장에서 주된 테마를 식별하시오.",
            "문장을 듣고 연관된 주제를 찾으세요.",
            "주어진 문장의 내용을 분석하여 중요한 주제를 파악하시오.",
            "문장의 주제를 읽고 파악하십시오.",
            "문장을 기반으로 핵심 주제를 찾으세요.",
            "주어진 문장을 듣고 주요 토픽을 결정하십시오.",
            "문장의 내용을 분석하고 주제를 선정하시오.",
            "문장을 통해 주요 주제를 이해하고 작성하시오.",
            "문장의 내용을 기반으로 주요 테마를 식별하십시오.",
            "주어진 텍스트에서 중심 주제를 찾으세요.",
            "문장을 분석하고 관련된 토픽을 결정하십시오.",
            "주어진 문장의 주제를 읽고 이해하시오.",
            "문장에서 주요 테마를 찾아 작성하십시오.",
            "문장을 듣고 중심이 되는 주제를 결정하시오.",
            "문장의 내용을 기반으로 적절한 주제를 선정하세요.",
            "주어진 텍스트를 읽고 핵심 토픽을 찾으십시오.",
            "문장의 주제를 분석하고 작성하십시오.",
            "문장의 내용을 듣고 중요한 주제를 파악하시오.",
            "주어진 문장을 분석하고 주제를 결정하십시오.",
            "문장의 테마를 읽고 식별하시오.",
            "문장의 내용을 기반으로 주요 주제를 찾으세요.",
            "주어진 문장을 듣고 관련된 토픽을 작성하십시오.",
            "문장의 내용을 분석하고 중심 주제를 찾으시오.",
            "문장을 듣고 주요 테마를 식별하십시오.",
            "주어진 텍스트를 분석하고 관련된 주제를 찾으세요.",
            "문장을 분석하고 알맞은 토픽 주제를 작성하시오.",
        ]

    def generate(self, split: str):
        dataset = load_dataset("iknow-lab/kowow", split=split)

        for item in dataset:
            text = item["context"]
            pos = item["postive_topic"][0]
            neg = item["negatvie_topic"][0]
            instruction = random.choice(self.instructions)

            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": [neg],
            }
