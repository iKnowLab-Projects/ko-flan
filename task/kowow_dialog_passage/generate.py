from task.base import BaseGenerator
from dataclasses import dataclass
from datasets import load_dataset
import random
from typing import List
import json


class KOWOWDIALOGPASSAGEGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.instructions = [
            "이 대화를 읽고 대화의 주제와 관련된 문장을 작성하시오.",
            "대화 내용을 분석하고 주요 토픽에 대한 문장을 작성하세요.",
            "대화를 듣고 중심 주제를 기반으로 한 문장을 생성하시오.",
            "주어진 대화를 이해하고 관련된 주제의 문장을 만드세요.",
            "대화의 내용을 토대로 중요한 주제에 대한 문장을 작성하십시오.",
            "대화를 분석하고 해당 주제에 대한 문장을 작성하세요.",
            "주어진 대화를 분석하고 주제에 맞는 문장을 만드세요.",
            "대화의 주제를 파악하고 관련된 문장을 작성하십시오.",
            "대화 내용을 듣고 주제와 관련된 문장을 생성하십시오.",
            "이 대화를 읽고 주제에 맞는 문장을 작성해 보세요.",
            "대화를 분석하고 주제와 관련된 문장을 작성하시오.",
            "주어진 대화의 주제를 이해하고 문장을 생성하세요.",
            "대화 내용을 기반으로 주제와 관련된 문장을 작성하십시오.",
            "이 대화를 읽고 주제와 연관된 문장을 작성하세요.",
            "대화의 내용을 분석하고 주제에 대한 문장을 작성하시오.",
            "대화를 듣고 주제와 관련된 문장을 생성하세요.",
            "주어진 대화의 내용을 분석하고 문장을 작성하십시오.",
            "대화의 주제를 파악하고 관련된 문장을 생성하시오.",
            "대화 내용을 읽고 주제에 맞는 문장을 작성하십시오.",
            "대화를 분석하고 주제와 관련된 문장을 작성해 보세요.",
            "이 대화를 분석하고 주제와 연관된 문장을 작성하시오.",
            "대화의 내용을 이해하고 주제에 대한 문장을 생성하세요.",
            "대화 내용을 기반으로 주제와 관련된 문장을 작성하세요.",
            "이 대화를 읽고 주제에 따른 문장을 작성하십시오.",
            "대화를 분석하고 주제에 맞는 문장을 생성하시오.",
            "주어진 대화를 읽고 관련된 주제의 문장을 만드세요.",
            "대화의 주제를 이해하고 관련된 문장을 작성하세요.",
            "대화 내용을 분석하고 주제에 맞는 문장을 생성하십시오.",
            "이 대화를 듣고 주제와 연관된 문장을 작성하시오.",
            "대화를 읽고 주제와 관련된 문장을 작성해 보세요."
        ]

    def generate(self, split: str):
        dataset = load_dataset('iknow-lab/kowow_dialog',split=split)

        for item in dataset:
            text = item["context"]
            pos = item['postive_passage']
            neg = item['negative_passage']
            instruction = random.choice(self.instructions)

            yield {
                "instruction": instruction,
                "input": text,
                "positives": pos,
                "negatives": neg,
            }
