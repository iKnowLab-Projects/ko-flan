from task.base import BaseGenerator
from dataclasses import dataclass
from datasets import load_dataset
import random
from typing import List
import json


class KO_RELATION_RELATIONGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.instructions = [
            '주어진 문장에서 A 단어와 B 단어 사이의 연결성을 설명하시오',
            'A 단어와 B 단어는 어떻게 연관되어 있나요?',
            'A와 B의 관계에 대하여 서술하시오.',
            'A와 B 사이에 나타나는 관계의 특징은 무엇인가요?',
            '주어진 문맥에서 A와 B는 어떠한 관계를 가지고 있습니까',
            'A 단어가 B 단어와 어떤 연관성을 보이는지 분석하시오.',
            'A와 B의 상호작용이나 관계를 근거와 함께 설명하시오.',
            'A와 B 사이의 관계를 중심으로 주어진 문장을 해석하시오.',
            '주어진 문장을 통해 A와 B의 관계를 어떻게 이해하셨나요?',
            'A와 B의 연결고리에 대한 당신의 생각은 무엇인가요?',
            'A와 B 사이의 관계적 특성에 대해 논하시오.',
            'A 단어와 B 단어의 상호 관련성에 대하여 자세히 서술하시오.',
            'A와 B가 어떻게 서로에게 영향을 미치는지 논하시오.',
            '주어진 내용에서 A와 B 사이의 동기나 연결 요소를 찾아 설명하시오.',
            'A와 B의 관계에 있어 주요한 포인트는 무엇이라고 생각하시나요?',
            'A와 B 사이의 연관성을 주어진 문장의 맥락에서 파악하시오.',
            'A 단어와 B 단어가 주어진 문장에서 어떻게 연결되어 있는지 분석하시오.',
            'A와 B의 관계를 주어진 문장의 다른 요소들과 연계하여 설명하시오.',
            'A와 B 사이의 연결성을 강조하는 주어진 문장의 부분은 무엇인가요?',
            'A와 B의 관계를 통해 주어진 문장의 전반적인 흐름을 파악하시오.',
            'A 단어와 B 단어의 관계를 기반으로 주어진 문장의 중심 아이디어를 찾아내시오.',
            '주어진 문장에서 A와 B의 관계의 복잡성을 분석하시오.',
            'A와 B 사이의 관계에 대한 주어진 문장의 특별한 점은 무엇인가요?',
            'A와 B의 관계에 대해 어떤 인사이트를 얻었나요?',
            'A와 B 사이의 관계를 주어진 문장의 구조를 통해 이해하시오.',
            '주어진 문장의 내용을 바탕으로 A와 B 사이의 중요한 연관성을 파악하시오.',
        ]

        self.negative_sample = [
            ["속성관계", "특징관계", "성질관계"],
            ["동치관계", "같음관계", "일치관계"],
            ["부정적관계", "나쁜관계", "안좋은관계"],
            ["원인관계", "인과관계", "유발관계"],
            ["자식관계", "후손관계", "하위관계"],
            ["동료관계", "협업관계", "긍정적관계"],
            ["구성원관계", "멤버관계", "소속관계"],
            ["적대적관계", "상반된관계", "적적관계"],
            ["부모관계", "선조관계", "상위관계"],
            ["부분관계", "하위구성관계", "일부관계"],
            ["장소관계", "위치관계", "공간관계"],
            ["친척관계", "혈연관계", "가족관계"],
            ["형제자매관계", "형동생관계"],
            ["배우자관계", "결혼관계", "아내남편관계"],
            ["시간관계", "기간관계", "시점관계"],
            ["예정관계", "미래관계", "계획관계"],
            ["용도관계", "사용관계", "활용관계"],
            ["관계없음", "연관없음", "비관계"]
        ]

    def generate(self, split: str):

        dataset = load_dataset("iknow-lab/korean_relation", split=split)

        for item in dataset:
            text = 'A :' + item["subj_word"] + " " + "B :" + \
                item["obj_word"] + " " + item["sentence"]
            pos = item['relation']
            neg = random.choice(
                list(filter(lambda x: x != item['relation'], self.negative_sample)))
            instruction = random.choice(self.instructions)

            yield {
                "instruction": instruction,
                "input": text,
                "positives": pos,
                "negatives": neg,
            }
