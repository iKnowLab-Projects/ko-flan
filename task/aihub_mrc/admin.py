from bs4 import BeautifulSoup
from task.base import BaseGenerator
from datasets import load_dataset
from itertools import chain
from typing import List
import json
import random


def check_label(negative: str, positives: List[str]):
    for pos in positives:
        if negative in pos or pos in negative:
            return False

    return True

def html4text(html):
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text()

def handle_mrc_item(item):
    POS = ["예", "맞습니다", "그렇습니다", "일치합니다", "정답입니다."]
    NEG = ["아니오", "아닙니다", "틀렸습니다", "정답이 아닙니다"]
    UNK = ["모릅니다", "알 수 없습니다", "정답이 없습니다"]

    outputs = []
    paragraphs = [p for dataset in item["data"] for p in dataset["paragraphs"]]
    for paragraph in paragraphs:
        context = html4text(paragraph["context"])
        questions = paragraph["qas"]
        all_answers = set([qa["answers"]["text"] for qa in questions])

        for qa in questions:
            pos = qa["answers"]["text"]
            impossible = qa.get("is_impossible", False)

            if impossible:
                neg = [pos] # 대답 불가능한 질문은 기존 positive 정답을 negative로 준다
                pos = UNK
            elif pos == "Yes":
                pos, neg = POS, NEG
            elif pos == "No":
                pos, neg = NEG, POS
            else:
                pos = [pos]
                neg = list(all_answers - set(pos))

            if len(questions) > 0:
                outputs.append({
                    "instruction": qa["question"],
                    "input": context,
                    "positives": pos,
                    "negatives": neg,
                })

    return outputs

class AIHubAdminMRCGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, split: str):
        dataset = load_dataset("iknow-lab/aihub_mrc_admin", split=split)
        # 일단 tableqa는 배제한다.
        dataset = dataset.filter(lambda x: "tableqa" in x["file"])
        data = [json.loads(json_str) for json_str in dataset["content"]]

        for item in data:
            yield from handle_mrc_item(item)
