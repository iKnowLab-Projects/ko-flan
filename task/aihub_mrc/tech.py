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


def handle_tech_mrc_item(item):
    outputs = []

    for context_dict in item["dataset"]["context_info"]:
        context = html4text(context_dict["context"])
        if len(context) == 0:
            continue

        questions = context_dict["qas"]
        all_answers = set([qa["answer"] for qa in questions])

        for qa in questions:
            type = qa["answer_type"]
            pos = [qa["answer"]]

            if type == "다지선다형":
                neg = qa["wrong_answer"]
            if type == "Yes/No 단문형":
                pos = ["예", "맞습니다", "그렇습니다", "일치합니다", "정답입니다."]
                neg = ["아니오", "아닙니다", "틀렸습니다", "정답이 아닙니다"]
            else:
                neg = list(all_answers - set(pos))

            questions = [
                x for x in [qa["question-1"], qa["question-2"]] if x is not None
            ]

            if len(questions) > 0:
                outputs.append(
                    {
                        "instruction": random.choice(questions),
                        "input": context,
                        "positives": pos,
                        "negatives": neg,
                    }
                )

    return outputs


class AIHubTechMRCGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, split: str):
        dataset = load_dataset("iknow-lab/aihub_mrc_tech", split=split)
        data = [json.loads(json_str) for json_str in dataset["content"]]

        for item in data:
            yield from handle_tech_mrc_item(item)
