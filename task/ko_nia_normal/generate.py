from task.base import BaseGenerator

from datasets import load_dataset
from collections import defaultdict
import random


class KoNiaGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, split: str):
        # dataset = load_dataset("iknow-lab/ko_nia_normal", split="train", use_auth_token=True).shuffle(seed=42)
        # dataset = dataset.train_test_split(test_size=0.1)[split]

        dataset = load_dataset(
            "iknow-lab/ko_nia_normal", split="train", use_auth_token=True
        ).shuffle(seed=42)
        dataset = dataset.train_test_split(test_size=0.1)[split]

        for item in dataset:
            context = item["paragraphs"][0]["context"]

            # print(context)

            all_questions = [
                qas["question"]
                for paragraph in item["paragraphs"]
                for qas in paragraph["qas"]
            ]

            question = random.choice(all_questions)
            neg_list = []
            pos_list = []

            for paragraph in item["paragraphs"]:
                for qas in paragraph["qas"]:
                    if qas["question"] == question:
                        pos_list.append(qas["answers"][0]["text"])
                    else:
                        neg_list.append(qas["answers"][0]["text"])

            if len(pos_list) > 0 and len(neg_list) > 0:
                yield {
                    "instruction": question,
                    "input": context,
                    "positives": pos_list,
                    "negatives": neg_list,
                }
