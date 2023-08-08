from task.base import BaseGenerator

from datasets import load_dataset
import random


class TlTextEntailmentGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.positives = ["예", "그렇습니다", "맞습니다", "네"]
        self.negatives = ["아니요", "아닙니다", "그렇지 않습니다", "틀립니다"]
        

    def generate(self, split: str):
        dataset = load_dataset("iknow-lab/TL_text_entailment", split="train", use_auth_token=True).shuffle(seed=42)
        dataset = dataset.train_test_split(test_size=0.1)[split]

        for item in dataset:
            
            context = item["paragraphs"][0]["context"]

            all_questions = [qas["question"] for paragraph in item["paragraphs"] for qas in paragraph["qas"]]

            question = random.choice(all_questions)

            for paragraph in item["paragraphs"]:
                for qas in paragraph["qas"]:
                    if qas["question"] == question and qas["answers"]["text"] == "Yes":
                        pos, neg = self.positives, self.negatives
                    elif qas["question"] == question and qas["answers"]["text"] == "No":
                        neg, pos = self.positives, self.negatives
                    else:
                        continue
                    
            yield {
                "instruction": question,
                "input": context,
                "positives": pos,
                "negatives": neg,
            }
                                                                