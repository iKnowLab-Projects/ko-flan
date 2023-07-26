from task.base import BaseGenerator
from datasets import load_dataset
import random


class KobestWicGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.positives = ["맞습니다", "예", "네", "그렇습니다", "의미가 동일합니다", "동일한 의미입니다"]
        self.negatives = ["아니요", "아닙니다", "그렇지 않습니다", "의미가 다릅니다", "동일하지 않습니다"]
        self.instructions = [
            "주어진 두 문장에서 단어 {word}은(는) 동일한 의미로 사용되었나요?",
            "주어진 두 문장에서 단어 {word}은(는) 동일한 의미로 사용되었다",
            "주어진 두 문장에서 단어 {word}은(는) 같은 의미로 사용되었다",
            "주어진 두 문장에서 단어 {word}은(는) 같은 의미를 지닌다",
            "주어진 두 문장에서 단어 {word}은(는) 같은 의미를 지니고 있나요?",
            "단어 {word}은(는) 두 문장에서 동일한 의미이다",
            "단어 {word}은(는) 두 문장에서 동일한 의미로 사용되었다",
            "단어 {word}은(는) 두 문장에서 동일한 의미로 사용되었나요?",
            "단어 {word}은(는) 두 문장에서 같은 의미로 사용되었다",
            "단어 {word}은(는) 두 문장에서 같은 의미를 지니고 있나요?",
            "단어 {word}은(는) 두 문장에서 같은 의미를 지닌다",
            "두 문장에서 {word}은(는) 동일한 의미로 사용되고 있다",
        ]

    def generate(self, split: str):

        dataset = load_dataset("skt/kobest_v1", "wic", split=split)

        for item in dataset:
            word = item["word"]
            instruction = random.choice(self.instructions)
            instruction = instruction.format(word=f'{word}')
            text = "문장1: " + item["context_1"] + " 문장2: " + item["context_2"]
            
            if item["label"] == 0: # false 
                pos, neg = self.negatives, self.positives
            else:
                pos, neg = self.positives, self.negatives
                
            yield {
                "instruction": instruction,
                "input": text,
                "positives": pos,
                "negatives": neg,
            }
