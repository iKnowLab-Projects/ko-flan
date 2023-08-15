from task.base import BaseGenerator

from datasets import load_dataset
import random


class NIKLexGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.labels = set(["비슷한말", "반대말", "상위어", "하위어"])

        self.instructions = """
            두 개의 용어가 어떻게 연관되어 있는지 설명해주세요.
            이 두 용어는 어떤 관계를 가지고 있나요?
            두 단어 간의 연관성을 알려주세요.
            이 두 용어 사이에 어떤 상호작용이 있는지 설명해주세요.
            두 개의 용어가 어떻게 상호작용하고 있는지 알려주세요.
            두 단어 사이의 관련성을 어떻게 설명하실 건가요?
            이 두 용어는 서로 어떤 방식으로 서로 영향을 주고 받나요?
            두 개의 단어가 함께 사용될 때 어떤 의미가 발생하는지 설명해주세요.
            두 단어 간의 관계를 자세히 알려주세요.
            두 단어 사이에 존재하는 상호관계를 설명해주세요.
            두 단어 사이의 상호작용을 알려주세요.
            두 개의 용어가 어떻게 서로 연결되어 있는지 설명해주세요.
            이 두 용어의 관련성을 어떻게 해석할 수 있을까요?
            두 단어 간의 상관관계를 설명해주세요.
            이 두 용어의 연관성에 대해 자세히 설명해주세요.
            두 단어의 관계에 대해 더 자세히 알려주세요.
        """.strip().split(
            "\n"
        )
        self.input_formats = [
            "A: {A}\nB: {B}",
            "1: {A}\n2: {B}",
            "{A},{B}",
            "단어1: {A}, 단어2: {B}",
            "{A} 그리고 {B}",
        ]
        self.instructions = [x.strip() for x in self.instructions]

    def generate(self, split: str):
        dataset = load_dataset(
            "iknow-lab/niklex", split="train", use_auth_token=True
        ).shuffle(seed=42)
        dataset = dataset.train_test_split(test_size=0.1)[split]

        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            word1, word2 = item["word1"], item["word2"]
            input_format = random.choice(self.input_formats)
            text = input_format.format(A=word1, B=word2)
            label = item["type"]

            if len(text) > 0:
                yield {
                    "instruction": instruction,
                    "input": text,
                    "positives": [label],
                    "negatives": list(self.labels - set([label])),
                }
