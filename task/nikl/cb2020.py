from task.base import BaseGenerator
from datasets import load_dataset
import random


class CB2020Generator(BaseGenerator):
    # 국립국어원 추론_확신성 분석 말뭉치 2020
    def __init__(self) -> None:
        super().__init__()

        self.instructions = """
이 문장은 "{Proposition}"에 대한 문장과 글 사이의 관련성을 암시합니다.
"{Proposition}"이라는 문장과 글 사이에는 무슨 관계가 있을지 추론해 보세요.
이 문장은 "{Proposition}"과 글 사이의 연관성을 어떻게 이해할 수 있을지 알려줍니다.
"{Proposition}" 문장과 글 사이에서 어떤 상호작용이 있을 것으로 보입니다.
이 문장은 "{Proposition}"과 글 간의 상관관계를 유추할 수 있도록 돕습니다.
"{Proposition}"과 글 사이에서 어떤 맥락적인 연결이 있을지 짐작할 수 있습니다.
이 문장은 "{Proposition}"과 글의 관계를 추론할 수 있게 해줍니다.
"{Proposition}"과 글 사이에서 어떤 내용적인 관련성을 찾아볼 수 있을 것입니다.
이 문장은 "{Proposition}"과 글 사이의 유사점이나 차이점을 이해하는 데 도움이 됩니다.
"{Proposition}"과 글 사이에서 어떤 연계가 있을지 상상해 보세요.
""".strip().split(
            "\n"
        )

        self.labels = {
            "Entailment": ["수반", "수반관계", "동반관계", "동시에 일어나다", "함께하는 관계", "포함하는 문장"],
            "Contradict": [
                "모순",
                "모순관계",
                "상충관계",
                "대립하는 관계",
                "모순되는 관계",
                "반대되는 관계",
                "충돌하는 관계",
            ],
        }

    def generate(self, split: str):
        dataset = load_dataset("iknow-lab/nikl_cb_2020_v1.1", split=split)

        for item in dataset:
            instruction = random.choice(self.instructions).format(
                Proposition=item["Proposition"]
            )
            label = item["class_Restrict"]
            text = item["Discourse"]

            if label == "Entailment":
                neg_label = "Contradict"
            elif label == "Contradict":
                neg_label = "Contradict"
            else:
                raise ValueError(f"{label}은 알 수 없는 값이에요")

            yield {
                "instruction": instruction,
                "input": text,
                "positives": self.labels[label],
                "negatives": self.labels[neg_label],
            }
