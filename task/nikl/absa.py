from task.base import BaseGenerator
from datasets import load_dataset
import random


class ABSAGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.labels = {
            "positive": ["긍정", "좋아한다", "선호", "좋아요"],
            "negative": ["부정", "싫어한다", "불호", "별로에요"],
            "neutral": ["중립", "감정이 없다", "알 수 없다", "몰라요", "모름", "감정이 느껴지지 않습니다", "무감정", "중립적"]
        }
        self.label_keys = set(self.labels.keys())

        self.instructions_normal = """
            글을 읽고 {target}에 대한 감정을 알려주세요.
            {target}에 대한 감정을 문장에서 파악할 수 있을까요?
            게시글을 읽고 {target}에 대한 감정을 추측해볼 수 있을까요?
            글 속에서 {target}과 관련하여 어떤 감정이 묻어나는지 알려주세요.
            내용을 분석하여 {target}에 대한 감정을 파악할 수 있을까요?
            {target}과 관련하여 글에 어떤 정서가 녹아있는지 생각해보세요.
            글을 읽고 {target}에 대한 감정을 읽어낼 수 있을까요?
            문단 속에서 {target}에 대한 감정을 찾아낼 수 있을까요?
            {target}과 관련된 문장에서 어떤 감정이 느껴지는지 설명해주세요.
            글에서 느껴지는 감정 중에서 {target}에 대한 것을 찾아낼 수 있을까요?
        """.strip().split('\n')
        self.instructions_normal = [x.strip() for x in self.instructions_normal]

        self.instructions_property = """
            주어진 문장을 읽고 {target}의 {property}에 대한 감정을 알려주세요.
            {target}의 {property}에 대한 감정을 문장에서 파악할 수 있을까요?
            글을 읽고 {target}의 {property}에 대한 감정을 추측해볼 수 있을까요?
            글 속에서 {target}의 {property}과 관련하여 어떤 감정이 묻어나는지 알려주세요.
            글의 문장을 분석하여 {target}의 {property}에 대한 감정을 파악할 수 있을까요?
            {target}의 {property}과 관련하여 글에 어떤 정서가 녹아있는지 생각해보세요.
            글을 읽고 {target}의 {property}에 대한 감정을 읽어낼 수 있을까요?
            글 속 문장 중에서 {target}의 {property}에 대한 감정을 찾아낼 수 있을까요?
            {target}의 {property}와 관련된 문장에서 어떤 감정이 느껴지는지 설명해주세요.
            글에서 느껴지는 감정 중에서 {target}의 {property}에 대한 것을 찾아낼 수 있을까요?
        """.strip().split('\n')
        self.instructions_property = [x.strip() for x in self.instructions_property]

    def generate(self, split: str):
        dataset = load_dataset("iknow-lab/nikl_absa_2021_v1.1", split="train", use_auth_token=True).shuffle(seed=42)
        dataset = dataset.train_test_split(test_size=0.1)[split]

        for item in dataset:
            sents = [s["sentence_form"] for s in item["sentence"]]
            text = " ".join(sents)

            for opinion in item["opinions"]:
                category = opinion["category"]
                if category == "OUT OF SCOPE":
                    continue

                label = opinion["opinion polarity"]
                if label == "conflict":
                    continue

                target, property = category.split("#", 1)

                # 무작위로 instance를 고른다
                if property == "일반":
                    instruction = random.choice(self.instructions_normal).format(target=target)
                else:
                    instruction = random.choice(self.instructions_property).format(target=target, property=property)

                pos = self.labels[label]
                neg_labels = self.label_keys - set([label])
                neg = [l for neg_label in neg_labels for l in self.labels[neg_label]]

                yield {
                    "instruction": instruction,
                    "input": text,
                    "positives": pos,
                    "negatives": neg,
                }

