from task.base import BaseGenerator

from datasets import load_dataset
import random


class KlueYnatGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.labelToTextDict={
            0: 'IT과학', 1: '경제', 2: '사회',
            3: '생활문화', 4: '세계', 5: '스포츠',
            6: '정치'}
        self.instructions = ['문장을 읽고 주제를 분류하세요.',
            '다음 문장을 해석하고 적절한 카테고리로 분류해보세요.',
            '이 문장과 관련된 분야를 파악해보세요.',
            '다음 문장을 이해하고 어떠한 주제를 떠올릴 수 있는지 생각해보세요.',
            '주어진 텍스트를 읽고 해당 문장이 속한 분야를 정해보세요',
            '다음 문장을 분석하여 어떠한 주제와 관련이 있는지 판단해보세요.',
            '주어진 문장을 읽고 어떤 주제와 일치하는지 결정해보세요.',
            '다음 단락을 읽고 어떤 주제에 대해 이야기하는지 파악해보세요.',
            '주어진 문장을 해석하고 이 문장과 관련된 분야나 주제를 찾아보세요.',
            '주어진 텍스트를 분석하여 주제를 도출해보세요.',
            '다음 문장을 해석하고 어떤 분야의 내용인지 유추해보세요.',
            '주어진 문장을 읽고 어떤 분야나 주제와 연관되어 있는지 파악해보세요.',
            '다음 문장을 해석하고 이와 관련된 주제를 분류해보세요.',
            '주어진 텍스트를 읽고 이 문장이 속한 분야나 카테고리를 판단해보세요.',
            '다음 문장을 이해하고 어떤 주제와 관련되어 있는지 생각해보세요.',
            '주어진 문장을 읽고 이와 관련된 분야나 주제를 결정해보세요.',
            '주어진 텍스트를 분석하여 이 문장이 속한 주제를 파악해보세요.',
            '다음 문장을 읽고 주요 내용을 정리하여 어떤 주제와 관련이 있는지 찾아보세요.',
            '주어진 문장을 해석하고 이와 관련된 분야를 찾아보세요.',
            '주어진 텍스트를 읽고 주제를 추론해보세요.',
            '다음 문장을 분석하여 이 문장이 어떤 주제와 관련이 있는지 판단해보세요.',
            '주어진 문장을 해석하고 어떤 분야나 주제와 연관되어 있는지 판단해보세요.',
            '주어진 텍스트를 읽고 이 문장이 어떤 분야에 속하는지 유추해보세요.',
            '다음 문장을 이해하고 어떤 주제와 관련이 있는지 생각해보세요.',
            '주어진 텍스트를 분석하여 이 문장이 속한 주제를 찾아보세요.',
            '다음 문장을 읽고 주요 내용을 정리하여 어떤 주제와 관련이 있는지 결정해보세요.']

    def generate(self, split: str):
        if split == 'test':
            split = 'validation'
        dataset = load_dataset('yangwang825/klue-ynat', split=split)
        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            text = item['text']
            pos = self.labelToTextDict[item['label']]
            neg = [x for x in self.labelToTextDict.values() if x != pos]
            
            yield {
                "instruction": instruction,
                "input": text,
                "positives": pos,
                "negatives": neg,
            }