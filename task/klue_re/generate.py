from task.base import BaseGenerator

from datasets import load_dataset
import random


class KlueReGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.word1 = ''
        self.word2 = ''
        self.labelToTextDict={
            'no_relation':'관계가 없습니다.',
            'per:alternate_names':'다른 이름',
            'per:children': '자식',
            'per:colleagues': '동료',
            'per:date_of_birth': '태어난 날짜',
            'per:date_of_death': '죽은 날짜',
            'per:employee_of': '소속 조직',
            'per:origin': '출신지',
            'per:other_family': '가족 구성원',
            'per:parents': '부모',
            'per:place_of_birth': '태어난 곳',
            'per:place_of_death': '죽은 곳',
            # 'per:place_of_residence': ''
            'per:product': '작품',
            'per:religion': '종교',
            'per:schools_attended': '다닌 학교',
            'per:siblings': '형제 자매',
            'per:spouse': '배우자',
            # 'per:title': ''
            'org:alternate_names': '다른 이름',
            # 'org:dissolved': '',
            'org:founded': '만들어진 년도',
            'org:founded_by': '설립자',
            'org:member_of': '소속 기관',
            'org:members': '소속 기관',
            'org:number_of_employees/members': '소속된 숫자',
            'org:place_of_headquarters': '소속된 위치',
            'org:political/religious_affiliation': '소속 이념',
            'org:product': '생산물',
            'org:top_members/employees': '조직의 대표자 또는 구성원',
            }
        self.instructions = ['문장을 읽은 후 단어들의 관계를 찾아보세요.',
            '다음 문장을 해석하고 다음 단어들을 적절한 관계로 분류해보세요.', 
            '문장 속 정보를 이용해 단어의 관계를 파악해보세요.',
            '텍스트를 이해하고 다음 단어들로부터 어떠한 관계를 떠올릴 수 있는지 생각해보세요.',
            '주어진 텍스트를 읽고 단어가 속한 관계를 정해보세요',
            '문장을 분석하여 단어가 어떠한 관계로 관련이 있는지 판단해보세요.',
            '주어진 문장을 읽고 단어들이 어떤 주제로 관련됐는지 결정해보세요.',
            '다음 문장을 읽고 단어들이 어떤 관계로 연관돼 있는지 파악해보세요.',
            '주어진 문장을 해석하고 단어들이 서로 어떻게 관련됐는지 찾아보세요.',
            '주어진 텍스트를 분석하여 단어 간 연관 관계를 도출해보세요.',
            '다음 문장을 해석하고 단어들이 어떤 관계인지 유추해보세요.',
            '주어진 문장을 읽고 단어들을 어떤 분야나 주제로 연관지을 수 있는지 파악해보세요.',
            '다음 문장을 해석하고 단어들의 관련성을 파악해보세요.',
            '입력된 텍스트를 읽고 단어들의 관계를 판단하고 적절하게 분류해보세요.',
            '다음 문장을 이해하고 단어들이 어떤 주제와 관련되어 있는지 생각해보세요.',
            '주어진 문장을 읽고 각각의 단어들의 관계를 결정해보세요.',
            '주어진 텍스트를 분석하여 단어가 속한 관계를 파악해보세요.',
            '다음 문장을 읽고 내용을 파악하여 단어가 서로 어떤 주제로 관련이 있는지 찾아보세요.',
            '주어진 문장을 해석하고 단어의 관련성을 찾아보세요.',
            '주어진 텍스트를 읽고 단어가 갖는 관계를 추론해보세요.',
            '다음 문장을 분석하여 단어들이 어떻게 관련이 있는지 판단해보세요.',
            '주어진 문장을 해석하고 단어가 어떤 분야나 주제로 연관되어 있는지 판단해보세요.',
            '주어진 텍스트를 읽고 단어가 어떤 관계를 갖는지 유추해보세요.',
            '다음 문장을 이해하고 단어가 어떤 주제와 관련성이 있는지 생각해보세요.',
            '주어진 텍스트를 분석하여 단어로 부터 연관성을 찾아보세요.',
            '다음 문장을 읽고 단어들이 어떻게 관련이 있는지 결정해보세요.']

    def generate(self, split: str):
        if split == 'test':
            split = 'validation'
        dataset = load_dataset('LeverageX/klue-re', split=split)
        self.labelToTextDict = {k:v+' 관계입니다.' if k!='no_relation' else v for k,v in self.labelToTextDict.items()}
        for item in dataset:
            # 무작위로 instance를 고른다
            instruction = random.choice(self.instructions)
            self.word1 = item['object_entity']['word']
            self.word2 = item['subject_entity']['word']
            text = item['sentence']
            try: 
                pos = self.labelToTextDict[item['label']]
            except KeyError: 
                continue
            
            neg = [x for x in self.labelToTextDict.values() if x != pos]
            yield {
                "instruction": instruction,
                "input": f'문장: {text}, 단어 1: {self.word1}, 단어 2:{self.word2}',
                "positives": [pos],
                "negatives": neg
            }