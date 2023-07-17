from task.base import BaseGenerator

from datasets import load_dataset
import random
import pandas as pd

class KLUE_NLIGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

        self.instructions = [
    "제시된 문장과 일관성 있는 내용을 가진 문장을 작성하세요.",
    "주어진 문장에 맞춰 연결될 수 있는 문장을 작성해 주세요.",
    "아래의 문장이 주제인 문장을 만들어 주세요.",
    "주어진 문장을 기반으로 이어질 수 있는 내용을 가진 문장을 작성해 주세요.",
    "제시된 문장과 같은 맥락을 이어가는 문장을 만들어 보세요.",
    "제시된 문장을 계속하여 내용을 더 추가해보세요.",
    "이어질 수 있는 문장을 작성하여 주어진 문장을 확장해 주세요.",
    "주어진 문장을 이어받아 동일한 주제에 대해 작성해 주세요.",
    "아래 문장과 잘 연결되는 문장을 작성해 주세요.",
    "주어진 문장이 주도하는 논점을 기반으로 문장을 작성해 주세요.",
    "다음 문장이 자연스럽게 이어질 수 있도록 문장을 작성해 주세요.",
    "제시된 문장과 함께 사용될 수 있는 문장을 만들어 보세요.",
    "주어진 문장을 따라 이어갈 수 있는 문장을 작성해 보세요.",
    "제시된 문장에 적절히 이어지는 문장을 만들어 주세요.",
    "주어진 문장에 대한 연속된 내용을 가진 문장을 만들어 보세요.",
    "제시된 문장에 적합하게 이어질 수 있는 문장을 작성해 보세요.",
    "주어진 문장을 통해 이어질 수 있는 새로운 문장을 만들어 주세요.",
    "제시된 문장과 잘 어울리는 문장을 작성해 보세요.",
    "주어진 문장을 바탕으로 문맥을 유지하는 문장을 만들어 보세요.",
    "제시된 문장에 맞게 이어지는 문장을 작성해 주세요.",
    "주어진 문장에 맞는 후속 문장을 작성해 주세요.",
    "제시된 문장을 기반으로 확장된 내용을 가진 문장을 만들어 보세요.",
    "주어진 문장과 동일한 주제로 이어지는 문장을 작성해 보세요.",
    "제시된 문장에 맞는 내용을 가진 문장을 작성해 보세요.",
    "주어진 문장에 따라 자연스럽게 이어지는 문장을 만들어 보세요.",
    "제시된 문장과 이어질 수 있는 문장을 작성해 주세요.",
    "주어진 문장을 이어갈 수 있는 방식으로 문장을 작성해 보세요.",
    "제시된 문장에 기반하여 연결된 문장을 만들어 보세요.",
    "주어진 문장과 동일한 주제로 문장을 만들어 보세요.",
    "제시된 문장과 자연스럽게 이어지는 문장을 작성해 보세요."
]


    def generate(self, split: str):
        if split == 'test':
            split = 'validation'

        dataset = load_dataset("klue",'nli', split=split)
        split ='test'
        df = pd.DataFrame(dataset)
        drop_df = df.drop(df[df['label']==1].index)
        new_df = pd.DataFrame()
        new_df['input'] = drop_df['premise']
        new_df.drop_duplicates(inplace = True)
        new_df['postive'] =None
        new_df['negative']=None
        for i in range(0, new_df.shape[0]):
          row = new_df.iloc[i]
          instruction = random.choice(self.instructions)
          text = row['input']
          new_df['postive'].iloc[i] = df.loc[(df['premise']==row['input'])&(df['label']==0),'hypothesis']
          new_df['negative'].iloc[i] = df.loc[(df['premise']==row['input'])&(df['label']==2),'hypothesis']
          if(len(new_df['postive'].iloc[i])>0):
             postive = new_df['postive'].iloc[i][0]
          else:
             postive = 0

          if(len(new_df['negative'].iloc[i])>0):
             negative = new_df['negative'].iloc[i][0]
          else:
             negative = 0
          if((postive == 0)|(negative == 0)):
              pass


          yield {
                "instruction": instruction,
                "input": text,
                "positives": postive,
                "negatives": negative,
            }

