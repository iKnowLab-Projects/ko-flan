from task.base import BaseGenerator
from datasets import load_dataset
import random
import pandas as pd
import numpy as np

class KLUE_MRCGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()


    def generate(self, split: str):
        if split == 'test':
            split = 'validation'

        dataset = load_dataset("klue",'mrc', split=split)
        split = 'test'
        df = pd.DataFrame(dataset)
        df['news_category']=df['news_category'].fillna(0)
        for i in range(0, df.shape[0]):
            row = df.iloc[i]
            text = row["context"]
            instruction = row['question']
            postive = row['answers']['text'][0]
            same_title_df = df.loc[(df['title']==row['title']) &  (df['answers'] != row['answers']),'answers']
            # 같은 title이 없는 경우 같은 category에 있는 negative 데이터 추출
            if(same_title_df.isnull().all()):
               negative =  np.random.choice(df.loc[( df['news_category']== row['news_category']) &(df['answers'] != row['answers']),'answers'])['text'][0]
            #같은 title이 있는 경우 같은 ttile에 있는 negative 데이터 추출
            else:
               negative  =  np.random.choice(same_title_df)['text'][0]

            yield {
                "instruction": instruction,
                "input": text,
                "positives":postive,
                "negatives":negative,
            }