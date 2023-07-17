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
        for i in range(0, df.shape[0]):
            row = df.iloc[i]
            text = row["context"]
            instruction = row['question']
            postive = row['answers']['text'][0]
            negative =  np.random.choice(df.loc[(((df['title']==row['title']) |( df['news_category'] == row['news_category'])) & df['answers'] != row['answers']),'answers'])['text'][0]
            yield {
                "instruction": instruction,
                "input": text,
                "positives":postive,
                "negatives":negative,
            }