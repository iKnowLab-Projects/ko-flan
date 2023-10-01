from task.base import BaseGenerator
from datasets import load_dataset
import random
import pandas as pd
import numpy as np


class AULMGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, split: str):
        dataset = load_dataset("heegyu/aulm-0809", split=split) #.train_test_split(0.1)[split]

        for item in dataset:
            conv = item["conversations"]

            if conv[0]['from'] == 'input':
                kg, inst, answer = conv[0]['value'], conv[1]['value'], conv[2]['value']
            else:
                kg, inst, answer = '', conv[0]['value'], conv[1]['value']

            if inst:
                yield {
                    "instruction": inst,
                    "input": kg,
                    "positives": [answer],
                    "negatives": [],
                }
