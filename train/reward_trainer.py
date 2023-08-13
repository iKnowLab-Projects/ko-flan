from typing import List, Dict
from dataclasses import dataclass
import random
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from .base import BaseTrainer, BaseTrainingArguments, collate_dictlist
from itertools import chain

import torch
import torch.nn.functional as F

import pandas as pd
from pprint import pprint
from datasets import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class RewardModelArguments(BaseTrainingArguments):
    pass

@dataclass
class RewardModelCollator(object):

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    truncation: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = collate_dictlist(features)
        pos = [random.choice(x) for x in features["positives"]]
        neg = [random.choice(x) for x in features["negatives"]]
        prompt = [x + self.tokenizer.sep_token + y for x, y in zip(features["input"], features["instruction"])]
        
        pos_batch = self.tokenizer(
            prompt,
            pos,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )
        neg_batch = self.tokenizer(
            prompt,
            neg,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )

        del features["positives"]
        del features["negatives"]

        return {
            "positives": pos_batch,
            "negatives": neg_batch,
            **features
        }
    
class RewardTrainer(BaseTrainer):
    
    def prepare_dataset(self):
        self.dataset = load_dataset(self.args.dataset)
        
        self.tokenizer.truncation_side = 'left'
        tasks = set(self.dataset['train']["task"] + self.dataset['test']["task"])
        self.task2id = {k: i for i, k in enumerate(tasks)}
        self.id2task = {i: k for i, k in enumerate(tasks)}

        return {
            'train': self.dataset['train'],
            'validation': self.dataset['test'],
        }

    def get_collator(self):
        return RewardModelCollator(
            tokenizer=self.tokenizer,
            max_length=self.args.max_seq_length ,
            pad_to_multiple_of=8,
            padding="max_length",
            return_tensors="pt",
        )
    
    def _shared_step(self, batch):
        pos = self.model(**batch["positives"]).logits#[:, 0]
        neg = self.model(**batch["negatives"]).logits#[:, 0]
        loss = -F.logsigmoid(pos - neg)
        acc = pos > neg
        return {
            "loss": loss, 
            "acc": acc,
            "task": torch.tensor([self.task2id[k] for k in batch["task"]], dtype=torch.int32, device=loss.device)
        }
    
    def training_step(self, batch):
        loss = self._shared_step(batch)["loss"]
        return loss.mean()

    def evaluation_step(self, batch):
        return self._shared_step(batch)


    def collate_evaluation(self, results: List[Dict]):
        losses = torch.stack(results['loss']).reshape(-1).tolist()
        accuracies = torch.stack(results['acc']).reshape(-1).tolist()
        tasks = torch.stack(results['task']).reshape(-1).tolist()
        tasks = [self.id2task[k] for k in tasks]


        df = pd.DataFrame({
            "loss": losses,
            "acc": accuracies,
            "task": tasks
        })
        result = df.groupby("task").mean()
        
        pprint("evaluation result")
        print(result)

        eval_results = {}
        for task, row in result.iterrows():
            loss, acc = row["loss"], row["acc"]
            eval_results[f"{task}/loss"] = loss
            eval_results[f"{task}/acc"] = acc

        return eval_results
    

if __name__ == "__main__":
    RewardTrainer.main(RewardModelArguments)