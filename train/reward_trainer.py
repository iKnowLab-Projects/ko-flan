from typing import List, Dict
from dataclasses import dataclass
import random
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from .base import BaseTrainer, BaseTrainingArguments, collate_dictlist

import torch

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
    padding: Union[bool, str, PaddingStrategy] = "max_length"
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

    def training_step(self, batch):
        pos = self.model(**batch["positives"]).logits
        neg = self.model(**batch["negatives"]).logits
        return (pos - neg).sigmoid().mean()

    def evaluation_step(self, batch):
        loss = self.training_step(batch)

        return {
            'loss': loss,
        }

    def collate_evaluation(self, results: List[Dict]):
        eval_mean_loss = torch.stack(results['loss']).mean().item()
        eval_results = {
            "loss": eval_mean_loss,
        }
        pprint("evaluation result")
        pprint(eval_results)
        return eval_results
    

if __name__ == "__main__":
    RewardTrainer.main(RewardModelArguments)