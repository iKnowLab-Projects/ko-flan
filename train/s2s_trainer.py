from typing import List, Dict
from dataclasses import dataclass
import random
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from .base import BaseTrainer, BaseTrainingArguments, collate_dictlist
from itertools import chain

import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np
from pprint import pprint
from datasets import load_dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class Seq2SeqArguments(BaseTrainingArguments):
    
    def __post_init__(self):
        super().__post_init__()

@dataclass
class Seq2SeqCollator(object):

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    truncation: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = collate_dictlist(features)
        labels = [random.choice(x) for x in features["positives"]]
        labels = [self.tokenizer.bos_token + label + self.tokenizer.eos_token for label in labels]
        input_ids = [instruction + "\n" + input for input, instruction in zip(features["input"], features["instruction"])]
        
        encoder_inputs = self.tokenizer(
            input_ids,
            padding=self.padding,
            add_special_tokens=False,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )

        # <s> contents </s>
        decoder_inputs = self.tokenizer(
            labels,
            padding=self.padding,
            add_special_tokens=False,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )

        return {
            "model_inputs": {
                "decoder_input_ids": decoder_inputs["input_ids"][:, :-1],
                "decoder_attention_mask": decoder_inputs["attention_mask"][:, :-1],
                "labels": decoder_inputs["input_ids"][:, 1:],
                **encoder_inputs   
            },
            **features
        }
    
class Seq2SeqTrainer(BaseTrainer):
    
    def prepare_dataset(self):
        self.dataset = load_dataset(self.args.dataset)
        
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = self.tokenizer.eos_token_id

        return {
            'train': self.dataset['train'],
            'validation': self.dataset['test'] #.select(range(50)),
        }

    def get_collator(self):
        return Seq2SeqCollator(
            tokenizer=self.tokenizer,
            max_length=self.args.max_seq_length ,
            pad_to_multiple_of=8,
            padding="max_length",
            return_tensors="pt",
        )
    
    def _shared_step(self, batch):
        loss = self.model(**batch["model_inputs"]).loss
        return loss
    
    def training_step(self, batch):
        loss = self._shared_step(batch)
        return loss.mean()

    def evaluation_step(self, batch):
        return self._shared_step(batch)


    def collate_evaluation(self, results: List[Dict]):
        loss = torch.stack(results['loss']).mean().item()
        eval_results = {
            "loss": loss
        }
        return eval_results
    

if __name__ == "__main__":
    Seq2SeqTrainer.main(Seq2SeqArguments)