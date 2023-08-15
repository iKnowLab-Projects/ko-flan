from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import dataclasses
import pprint
from functools import partial
import os
from tqdm import tqdm, trange
import numpy as np
import mlxu
import fnmatch
import click

from flax.traverse_util import flatten_dict
from lm_eval import evaluator
from lm_eval.base import LM
from lm_eval.tasks import get_task_dict, ALL_TASKS
from train.reward_trainer import RewardModelCollator
from torch.utils.data import DataLoader


def _is_json_task(task_name):
    return task_name == "json" or task_name.startswith("json=")


def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        if _is_json_task(pattern):
            task_names.add(pattern)

        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))


class LMEvalHarnessInterface(LM):
    def __init__(self, model_name, batch_size: int, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.batch_size = batch_size
        self.device = device

    def _iter_batch(self, inputs):
        num_cols = len(inputs)
        batch = ([] for _ in range(num_cols))

        for i in range(len(inputs[0])):
            for j in range(num_cols):
                batch[j].append(inputs[j][i])

            if len(batch[0]) == self.batch_size:
                yield batch
                batch = ([] for _ in num_cols)

        if len(batch[0]) > 0:
            yield batch

    def greedy_until(self, inputs):
        # prefix, until = zip(*inputs)
        # return self.lm_client.greedy_until(prefix, until)
        pass

    def loglikelihood_rolling(self, inputs):
        # loglikelihood, is_greedy = self.lm_client.loglikelihood_rolling(inputs)
        # return list(zip(loglikelihood, is_greedy))
        pass

    def loglikelihood(self, inputs):
        ll = []

        for batch in self._iter_batch(inputs):
            prefix, text = zip(*batch)
            print(prefix, text)
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True).to(
                self.device
            )

            scores = self.model(**inputs).logits[:, 0].tolist()
            ll.extend(scores)

        return list(zip(ll, [False] * len(ll)))


@click.command()
@click.argument("task", type=str)
@click.argument("model", type=str)
@click.option("--limit", default=0)
@click.option("--batch_size", default=8)
def main(task: str, model: str, limit: int, batch_size: int):
    model = LMEvalHarnessInterface(model, batch_size, "cuda:0")

    task_list = pattern_match(task.split(","), ALL_TASKS)
    print("tasks", task, "->", task_list)  # , "in", ALL_TASKS)

    print("eval results")
    results = evaluator.simple_evaluate(
        model,
        model_args="",
        tasks=get_task_dict(task_list),
        no_cache=True,
        num_fewshot=0,
        device="cuda:0",
        limit=None if limit <= 0 else limit,
    )
    pprint.pprint(results)


"""
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git@polyglot
python -m eval.eval_reward "kobest_*" checkpoint/koflan-base-0731/epoch-9
"""
if __name__ == "__main__":
    main()
