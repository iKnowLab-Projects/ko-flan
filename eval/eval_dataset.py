import click
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
from pprint import pprint

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling,
    AutoModelForSeq2SeqLM
)

from .eval_mapper import EVAL_LIST


class KoFlanEvalDataset(Dataset):
    def __init__(self, tokenizer, max_length: int = 512) -> None:
        super().__init__()

        # self.items = [item for case in tqdm(dataset, desc="preparing...") for item in self.dataset_unrolling(deepcopy(case))]
        items = []

        for task, dataset_info in EVAL_LIST.items():
            mapper = dataset_info["mapper"]
            dataset = load_dataset(**dataset_info["load_args"])

            for task_idx, case in enumerate(dataset):
                for item in self.dataset_unrolling(mapper(case)):
                    item["task"] = task
                    item["id"] = f"{task}-{task_idx}"
                    items.append(item)

        self.items = items
        self.max_length = max_length
        self.tokenizer = tokenizer
        tokenizer.truncation_side = "left"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index) -> Dict:
        item = self.items[index]
        instruction = f"{item['input']} [SEP] {item['instruction']}"
        inputs = self.tokenizer(
            instruction, item["answer"], truncation=True, max_length=self.max_length
        )
        return inputs

    def dataset_unrolling(self, item: Dict):
        positives = item.pop("positives")
        negatives = item.pop("negatives")

        for pos in positives:
            yield {"answer_type": "positive", "answer": pos, **item}

        for neg in negatives:
            yield {"answer_type": "negative", "answer": neg, **item}


@torch.no_grad()
@click.command()
@click.option("--model_name_or_path", default="checkpoint/test/epoch-9")
@click.option("--batch_size", default=8)
@click.option("--device", default="cuda:0")
def main(model_name_or_path: str, device: str, batch_size: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = (
         AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        .to(device)
        .eval()
    )

    # dataset = load_dataset(dataset, split=split)
    dataset = KoFlanEvalDataset(tokenizer=tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer, mlm=False, pad_to_multiple_of=8
        ),
    )

    df = pd.DataFrame(dataset.items)
    print("샘플")
    pprint(df.groupby("task").sample(1).to_dict(orient="records"))

    scores = []

    for batch in tqdm(data_loader, desc="evaluating..."):
        batch = {k: v.to(device) for k, v in batch.items()}
        del batch["labels"]
        logits = model(**batch).logits.squeeze(1).tolist()
        scores.extend(logits)

    df["score"] = scores

    # positives 중 가장 높은 점수와 negatives 중에서 가장 높은 점수를 비교해서
    # 가장 좋은 positive가 가장 좋은 negative보다 높으면 win으로 취급한다.
    # 이후 task별로 win rate를 평가한다.
    
    df_score = df[["task", "id", "answer_type", "score"]]
    df_score = (
        df_score.groupby(["task", "id", "answer_type"])["score"]
        .apply(list)
        .reset_index()
    )
    df_score.score = df_score.score.map(max)
    df_score = df_score.pivot(
        index=["task", "id"], columns="answer_type", values="score"
    )
    print(df_score)
    print(df_score.columns)

    df_score["win"] = df_score[("positive")] > df_score[("negative")]
    task_win_rate = df_score.groupby("task").agg({"win": "mean"})

    print("task 별 최종 점수")
    pprint(task_win_rate.to_dict()["win"])
    task_win_rate.to_csv("task_score.csv")


if __name__ == "__main__":
    main()
