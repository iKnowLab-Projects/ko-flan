import transformers
from transformers import HfArgumentParser, set_seed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from datasets import disable_caching, load_dataset
import accelerate
from accelerate import Accelerator
from tqdm.auto import tqdm
from accelerate.logging import get_logger

import os
import evaluate
from pprint import pprint
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig, AutoModelForTokenClassification, TrainingArguments


MODEL_TYPES = {
    "sequence-classification": AutoModelForSequenceClassification,
    "token-classification": AutoModelForTokenClassification,
    "causal-lm": AutoModelForCausalLM,
}

@dataclass
class BaseTrainingArguments(TrainingArguments):
    save_epochs: int = 1

    dataset: Optional[str] = None
    max_seq_length: int = 512
    project: str = "ko-flan"
    model_name_or_path: str = ""
    tokenizer_name: Optional[str] = None
    num_labels: int = 1

    config_name: Optional[str] = None
    revision: Optional[str] = None
    from_flax: bool = False
    model_type: str = "sequence-classification"


def collate_dictlist(dl):
    from collections import defaultdict

    out = defaultdict(list)

    for d in dl:
        for k, v in d.items():
            out[k].append(v)

    return out

class BaseTrainer:
    def __init__(self, accelerator: Accelerator, args: BaseTrainingArguments) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.args = args

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_model(self, args):
        model_cls = MODEL_TYPES[args.model_type]

        kwargs = {}

        if args.model_type == "sequence-classification":
            kwargs["num_labels"] = args.num_labels

        if args.config_name is not None:
            config = AutoConfig.from_pretrained(args.config_name)
            model = model_cls(config, **kwargs)
        elif args.model_name_or_path is not None:
            model = model_cls.from_pretrained(
                args.model_name_or_path,
                revision=args.revision,
                from_flax=args.from_flax,
                **kwargs
                )
        else:
            raise Exception("config_name or model_name_or_path 가 지정되어야 합니다.")
    
        return model

    def get_tokenizer(self, args):
        from transformers import AutoTokenizer

        if args.tokenizer_name is not None:
            return AutoTokenizer.from_pretrained(args.tokenizer_name)
        elif args.model_name_or_path is not None:
            return AutoTokenizer.from_pretrained(args.model_name_or_path)
        else:
            raise Exception("config_name or model_name_or_path 가 지정되어야 합니다.")

    def setup(self):
        self.model = self.get_model(self.args)
        self.tokenizer = self.get_tokenizer(self.args)

        datasets = self.prepare_dataset()

        self.train_dataloader = self._create_dataloader(
            datasets.get('train'),
            True
        )
        self.eval_dataloader = self._create_dataloader(
            datasets.get('validation'),
        )

        steps_per_epoch = len(datasets.get('train')) / (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
        )
        total_steps = int(self.args.num_train_epochs * steps_per_epoch)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        lr_scheduler = None

        if self.args.lr_scheduler_type == "linear":
            lr_scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer, 
                self.args.warmup_steps if self.args.warmup_steps > 0 else int(total_steps * self.args.warmup_ratio),
                total_steps
                )
        elif self.args.lr_scheduler_type == "cosine":
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer, 
                self.args.warmup_steps if self.args.warmup_steps > 0 else int(total_steps * self.args.warmup_ratio),
                total_steps
                )


        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.model, optimizer, self.train_dataloader, lr_scheduler, self.eval_dataloader
        )

        self.accelerator.register_for_checkpointing(lr_scheduler)

    def prepare_dataset(self):
        pass

    def get_collator(self):
        return None

    def training_step(self, batch):
        """
            return loss
        """
        pass

    def evaluation_step(self, batch):
        """
            return dict
        """
        pass

    def collate_evaluation(self, results: List[Dict]):
        """
            return dict(metric)
        """
        return None

    def _create_dataloader(self, dataset, shuffle=True):
        if dataset is None:
            return None

        kwargs = {}
        collator = self.get_collator()
        if collator is not None:
            kwargs['collate_fn'] = collator

        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=shuffle,
            **kwargs
            )

    def train(self):
        global_step = 0
        optimizer_step = 0

        for epoch in tqdm(
            range(int(self.args.num_train_epochs)),
            position=0,
            disable=not self.accelerator.is_local_main_process,
        ):
            self.model.train()
            epoch_tqdm = tqdm(
                self.train_dataloader,
                disable=not self.accelerator.is_local_main_process,
                position=1,
                leave=False,
            )

            for step, batch in enumerate(epoch_tqdm):
                with self.accelerator.accumulate(self.model):
                    step_output = self.training_step(batch)
                    if torch.is_tensor(step_output):
                        loss = step_output
                    else:
                        loss = step_output['loss']

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    optimizer_step += 1

                    if (
                        self.accelerator.is_main_process
                        and optimizer_step % self.args.logging_steps == 0
                    ):
                        if torch.is_tensor(step_output):
                            metrics = {"train/loss": step_output.item()}
                        else:
                            metrics = {f"train/{k}": v.item() for k, v in step_output.items()}

                        metrics["optimizer_step"] = optimizer_step
                        metrics["train/learning_rate"] = self.lr_scheduler.scheduler._last_lr[0]
                        metrics["train/loss"] = loss.item() * self.args.gradient_accumulation_steps
                        self.accelerator.log(metrics)
                        print()
                        pprint(metrics)
                        print()

                    epoch_tqdm.set_description(
                        f"loss: {loss.item() * self.args.gradient_accumulation_steps}"
                    )

                    if (
                        self.args.do_eval
                        and self.args.evaluation_strategy == "steps"
                        and optimizer_step % self.args.eval_steps == 0
                    ):
                        self.evaluate(epoch, optimizer_step)

                    global_step += 1

            if self.args.evaluation_strategy == "epoch" and epoch % self.args.save_epochs == 0:
                if self.accelerator.is_main_process:
                    self.save_model(f"epoch-{epoch}")
                self.accelerator.wait_for_everyone()

            if self.args.do_eval and self.args.evaluation_strategy == "epoch":
                self.evaluate(epoch, optimizer_step)

        if self.args.save_strategy == "last":
            if self.accelerator.is_main_process:
                self.save_model(f"epoch-{epoch}-last")
            self.accelerator.wait_for_everyone()

    def save_model(self, name):
        run_name = self.args.run_name.replace("/", "__")
        path = f"{self.args.output_dir}/{run_name}/{name}"
        device = next(self.model.parameters()).device
        unwrapped_model = self.accelerator.unwrap_model(self.model).cpu()
        unwrapped_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        unwrapped_model.to(device)

    @torch.no_grad()
    def evaluate(self, epoch, optimizer_step):
        self.model.eval()

        epoch_tqdm = tqdm(
            self.eval_dataloader,
            disable=not self.accelerator.is_local_main_process,
            position=1,
            leave=False,
        )
        step_outputs = []
        for step, batch in enumerate(epoch_tqdm):
            outputs = self.evaluation_step(batch)
            if torch.is_tensor(outputs):
                outputs = {"loss": outputs}
            step_outputs.append(outputs)

        eval_outputs = self.accelerator.gather_for_metrics(step_outputs)

        if self.accelerator.is_local_main_process:
            eval_outputs = collate_dictlist(eval_outputs)
            eval_results = self.collate_evaluation(eval_outputs)
            eval_results = {f"eval/{k}": v for k, v in eval_results.items()}
            self.accelerator.log(eval_results)

        self.accelerator.wait_for_everyone()
        self.model.train()

    @classmethod
    def main(trainer_cls, arg_cls: BaseTrainingArguments):
        parser = HfArgumentParser(
            (arg_cls,)
        )
        args = parser.parse_args()

        set_seed(args.seed)

        os.environ["WANDB_NAME"] = args.run_name
        accelerator = Accelerator(
            log_with="wandb",
            kwargs_handlers=[
                accelerate.DistributedDataParallelKwargs(broadcast_buffers=False,)
            ],
            gradient_accumulation_steps=args.gradient_accumulation_steps
            )
        accelerator.init_trackers(
            args.project,
            config=args
        )
        trainer = trainer_cls(accelerator, args)
        trainer.setup()
        if args.do_train:
            trainer.train()
        elif args.do_eval:
            trainer.evaluate(0, 0)

        accelerator.end_training()