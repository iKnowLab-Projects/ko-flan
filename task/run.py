import os
import click
from task import find_task
from typing import Optional
from tqdm.auto import tqdm
import jsonlines
import json
from pathlib import Path
import random
import ray
import psutil


@ray.remote
class ParallelTqdm:

    def __init__(self, total: int) -> None:
        self.tqdm_task = tqdm(total=total, leave=False, position=0)
    
    def update(self, n: int = 1):
        self.tqdm_task.update(n)

@ray.remote
class SplitGenerator:        
    def __init__(self, tqdm, split, max_instance_per_task, output_dir, require_negative) -> None:
        self.split = split
        self.max_instance_per_task = max_instance_per_task
        self.output_dir = output_dir
        self.tqdm = tqdm
        self.require_negative = require_negative

    def check_item(self, item):
        for c in ["task", "id", "input", "instruction"]:
            assert isinstance(item[c], str)
            assert len(item[c]) > 0
        
        # type check
        for c in ["positives", "negatives"]:
            assert isinstance(item[c], list)
        
        assert len(item["positives"]) > 0
        if self.require_negative:
            assert len(item["negatives"]) > 0
        
    def map_generator(self, generator, split, task):        
        def mapper(pair):
            index, item = pair
            item["task"] = task
            item["id"] = f"{task}-{index + 1}"
            return item

        items = map(mapper, enumerate(generator.generate(split)))

        if self.require_negative:
            items = filter(lambda x: x["negatives"] is not None and len(x["negatives"]) > 0, items)

        items = list(items)
        
        for index, item in enumerate(items):
            try:
                self.check_item(item)
            except:
                print(f"에러 발생: task: {task}, split: {split}, generator: {type(generator)}, index: {index}")
                print("문제의 값: ", item)
                raise

        return items
    
    def write_task(self, task, generator_cls):
        with jsonlines.open(Path(self.output_dir, self.split, f"{task}.json"), "w") as fout:
            generator = generator_cls()
            items = self.map_generator(generator, self.split, task)
            
            if self.max_instance_per_task > 0 and len(items) >= self.max_instance_per_task:
                items = random.choices(items, k=self.max_instance_per_task)

            for item in items:
                fout.write(item)

            self.tqdm.update.remote()

        return task, len(items)
    
    def write_all_tasks(self, all_tasks):
        futures = [self.write_task.remote(task, generator_cls) for task, generator_cls in all_tasks.items()]
        split_detail = dict(ray.get(futures))
        return split_detail


@click.command()
@click.option("--splits", default="train,test")
@click.option("--tasks", default="*")
@click.option("--output_dir", default="./data")
@click.option("--max_instance_per_task", default=-1)
@click.option("--require_negative/--allow_no_negative", default=True)
@click.option("--num_proc", default=0, help="number of parallel ray processes, 0 to max cpu")
def main(
    splits: str, tasks: str, output_dir: str, max_instance_per_task: Optional[int],
    num_proc: int, require_negative: bool
):
    if num_proc == 0:
        num_proc = psutil.cpu_count()
    ray.init(num_cpus=num_proc)
    print("num_cpus", num_proc)

    splits = splits.split(",")
    all_tasks = dict()
    for task in tasks.split(","):
        all_tasks.update(find_task(task))
    print("total tasks:", len(all_tasks), list(all_tasks.keys()))

    for split in splits:
        os.makedirs(Path(output_dir, split), exist_ok=True)

    tqdm_split = tqdm(splits, position=1)
    for split in tqdm_split:
        details = {
            "splits": split,
            "tasks": ",".join(list(all_tasks.keys())),
            "max_instance_per_task": max_instance_per_task,
        }
        tqdm_task = ParallelTqdm.remote(len(all_tasks))

        futures = []
        for task, generator_cls in all_tasks.items():
            generator = SplitGenerator.remote(
                tqdm=tqdm_task,
                split=split,
                output_dir=output_dir,
                require_negative=require_negative,
                max_instance_per_task=max_instance_per_task
                )
            futures.append(generator.write_task.remote(task, generator_cls))

        split_detail = ray.get(futures)
        print(split, split_detail)
        split_detail = dict(split_detail)

        details["split_" + split] = split_detail
        details[f"split_{split}_total"] = sum(split_detail.values())

        with Path(output_dir, f"details_{split}.json").open("w") as fdetail:
            json.dump(details, fdetail, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
