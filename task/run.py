import sys
import click
from task import find_task
from typing import Optional
from tqdm.auto import tqdm
import jsonlines
import json
from pathlib import Path


@click.command()
@click.option("--splits", default="train,test")
@click.option("--tasks", default="*")
@click.option("--output_dir", default="./data")
@click.option("--max_instance_per_task", default=2000)
def main(
    splits: str, tasks: str, output_dir: str, max_instance_per_task: Optional[int]
):
    splits = splits.split(",")
    tasks = find_task(tasks)

    details = {
        "splits": ",".join(splits),
        "tasks": ",".join(list(tasks.keys())),
        "max_instance_per_task": max_instance_per_task
    }

    tqdm_split = tqdm(splits, position=1)
    for split in tqdm_split:
        tqdm.desc = split + " split"
        split_detail = {}

        with jsonlines.open(f"{output_dir}/{split}.jsonl", "w") as fout:
            tqdm_task = tqdm(tasks.items(), position=0, leave=False)

            for task, generator_cls in tqdm_task:
                tqdm_task.desc = task
                generator = generator_cls()

                for i, instance in enumerate(generator.generate(split)):
                    instance["task"] = task
                    instance["id"] = f"{task}-{i + 1}"

                    if i >= max_instance_per_task:
                        break
                    fout.write(instance)

                split_detail[task] = i

        details["split_" + split] = split_detail
        details[f"split_{split}_total"] = sum(split_detail.values())

    with Path(output_dir, "details.json").open("w") as fdetail:
        json.dump(details, fdetail, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
