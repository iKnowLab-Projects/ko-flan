import os
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
    all_tasks = dict()
    for task in tasks.split(","):
        all_tasks.update(find_task(task))
    print(list(all_tasks.keys()))

    for split in splits:
        os.makedirs(Path(output_dir, split), exist_ok=True)


    tqdm_split = tqdm(splits, position=1)
    for split in tqdm_split:
        details = {
            "splits": split,
            "tasks": ",".join(list(all_tasks.keys())),
            "max_instance_per_task": max_instance_per_task,
        }
        tqdm.desc = split + " split"
        split_detail = {}

        tqdm_task = tqdm(all_tasks.items(), position=0, leave=False)

        for task, generator_cls in tqdm_task:
            with jsonlines.open(Path(output_dir, split, f"{task}.json"), "w") as fout:
                tqdm_task.desc = task
                generator = generator_cls()

                for i, instance in enumerate(generator.generate(split)):
                    instance["task"] = task
                    instance["id"] = f"{task}-{i + 1}"

                    if max_instance_per_task > 0 and i >= max_instance_per_task:
                        break
                    fout.write(instance)

                split_detail[task] = i

        details["split_" + split] = split_detail
        details[f"split_{split}_total"] = sum(split_detail.values())

        with Path(output_dir, f"details_{split}.json").open("w") as fdetail:
            json.dump(details, fdetail, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
