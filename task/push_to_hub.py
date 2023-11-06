import click
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, DatasetDict, Sequence, Value, Features
import dask.dataframe as dd
import json


def check_item(item):
    for c in ["task", "id", "input", "instruction"]:
        assert isinstance(item[c], str)
        if c != "input":
            assert len(item[c]) > 0

    assert len(item["positives"]) > 0
    # assert len(item["negatives"]) > 0


@click.command()
@click.argument("hub_id")
@click.argument("data_dir", default="data")
@click.option("--public", default=False)
def main(public: bool, data_dir: str, hub_id: str):
    # ds = load_dataset(
    #     "json",
    #     data_files={
    #         "train": f"{data_dir}/train/*.json",
    #         "test": f"{data_dir}/test/*.json",
    #     },
    #     cache_dir=None
    # )


    train = dd.read_json(f"{data_dir}/train/*.json").compute().reset_index(drop=True)
    test = dd.read_json(f"{data_dir}/test/*.json").compute().reset_index(drop=True)

    # train["positives"] = train.positives.map(json.loads)
    # train["negatives"] = train.negatives.map(json.loads)
    # print(train.dtypes)

    ds = DatasetDict()
    # features = Features(**{
    #     'instruction': Value(dtype='string', id=None), 
    #     'input': Value(dtype='string', id=None), 
    #     'positives': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), 
    #     'negatives': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), 
    #     'task': Value(dtype='string', id=None),
    #     'id': Value(dtype='string', id=None)
    #     })


    ds["train"] = Dataset.from_pandas(train)#, features=features)
    ds["test"] = Dataset.from_pandas(test)#, features=features)
    print(ds["train"][:5])

    for split in ["train", "test"]:
        dataset = ds[split]

        for item in tqdm(dataset, desc=f"checking {split}"):
            try:
                check_item(item)
            except AssertionError as e:
                print("error at", item)
                raise e

        ds[split] = dataset

    print(ds)
    ds.push_to_hub(hub_id, not public,)


if __name__ == "__main__":
    main()
