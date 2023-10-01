import click
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset, DatasetDict
import dask.dataframe as dd


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
    # )

    train = dd.read_json(f"{data_dir}/train/*.json").compute()
    test = dd.read_json(f"{data_dir}/test/*.json").compute()
    ds = DatasetDict()
    ds["train"] = Dataset.from_pandas(train)
    ds["test"] = Dataset.from_pandas(test)
    print(ds["train"][:5])


    for split in ["train", "test"]:
        for item in tqdm(ds[split], desc=f"checking {split}"):
            try:
                check_item(item)
            except AssertionError as e:
                print("error at", item)
                raise e

    ds.push_to_hub(hub_id, not public)


if __name__ == "__main__":
    main()
