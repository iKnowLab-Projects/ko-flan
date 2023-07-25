import click
from tqdm.auto import tqdm
from datasets import load_dataset


def check_item(item):
    for c in ["task", "id", "input", "instruction"]:
        assert isinstance(item[c], str)
        assert len(item[c]) > 0

    assert len(item["positives"]) > 0
    assert len(item["negatives"]) > 0

@click.command()
@click.argument("data_dir", default="data")
@click.argument("hub_id")
@click.option("--public", default=False)
def main(
    public: bool,
    data_dir: str,
    hub_id: str
    ):
    ds = load_dataset("json", data_files={
        "train": f"{data_dir}/train/*.json",
        "test": f"{data_dir}/test/*.json"
    })


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