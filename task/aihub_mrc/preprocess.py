import glob
import json, jsonlines
from bs4 import BeautifulSoup
from tqdm import tqdm
import random
import os


def html4text(html):
    soup = BeautifulSoup(html)
    return soup.get_text()


def handle_tech_mrc_item(item):
    outputs = []

    for context_dict in item["dataset"]["context_info"]:
        context = html4text(context_dict["context"])
        questions = context_dict["qas"]
        all_answers = set([qa["answer"] for qa in questions])

        for qa in questions:
            type = qa["answer_type"]
            pos = qa["answer"]

            if type == "다지선다형":
                neg = qa["wrong_answer"]


def tech_mrc():
    train_path = "C:/Users/heegyukim/Downloads/152.기술과학 문서 기계독해 데이터/01.데이터/Training/02.라벨링데이터/**/*.json"
    test_path = "C:/Users/heegyukim/Downloads/152.기술과학 문서 기계독해 데이터/01.데이터/Validation/02.라벨링데이터/**/*.json"

    for split, path in zip(["train", "test"], [train_path, test_path]):
        with jsonlines.open(f"tech_mrc_{split}.json", "w") as fout:
            for file in tqdm(glob.glob(path), desc=split):
                with open(file, encoding="utf-8") as f:
                    content = f.read()
                    fout.write({"file": os.path.basename(file), "content": content})


def admin_mrc():
    train_path = "C:/Users/heegyukim/Downloads/016.행정 문서 대상 기계독해 데이터/01.데이터/1.Training/라벨링데이터/**/*.json"
    test_path = "C:/Users/heegyukim/Downloads/016.행정 문서 대상 기계독해 데이터/01.데이터/2.Validation/라벨링데이터/**/*.json"

    for split, path in zip(["train", "test"], [train_path, test_path]):
        with jsonlines.open(f"tech_admin_{split}.json", "w") as fout:
            for file in tqdm(glob.glob(path), desc=split):
                with open(file, encoding="utf-8") as f:
                    content = f.read()
                    fout.write({"file": os.path.basename(file), "content": content})


if __name__ == "__main__":
    # tech_mrc()
    admin_mrc()
