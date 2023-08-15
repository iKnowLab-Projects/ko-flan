from task.base import BaseGenerator
from dataclasses import dataclass
from datasets import load_dataset
import re
from typing import List


class CSATQAGenerator(BaseGenerator):
    def __init__(self) -> None:
        super().__init__()

    def generate(self, split: str):
        # test 밖에 없음
        if split != "test":
            return

        dataset = load_dataset("HAERAE-HUB/csatqa", "full", split=split)
        for item in dataset:
            label = item["gold"]
            text = item["context"]
            instruction = item["question"]

            # instruction에 input까지 들어가있는 경우
            if text is None:
                unpack = instruction.split("\n", 1)
                if len(unpack) == 2:
                    instruction, text = unpack
                else:
                    # 다음 중 문법적으로 가장 정확한 문장은? 과 같이 아예 input이 없는 경우
                    # 모든 선택지를 본문에 넣는다.
                    instruction = unpack[0]
                    text = "\n".join(
                        f"{i}: " + item[f"option#{i}"] for i in range(1, 6)
                    )

            # 여러 공백이나 특문 이어진 경우 제거(table)
            text = re.sub("(\s)+", "\\1", text).replace("·", "")
            instruction = re.sub("(\s)+", "\\1", instruction).replace("·", "")

            pos = item[f"option#{label}"]
            neg = [item[f"option#{i}"] for i in range(1, 6) if i != label]

            yield {
                "instruction": instruction,
                "input": text,
                "positives": [pos],
                "negatives": neg,
            }
