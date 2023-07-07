from dataclasses import dataclass
from typing import List


@dataclass
class Template:
    instruction: str
    positives: List[str]
    negatives: List[str]
