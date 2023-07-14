import re
from .nsmc.generate import NSMCGenerator
from .apeach.generate import APEACHGenerator
from .korquadv1.generate import KorQuADv1Generator
from .klue_ynat.generate import KlueYnatGenerator

ALL_TASKS = {
    "nsmc": NSMCGenerator, 
    "apeach": APEACHGenerator,
    "korquad_v1.0": KorQuADv1Generator,
    "klue_ynat": KlueYnatGenerator
    }


def find_task(pattern: str):
    if pattern == "*":
        return ALL_TASKS
    else:
        pattern = re.compile(pattern)
        return {k: v for k, v in ALL_TASKS.items() if pattern.match(k)}
