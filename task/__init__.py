import re
from .nsmc.generate import NSMCGenerator
from .apeach.generate import APEACHGenerator
from .klue_mrc.generate import KLUE_MRCGenerator

ALL_TASKS = {"nsmc": NSMCGenerator, "apeach": APEACHGenerator, "klue_mrc": KLUE_MRCGenerator}


def find_task(pattern: str):
    if pattern == "*":
        return ALL_TASKS
    else:
        pattern = re.compile(pattern)
        return {k: v for k, v in ALL_TASKS.items() if pattern.match(k)}
