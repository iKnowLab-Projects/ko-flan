import re
from .nsmc.generate import NSMCGenerator
from .apeach.generate import APEACHGenerator
from .klue_mrc.generate import KLUE_MRCGenerator
from .korquadv1.generate import KorQuADv1Generator
from .klue_ynat.generate import KlueYnatGenerator
from .klue_nli.generate import KLUE_NLIGenerator
from .kornlu_kornli.generate import KorNLUGenerator
from .unsmile.generate import UnSmileGenerator
from .klue_re.generate import KlueReGenerator
from .kobest_copa.generate import KOBEST_COPAGenerator
from .kobest_hellaswag.generate import KOBEST_HELLASWAGGenerator
ALL_TASKS = {
    "nsmc": NSMCGenerator,
    "apeach": APEACHGenerator,
    "korquad_v1.0": KorQuADv1Generator,
    "klue_mrc": KLUE_MRCGenerator,
    "klue_nli": KLUE_NLIGenerator,
    "klue_ynat": KlueYnatGenerator,
    "kor_nlu": KorNLUGenerator,
    "unsmile": UnSmileGenerator,
    "klue_re":KlueReGenerator,
    "kobest_copa":KOBEST_COPAGenerator,
    "kobest_hellaswag":KOBEST_HELLASWAGGenerator
}


def find_task(pattern: str):
    if pattern == "*":
        return ALL_TASKS
    else:
        pattern = re.compile(pattern)
        return {k: v for k, v in ALL_TASKS.items() if pattern.match(k)}
