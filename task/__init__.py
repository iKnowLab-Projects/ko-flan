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
from .kobest_boolq.generate import KobestBoolqGenerator
from .kobest_wic.generate import KobestWicGenerator
from .niklex.generate import NIKLexGenerator
from .mm_spellcorrect.generate import mmSpellCorrectGenerator
from .mm_2022dialogTopic.generate import mm2022DialogTopicGenerator
from .kowow_sentence_passage.generate import KOWOWSENTENCEPASSAGEGenerator
from .kowow_dialog_passage.generate import KOWOWDIALOGPASSAGEGenerator
from .kowow_dialog_topic.generate import KOWOWDIALOGTOPICGenerator
from .kowow_sentence_topic.generate import KOWOWSENTENCETOPICGenerator
from .kowow_passage_sentence.generate import KOWOWPASSAGESENTENCEGenerator
from .ko_nia_normal.generate import KoNiaGenerator
from .mm_dialog.generate import mmDialogGenerator
from .tl_text_entailment.generate import TlTextEntailmentGenerator
from . import nikl, aihub_mrc, haerae
from .ko_relation_relation.generate import KO_RELATION_RELATIONGenerator
from .kor_nli.generate import KorNLIGenerator


ALL_TASKS = {
    "nsmc": NSMCGenerator,
    "apeach": APEACHGenerator,
    "korquad_v1.0": KorQuADv1Generator,
    "klue_mrc": KLUE_MRCGenerator,
    "klue_nli": KLUE_NLIGenerator,
    "klue_ynat": KlueYnatGenerator,
    "kornlu_nli": KorNLUGenerator,
    "unsmile": UnSmileGenerator,
    "klue_re": KlueReGenerator,
    "kobest_copa": KOBEST_COPAGenerator,
    "kobest_hellaswag": KOBEST_HELLASWAGGenerator,
    "kobest_boolq": KobestBoolqGenerator,
    "kobest_wic": KobestWicGenerator,
    "niklex": NIKLexGenerator,
    "nikl_absa": nikl.ABSAGenerator,
    "mms_spellcorrect": mmSpellCorrectGenerator,
    "nikl_cb2020": nikl.CB2020Generator,
    "kowow_dialog_topic": KOWOWDIALOGTOPICGenerator,
    "kowow_dialog_passage": KOWOWDIALOGPASSAGEGenerator,
    "kowow_sentence_sentence": KOWOWSENTENCEPASSAGEGenerator,
    "kowow_sentence_topic": KOWOWSENTENCETOPICGenerator,
    "kowow_passage_sentence": KOWOWPASSAGESENTENCEGenerator,
    "mms_spellcorrect": mmSpellCorrectGenerator,
    "nikl_cb2020": nikl.CB2020Generator,
    "ko_nia_normal": KoNiaGenerator,
    "mm_spellcorrect": mmSpellCorrectGenerator,
    "mm_2022dialogTopic": mm2022DialogTopicGenerator,
    "mm_dialog": mmDialogGenerator,
    "nikl_cb2020": nikl.CB2020Generator,
    "nikl_cb2021": nikl.CB2021Generator,
    "tl_text_entailment": TlTextEntailmentGenerator,
    "ko_relation_relation": KO_RELATION_RELATIONGenerator,
    "aihub_mrc_tech": aihub_mrc.tech.AIHubTechMRCGenerator,
    "aihub_mrc_admin": aihub_mrc.admin.AIHubAdminMRCGenerator,
    "haerae_csatqa": haerae.csatqa.CSATQAGenerator,
    "kor_nli": KorNLIGenerator
}


def find_task(pattern: str):
    if pattern == "*":
        return ALL_TASKS
    else:
        pattern = re.compile(pattern)
        return {k: v for k, v in ALL_TASKS.items() if pattern.match(k)}
