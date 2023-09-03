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
from .kowow_sentence_sentence.generate import KOWOWSENTENCEPASSAGEGenerator
from .kowow_dialog_passage.generate import KOWOWDIALOGPASSAGEGenerator
from .kowow_dialog_topic.generate import KOWOWDIALOGTOPICGenerator
from .kowow_sentence_topic.generate import KOWOWSENTENCETOPICGenerator
from .kowow_passage_sentence.generate import KOWOWPASSAGESENTENCEGenerator
from .ko_nia_normal.generate import KoNiaGenerator
from .mm_dialog.generate import mmDialogGenerator
from .tl_text_entailment.generate import TlTextEntailmentGenerator
from . import nikl, aihub_mrc, haerae
from .ko_relation_relation.generate import KO_RELATION_RELATIONGenerator
from .ko_relation_field.generate import KO_RELATION_FIELDGenerator
from .mm_2021ner.generate import mm2021NERGenerator
from .mm_2022ner.generate import mm2022NERGenerator
from .mm_2022dialogTopic.generate import mm2022DialogTopicGenerator
from .mm_2022chatTopic.generate import mm2022ChatTopicGenerator
from .mm_2022newsTopic.generate import mm2022NewsTopicGenerator
from .aihub_dialogSummary_topic.generate import aihubDialogSummaryTopicGenerator
from .aihub_dialogSummary_summary.generate import aihubDialogSummarySummaryGenerator
from .aihub_thesisSummary.generate import aihubThesisSummaryGenerator
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
    "mm_spellcorrect": mmSpellCorrectGenerator,
    "nikl_cb2020": nikl.CB2020Generator,
    "kowow_dialog_topic": KOWOWDIALOGTOPICGenerator,
    "kowow_dialog_passage": KOWOWDIALOGPASSAGEGenerator,
    "kowow_sentence_sentence": KOWOWSENTENCEPASSAGEGenerator,
    "kowow_sentence_topic": KOWOWSENTENCETOPICGenerator,
    "kowow_passage_sentence": KOWOWPASSAGESENTENCEGenerator,
    "nikl_cb2020": nikl.CB2020Generator,
    "ko_nia_normal": KoNiaGenerator,
    "mm_2022dialogTopic": mm2022DialogTopicGenerator,
    "mm_2022newsTopic": mm2022NewsTopicGenerator,
    "mm_2022chatTopic": mm2022ChatTopicGenerator,
    "mm_2021ner": mm2021NERGenerator,
    "mm_2022ner": mm2022NERGenerator,
    "mm_dialog": mmDialogGenerator,
    "aihub_dialogSummaryTopic": aihubDialogSummaryTopicGenerator,
    "aihub_dialogSummarySummary": aihubDialogSummarySummaryGenerator,
    "aihub_thesisSummary": aihubThesisSummaryGenerator,
    "nikl_cb2020": nikl.CB2020Generator,
    "nikl_cb2021": nikl.CB2021Generator,
    "tl_text_entailment": TlTextEntailmentGenerator,
    "ko_relation_relation": KO_RELATION_RELATIONGenerator,
    "ko_relation_field": KO_RELATION_FIELDGenerator,
    "kor_nli": KorNLIGenerator

}


def find_task(pattern: str):
    if pattern == "*":
        return ALL_TASKS
    else:
        pattern = re.compile(pattern)
        return {k: v for k, v in ALL_TASKS.items() if pattern.match(k)}
