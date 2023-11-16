import re


klue_ynat_labelToTextDict = {
    0: "IT과학",
    1: "경제",
    2: "사회",
    3: "생활문화",
    4: "세계",
    5: "스포츠",
    6: "정치",
}


klue_ynat_labels = set(klue_ynat_labelToTextDict.values())


def klue_ynat_mapper(item):
    positives = [klue_ynat_labelToTextDict[item["label"]]]
    return {
        "instruction": "문장을 읽고 주제를 분류하세요",
        "input": item["title"],
        "positives": positives,
        "negatives": klue_ynat_labels - set(positives),
    }


kobest_wic_labels = ["아니오", "예"]


def kobest_wic_mapper(item):
    return {
        "instruction": "주어진 두 문장에서 단어 {word}은(는) 동일한 의미로 사용되었나요?".format(
            word=item["word"]
        ),
        "input": "문장1: {context_1}\n문장2: {context_2}".format(**item),
        "positives": [kobest_wic_labels[item["label"]]],
        "negatives": [kobest_wic_labels[1 - item["label"]]],
    }


copa_question = {"결과": "이후에 이어질 결과는?", "원인": "이러한 일이 일어난 원인은?"}


def kobest_copa_mapper(item):
    answers = [item["alternative_1"], item["alternative_2"]]
    return {
        "instruction": copa_question[item["question"]],
        "input": item["premise"],
        "positives": [answers[item["label"]]],
        "negatives": [answers[1 - item["label"]]],
    }


def kobest_hellaswag_mapper(item):
    answers = [item[f"ending_{i}"] for i in range(1, 5)]
    label = answers[item["label"]]
    answers.remove(label)

    return {
        "instruction": "이후에 이어질 내용으로 가장 적절한 것은?",
        "input": item["context"],
        "positives": [label],
        "negatives": answers,
    }


kobest_boolq_labels = ["아니오", "예"]


def kobest_boolq_mapper(item):
    return {
        "instruction": item["question"],
        "input": item["paragraph"],
        "positives": [kobest_boolq_labels[item["label"]]],
        "negatives": [kobest_boolq_labels[1 - item["label"]]],
    }


kobest_sentineg_labels = ["부정", "긍정"]


def kobest_sentineg_mapper(item):
    return {
        "instruction": "주어진 문장의 감정을 분류하세요",
        "input": item["sentence"],
        "positives": [kobest_sentineg_labels[item["label"]]],
        "negatives": [kobest_sentineg_labels[1 - item["label"]]],
    }


aihub_topic_labels = ['공통',
                      '토지',
                      '정보통신',
                      '교통',
                      '문화_체육_관광',
                      '농업_축산',
                      '세무',
                      '자동차',
                      '경제',
                      '행정',
                      '안전건설',
                      '상하수도',
                      '보건소',
                      '건축허가',
                      '복지',
                      '환경미화',
                      '산림',
                      '위생']


def aihub_topic_mapper(item):
    return {
        "instruction": '주어진 민원을 알맞은 카테고리로 분류하시오',
        "input": item["input"],
        "positives": item['positives'],
        "negatives": list(set(aihub_topic_labels) - set(item['positives']))
    }

def kowow_topic_mapper(item):
    context = item["context"].replace("0_Apprentice", "0").replace("1_Wizard", "1")
    return {
        "instruction": '대화 내용을 읽고 대화의 주제를 분류하세요.',
        "input": context,
        "positives": item["postive_topic"],
        "negatives": item["negative_topic"]
    }

aihub_dialog_topics = {'개인 및 관계',
 '미용과 건강',
 '상거래(쇼핑)',
 '시사/교육',
 '식음료',
 '여가 생활',
 '일과 직업',
 '주거와 생활',
 '행사'}

def aihub_dialog_topic_mapper(item):
    context = item["conversation"].replace("name 01", "A").replace("name 02", "B").replace("name 03", "C").replace("name 04", "D")
    return {
        "instruction": '대화 내용을 읽고 대화의 주제를 분류하세요.',
        "input": context,
        "positives": [item["topic"]],
        "negatives": list(aihub_dialog_topics - {item["topic"]})
    }

mm_chat_topics = {'가사 및 가족',
 '공연 및 관람',
 '날씨와 계절',
 '미용과 건강',
 '반려동물',
 '사회 생활 및 활동',
 '쇼핑과 상품',
 '시사(정치, 경제, 사회)',
 '식음료',
 '여행',
 '연애와 결혼',
 '일과 직업',
 '일상',
 '일상 트렌드',
 '콘텐츠',
 '학교 생활'}
def mm_chat_topic_mapper(item):
    context = item["paragraph"]
    return {
        "instruction": '대화 내용을 읽고 대화의 주제를 분류하세요.',
        "input": context,
        "positives": [item["topic"]],
        "negatives": list(mm_chat_topics - {item["topic"]})
    }

ko_relation_fields = {'지역_사회', '국제', 'IT_과학', '연예', '문화', '정치', '의약학', '스포츠', '경제',
       '기계공학', '인문학', '사회과학', '전기전자'}

def ko_relation_fields_mapper(item):
    context = item["sentence"]
    return {
        "instruction": '이 문장의 주제나 분야는 무엇이라고 생각하시나요?',
        "input": context,
        "positives": [item["field"]],
        "negatives": list(ko_relation_fields - {item["field"]})
    }

def haerae_csatqa_mapper(item):
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

    return {
        "instruction": instruction,
        "input": text,
        "positives": [pos],
        "negatives": neg,
    }

    
nsmc_labels = ["부정", "긍정"]


def nsmc_mapper(item):
    return {
        "instruction": "주어진 문장의 감정을 분류하세요",
        "input": item["document"],
        "positives": [nsmc_labels[item["label"]]],
        "negatives": [nsmc_labels[1 - item["label"]]],
    }


apeach_labels = ["아니요", "예"]

def apeach_mapper(item):
    return {
        "instruction": "이 문장에 혐오 표현이 담겨있나요?",
        "input": item["text"],
        "positives": [apeach_labels[item["class"]]],
        "negatives": [apeach_labels[1 - item["class"]]],
    }


EVAL_LIST = {
    "klue-ynat": dict(
        load_args=dict(path="klue", name="ynat", split="validation"),
        mapper=klue_ynat_mapper,
    ),
    "nsmc": dict(load_args=dict(path="nsmc", split="test"), mapper=nsmc_mapper),
    "apeach": dict(
        load_args=dict(path="jason9693/APEACH", split="test"), mapper=apeach_mapper
    ),
    "kobest-wic": dict(
        load_args=dict(path="skt/kobest_v1", name="wic", split="test"),
        mapper=kobest_wic_mapper,
    ),
    "kobest-copa": dict(
        load_args=dict(path="skt/kobest_v1", name="copa", split="test"),
        mapper=kobest_copa_mapper,
    ),
    "kobest-hellaswag": dict(
        load_args=dict(path="skt/kobest_v1", name="hellaswag", split="test"),
        mapper=kobest_hellaswag_mapper,
    ),
    "kobest-boolq": dict(
        load_args=dict(path="skt/kobest_v1", name="boolq", split="test"),
        mapper=kobest_boolq_mapper,
    ),
    "kobest-sentineg": dict(
        load_args=dict(path="skt/kobest_v1", name="sentineg", split="test"),
        mapper=kobest_sentineg_mapper,
    ),
    "aihub_complaints_topic": dict(
        load_args=dict(path="iknow-lab/aihub_complaints_topic", split="test"),
        mapper=aihub_topic_mapper,),
    "aihub_dialog_topic": dict(
        load_args=dict(path="iknow-lab/aihub_dialogSummary", split="test"),
        mapper=aihub_dialog_topic_mapper
    ),
    "kowow_dialog_topic": dict(
        load_args=dict(path="iknow-lab/kowow_dialog", split="test"),
        mapper=kowow_topic_mapper
    ),
    "mm_chat_topic": dict(
        load_args=dict(path="iknow-lab/mm_2022chatTopic", split="test"),
        mapper=mm_chat_topic_mapper
    ),
    "ko_relation_fields": dict(
        load_args=dict(path="iknow-lab/korean_relation", split="test"),
        mapper=ko_relation_fields_mapper
    ),
    
    ** {
        f"csatqa-{k}": dict(
            load_args=dict(path="HAERAE-HUB/csatqa", name=k, split="test"),
            mapper=haerae_csatqa_mapper
        )
        for k in ["GR", "LI", "RCH", "RCS", "RCSS", "WR"]
    }
    
}
