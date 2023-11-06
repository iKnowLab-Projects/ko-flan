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
        "positives": [kobest_boolq_labels[item["label"]]],
        "negatives": [kobest_boolq_labels[1 - item["label"]]],
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
        "negatives": aihub_topic_labels - set(item['positives'])
    }


nsmc_labels = ["부정", "긍정"]


def nsmc_mapper(item):
    return {
        "instruction": "주어진 문장의 감정을 분류하세요",
        "input": item["document"],
        "positives": [nsmc_labels[item["label"]]],
        "negatives": [nsmc_labels[1 - item["label"]]],
    }


apeach_labels = ["혐오 표현이 아닙니다", "혐오표현"]


def apeach_mapper(item):
    return {
        "instruction": "혐오성을 분류해보세요.",
        "input": item["text"],
        "positives": [nsmc_labels[item["class"]]],
        "negatives": [nsmc_labels[1 - item["class"]]],
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
        mapper=aihub_topic_mapper,)
}
