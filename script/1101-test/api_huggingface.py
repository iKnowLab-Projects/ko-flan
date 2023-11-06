# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("iknow-lab/ko-flan-zero-v0-0731")
model = AutoModelForSequenceClassification.from_pretrained("iknow-lab/ko-flan-zero-v0-0731")

def inference(instruction, input, labels):
    instruction = f"{input} [SEP] {instruction}"
    inputs = tokenizer([instruction] * len(labels), labels, truncation=True, padding=True, return_tensors="pt")
    
    scores = model(**inputs).logits.squeeze(1).tolist()
    output = dict(zip(labels, scores))

    print(instruction, output)

inference(
    "문장을 감성분류해주세요",
    "아 영화 개노잼",
    ["긍정적", "부정적"]
)

inference(
    "글과 관련된 내용을 만들어주세요",
    "예전에는 주말마다 극장에 놀러갔는데 요새는 좀 안가는 편이에요",
    ["영화에 관한 글이다", "드라마에 관한 글입니다"]
)


inference(
    "글을 읽고 시장에 미칠 영향을 판단해보세요",
    """인천발 KTX와 관련한 송도역 복합환승센터가 사실상 무산, 단순 철도·버스 위주 환승시설로 만들어진다. 이 때문에 인천시의 인천발 KTX 기점에 앵커시설인 복합환승센터를 통한 인근 지역 경제 활성화를 이뤄낸다는 계획의 차질이 불가피하다.

25일 시에 따르면 연수구 옥련동 104 일대 29만1천725㎡(8만8천평)에 추진 중인 2만8천62가구 규모의 송도역세권구역 도시개발사업과 연계, KTX 송도역 복합환승센터와 상업시설·업무시설 등의 조성을 추진 중이다. """,
    ["긍정", "부정", "중립"]
)
