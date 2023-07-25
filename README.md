# ko-flan

## 학습
### Reward Model
```bash
# script/train_reward.sh

python -m train.reward_trainer \
    --do_train --do_eval \
    --project ko-flan \
    --run_name test \
    --model_name_or_path monologg/koelectra-small-v3-discriminator \
    --dataset iknow-lab/koflan-test-110k-0725 \
    --logging_steps 500 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --output_dir ./checkpoint
```

## 실행방법

루트 디렉터리에서 아래처럼 실행할 경우, 모든 테스크별로 2천개씩 data/폴더에 생성됩니다.
```
python -m task.run --max_instance_per_task 10000
```

예시 2. nsmc, apeach task만 생성하고 테스크 당 최대 10개만 생성하기
```
python -m task.run --tasks "nsmc,apeach" --max_instance_per_task 10
```

## Task 추가방법
0. [노션 페이지](https://www.notion.so/c421dc9deeec42e092a1631602723ddb?v=c0e6ecf3cece4a09a381a91d3ac7dfa3&pvs=4)에서 본인이 진행할 Task를 정한뒤 담당자에 표시하고, "진행중" 상태로 바꿉니다.
1. task/ 에 디렉터리를 만들고 .py 파일을 생성하세요
2. task.base의 BaseGenerator를 상속하는 Generator 클래스를 만드세요.
3. generate() 메서드에서 주어진 split(train or test)에 맞는 결과를 yield 하면 됩니다. generator 예제 참고: [task/nsmc/generate.py](./task/nsmc/generate.py)
4. [task/__init__.py](task/__init__.py)안의 ALL_TASKS 에 구현한 generator class를 추가하세요.
5. [task/run.py](task/run.py)를 실행하세요.  
6. 결과가 기본적으로 data/ 폴더에 저장됩니다.


각 Generator는 아래와 같은 형식의 dict를 반환해야 합니다. positives와 negatives의 개수는 task에 따라 다릅니다
| task 유형 | positive 개수 | negative 개수 |
| --- | --- | --- |
| 이진분류 | 1 | 1 |
| 멀티클래스 분류 | 1 | N |
| 멀티라벨 분류 | N | N |
| MRC | 1 | N |

예시 1. 이진 분류
```json
{
    "instruction": "이 댓글을 읽고 작성자의 감정을 파악해보세요.", 
    "input": "굳 ㅋ", 
    "positives": ["긍정적인 리뷰"], 
    "negatives": ["부정"]
}
```

예시 2. 멀티클래스 분류, MRC
```json
# 멀티클래스 분류, instruction에 라벨을 포함하지 않아도 됩니다.
{
    "instruction": "문장을 읽고 주제를 분류하세요",
    "input": "어버이날 맑다가 흐려져…남부지방 옅은 황사",
    "positives": ["사회"],
    "negatives": ["IT과학", "생활문화", "스포츠", "세계", "정치"]
}

# MRC 
{
    "instruction": "바그너는 괴테의 파우스트를 읽고 무엇을 쓰고자 했는가?", 
    "input": "1839년 바그너는 괴테의 파우스트을 처음 읽고 그 내용에 마음이 끌려 이를 소재로 해서 하나의 교향곡을 쓰려는 뜻을 갖는다. 이 시기 바그너는 1838년에 빛 독촉으로 산전수전을 다 걲은 상황이라 좌절과 실망에 가득했으며 메피스토펠레스를 만나는 파우스트의 심경에 공감했다고 한다. 또한 파리에서 아브네크의 지휘로 파리 음악원 관현악단이 연주하는 베토벤의 교향곡 9번을 듣고 깊은 감명을 받았는데, 이것이 이듬해 1월에 파우스트의 서곡으로 쓰여진 이 작품에 조금이라도 영향을 끼쳤으리라는 것은 의심할 여지가 없다. 여기의 라단조 조성의 경우에도 그의 전기에 적혀 있는 것처럼 단순한 정신적 피로나 실의가 반영된 것이 아니라 베토벤의 합창교향곡 조성의 영향을 받은 것을 볼 수 있다. 그렇게 교향곡 작곡을 1839년부터 40년에 걸쳐 파리에서 착수했으나 1악장을 쓴 뒤에 중단했다. 또한 작품의 완성과 동시에 그는 이 서곡(1악장)을 파리 음악원의 연주회에서 연주할 파트보까지 준비하였으나, 실제로는 이루어지지는 않았다. 결국 초연은 4년 반이 지난 후에 드레스덴에서 연주되었고 재연도 이루어졌지만, 이후에 그대로 방치되고 말았다. 그 사이에 그는 리엔치와 방황하는 네덜란드인을 완성하고 탄호이저에도 착수하는 등 분주한 시간을 보냈는데, 그런 바쁜 생활이 이 곡을 잊게 한 것이 아닌가 하는 의견도 있다.", 
    "positives": ["교향곡"], 
    "negatives": ["1악장", "베토벤의 교향곡 9번", "파우스트"]
}
```

예시 3. 멀티라벨 분류
```json
# 멀티클래스 분류
{
    "instruction": "이 글에는 어떠한 종류의 혐오 표현이 있는지 알려주세요.", 
    "input": "이래서 여자는 게임을 하면 안된다 ㅋㅋ ㅂㅅ",
    "positives": ["여성/가족", "악플/욕설"],
    "negatives": ["남성", "성소수자", "인종/국적", "연령", "지역", "종교", "없습니다"]
}


```
## 저장한 모델을 huggingface hub에 올리기
```
# 로그인 안했다면
huggingface-cli login --token your_token

python -m task.push_to_hub data/ iknow-lab/koflan-test-110k-0725
```