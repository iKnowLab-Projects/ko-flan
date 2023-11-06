# pip install gradio_client
from gradio_client import Client
import json
from pprint import pprint



# client = Client("http://210.107.197.58:35010")
client = Client("https://iknow-lab-ko-flan-zero.hf.space/")
result = client.predict(
				"예전에는 주말마다 극장에 놀러갔는데 요새는 좀 안가는 편이에요",	# '입력 내용'
				"댓글 주제를 분류하세요",	# '지시문'
				"영화,드라마,게임,소설",	# '라벨(쉼표로 구분)'
				api_name="/predict",
)[1]
pprint(json.loads(result))