# 보이스피싱 탐지[피노키오] - Machine Learning Server

<h3>🧸 KoBERT 모델로 보이스피싱 탐지</h3>

<h4>📍Dataset</h4>

- 보이스 피싱 텍스트: [금융감독원 - 보이스피싱 체험관](https://www.fss.or.kr/fss/bbs/B0000203/list.do?menuNo=200686)

- 일반 대화 텍스트: [AI Hub - 감정분류를 위한 대화 음성 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=263)

- `일반대화: 0`, `보이스피싱: 1` 으로 라벨링


<h4>📍Dataset Train</h4>

- KoBERT/train.py
- KoBERT/model/train.pt : train model 저장

<h4>📍Text Inference</h4>

1. 스프링 서버 요청받기

   json형식으로 STT 결과값 text를 요청 받는다
 
```json
{
    "text": "네네 지금 보시면 이제 신용 등급 부분하고는 관계없이 저희쪽에서 연 이자 10프로 이하대에 저금리 대안상품으로 바꿔드림론 이런 상품으로 정부지원상품으로 지원차 연락을 좀 드린거고 현재 이용중이신 고금리건 어디어디 이용중이시죠"
}
```

2. KoBert 모델 추론

- 결과가 True인 경우 - 보이스 피싱 O

  [단어 기반 보이스피싱 위험도 측정 코드 실행](#🧸-단어-기반-보이스피싱-위험도-측정)

- 결과가 False인 경우 - 보이스 피싱 X

  아래와 같은 형태로 스프링 서버에 응답
  
  ```json
  { "phishing" : "false", "level" : "0" }
  ```

---
# <h3>🧸 단어 기반 보이스피싱 위험도 측정</h3>

<h4>📍보이스피싱 관련 단어 파일</h4>

- static/csv/

  500_가중치.csv
  type_token_가중치.csv

1. 위험도 측정

- wordDetect/classification

  20% 이하 : 0단계 [안전]
  40% 이하 : 1단계 [의심]
  60% 이하 : 2단계 [경고]
  나머지 : 3단계 [위험]

2. 최종 분석결과 응답

```json
{ 
  "phishing": "true", "level": 3
}
```
1, 2, 3단계가 감지될 경우 사용자에게 알림 서비스를 지원한다.