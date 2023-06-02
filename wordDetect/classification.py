from konlpy.tag import Okt
import pandas as pd
okt = Okt()

safe_type = 0

class voice:
    def __init__(self):

        self.df = pd.read_csv("static/csv/500_가중치.csv", encoding='utf-8')             # 전체 형태소 분석 (가중치) 파일 
        self.type_df = pd.read_csv("static/csv/type_token_가중치.csv", encoding='utf-8') # 범죄 유형 분류 기준 단어 파일
        
        self.cnt = 1        # 보이스피싱 확률 변수
        self.type1_cnt = 1  # 대출사기형 확률
        self.type2_cnt = 1  # 수사기관사칭형 확률        
        self.text = ''      # 음성에서 변환된 텍스트
        self.token_dict = {} # 단어:횟수 딕셔너리 생성
    
    def getLevel(self, text):
        print('\n※ 위험도 검사 시작 ※')
        self.text = text
        self.ing_cnt() # 피싱 탐지 함수 호출

    def detection(self):            
        self.token_ko = pd.DataFrame(okt.pos(self.text), columns=['단어', '형태소'])
        self.token_ko = self.token_ko[(self.token_ko['단어'].str.len() > 1)&(self.token_ko.형태소.isin(['Noun', 'Adverb']))]
            
        for i in self.token_ko.단어.values:
            if i in self.df.단어.values:
                self.cnt *= float(self.df.loc[self.df.단어==i, '확률'])
                if i not in self.token_dict:
                    self.token_dict[i] = 1
                else:
                    self.token_dict[i] = self.token_dict.get(i) + 1 
    
        if self.cnt > 100:
            self.cnt = 100  # 확률이 100%를 넘겼을 경우 100으로 초기화
            
    # 유형을 분류하는 함수 
    def categorizing(self):
        self.token_df = pd.DataFrame(zip(self.token_dict.keys(),self.token_dict.values()), columns=['의심 단어', '횟수'])
        self.token_df = self.token_df.sort_values(by='횟수', ascending=False)
        
        for i, x in zip(self.token_df['의심 단어'].values, self.token_df['횟수'].values):
            if i in self.type_df.type1_단어.values:
                self.type1_cnt *= float(self.type_df.loc[self.type_df.type1_단어==i, 'type1_확률']) ** x
            elif i in self.type_df.type2_단어.values:
                self.type2_cnt *= float(self.type_df.loc[self.type_df.type2_단어==i, 'type2_확률']) ** x
                
        if self.type1_cnt > self.type2_cnt:
            return '대출사기형'
        else:
            return '수사기관사칭형'
                        
    # 결과를 출력하는 함수
    def ing_cnt(self):
        self.detection() # 분석 함수 호출
        global safe_type
        
        if self.cnt <=20: safe_type = 0 # 0단계 : 안전
        elif self.cnt <= 40: safe_type = 1 # 1단계 : 의심
        elif self.cnt <= 60: safe_type = 2 # 2단계 : 경고
        else: safe_type = 3 # 3단계 : 위험
        
        bolded_safe_type = "\033[1m" + str(safe_type) + "\033[0m"

        print(f'▶ 보이스피싱 확률 : {self.cnt:.2f}% [{bolded_safe_type}]')

    
def run(text):
    v = voice()
    v.getLevel(text)
    return safe_type