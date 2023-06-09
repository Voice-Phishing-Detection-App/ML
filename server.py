from flask import Flask, jsonify, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os, warnings
from KoBERT import train
from KoBERT import predict
from wordDetect import classification

app = Flask(__name__)
CORS(app)
warnings.filterwarnings(action='ignore')


@app.route("/train", methods=['POST', 'GET'])
def trainData():
    train.run()
    return jsonify({'host': '127.0.0.1', 'port': '5000', 'status' : '200'})

@app.route("/predict", methods=['POST', 'GET'])
def prediction():
    text = request.json['text']
    print(text)
    # text = "여보세요 안녕하세요 고객님 김종현 대리입니다 아네 대리님 고객님 연락이 너무 안되셔가지고 지금 아 네 제가 너무 바빠서요 아 지금은 좀 연락이 이제 가능하신 건가요 아네네 대리님 그 남의 돈 드시니까 좋으세요 네 무슨말씀이시죠 제가 보니까 제게 상환이 다 안되있더라구요 상환이 안되셨다구요 네 어디가 상환이 안되셨다는 거죠 아 제거 KB국민은행에 상환이 다 안되있던데요 고객센터에서 알아보신 건가요 네 어디에 상환하신거 수요일에 상환하신 아닌데 잠시만요 목요일날 상환하신거 말씀하신거죠 네 그게 지금 고객센터로 넘어가는게 좀 서류가 늦게 들어가서 확인이 안되실수도 있으세요 어떻게 기록삭제가 바로되는데 확인이 안돼요 그쪽에서 기록삭제 하고 납부하셨잖아요"
    # print(os.getcwd())
    output = predict.run(text)
    print("▶ " + str(output))

    result = {}
    if output == True:
        level = classification.run(text)
        print("▶ " + str(level))
        result = { 'phishing' : output, 'level' : level }
    else: result = { 'phishing' : output, 'level' : '0' }

    return jsonify(result)
  

if __name__=="__main__":
  app.run(debug=True)