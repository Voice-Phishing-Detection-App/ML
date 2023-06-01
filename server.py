from flask import Flask, jsonify, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os, warnings
from KoBERT import train

app = Flask(__name__)
CORS(app)
warnings.filterwarnings(action='ignore')

@app.route("/", methods=['POST', 'GET'])
def main():
    train.run()
    return render_template("main.html")

if __name__=="__main__":
  app.run(debug=True)