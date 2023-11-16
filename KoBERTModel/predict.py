import torch
import gluonnlp as nlp
import numpy as np

from torch.utils.data import Dataset, DataLoader
from KoBERTModel.BERTDataset import BERTDataset
from KoBERTModel.BERTClassifier import BERTClassifier
from kobert.utils.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model


model = None
bertmodel, vocab = get_pytorch_kobert_model() # calling the bert model and the vocabulary
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_model():
    global model
    model = BERTClassifier(bertmodel,  dr_rate=0.4).to(device)

    model.load_state_dict(torch.load('KoBERTModel/model/train.pt'), strict = False)
    model.eval() 

def load_dataset(predict_sentence):
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data = [predict_sentence, '0']
    dataset_another = [data]
    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len=64, pad=True, pair=False)
    return DataLoader(another_test, batch_size = 32, num_workers = 5) # torch 형식 변환

def inference(predict_sentence): # input = 보이스피싱 탐지하고자 하는 sentence
    print("※ KoBERT 추론 시작 ※")

    test_dataloader = load_dataset(predict_sentence)
    
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        result = False
        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("일반 음성 전화")
            elif np.argmax(logits) == 1:
                test_eval.append("보이스피싱 전화")
                result = True

        print("▶ 입력하신 내용은 '" + test_eval[0] + "' 입니다.")
        return result

def run(text):
    load_model()
    return inference(text)