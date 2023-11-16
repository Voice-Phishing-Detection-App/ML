from KoBERTModel.BERTClassifier import BERTClassifier
from KoBERTModel.BERTDataset import BERTDataset

## importing KoBERT functions

from kobert.utils.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

## importing the required packages
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
# from keras.models import Sequential

import os
from time import time
from timeit import default_timer as timer

def run():
    print("---------- train -----------")
    print(os.getcwd())
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # 일반 음성: 0     보이스피싱 음성: 1
    dataset = pd.read_csv('static/csv/KoBERT_dataset_v2.5.csv').sample(frac=1.0)
    dataset.sample(n=15)
    dataset_tsv = []
    for text, label in zip(dataset['Transcript'], dataset['Label']):
        data = []
        data.append(text)
        data.append(str(label))
        dataset_tsv.append(data)
    

    train_set, val_set= train_test_split(dataset_tsv, 
                                test_size=0.2, 
                                random_state=42, 
                                shuffle=True)
    print(f"Numbers of train instances by class: {len(train_set)}")
    print(f"Numbers of val instances by class: {len(val_set)}")

    #net = BERTClassifier().cuda()

    # 파라미터 설정
    max_len = 64 # The maximum sequence length that this model might ever be used with. 
                # Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
    batch_size = 32
    warmup_ratio = 0.1
    num_epochs = 10   # only parameter changed from 5 to 10 compared to the documentation
    max_grad_norm = 1
    log_interval = 200
    learning_rate = 5e-5  # 4e-5

    bertmodel, vocab = get_pytorch_kobert_model() # calling the bert model and the vocabulary

    # 데이터셋 토큰화
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    # BERTDataset 함수에 sklearn을 통해 train & val 나눈 데이터 입력
    train_set = BERTDataset(train_set, 0, 1, tok, max_len, True, False)
    val_set = BERTDataset(val_set, 0, 1, tok, max_len, True, False)

    # torch 형식의 dataset을 만들어 입력 데이터셋의 전처리 마무리
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=5)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=5)

    # BERTClassifier 함수 초기에 환경설정에서 가져온 bertmodel을 불러오고
    # to(device)를 통해 GPU 사용을 설정한 model 함수 정의
    model = BERTClassifier(bertmodel,  dr_rate=0.4).to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # configuration f the optimizer and loss function
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    start_time = time()

    for e in range(num_epochs):
        train_acc = 0.0
        test_acc = 0.0

        # Training of the model with the train set
        model.train()
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            loss = loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

    torch.save(model.state_dict(), 'KoBERTModel/model/train.pt')
    run_time = time() - start_time
    run_time

# 모델의 정확도 측정을 위한 함수
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def get_metrics(pred, label, threshold=0.5):
    pred = (pred > threshold).astype('float32')
    tp = ((pred == 1) & (label == 1)).sum()
    fp = ((pred == 1) & (label == 0)).sum()
    fn = ((pred == 0) & (label == 1)).sum()
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * recall * precision / (precision + recall)
    
    return {
        'recall': recall,
        'precision': precision,
        'f1': f1
    }
