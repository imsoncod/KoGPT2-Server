# -*- coding: utf-8 -*-
import argparse
import logging
import math

import gluonnlp as nlp
import mxnet as mx
import pandas as pd
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.mxnet_kogpt2 import get_mxnet_kogpt2_model
from kogpt2.utils import get_tokenizer
from mxnet import gluon, nd
from mxnet.gluon import nn

from flask import Flask, request, jsonify
from urllib import request as download

app = Flask(__name__)

#API
@app.route('/<int:cno>')
def answer(cno):
    q = request.args.get('q')
    if cno == 1:
        return jsonify({'1번 봇' : chat(chatbot1, q)})
    elif cno == 2:    
        return jsonify({'2번 봇' : chat(chatbot2, q)})
    elif cno == 3:
        return jsonify({'3번 봇' : chat(chatbot3, q)})  

# 딥러닝에 필요한 파라미터들 정의
parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

# 훈련 데이터 세트로 학습
parser.add_argument('--reload',
                    action='store_true',
                    default=False)                    

opt = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<Question>'
S_TKN = '<Answer>'
BOS = '<s>'
EOS = '</s>'
MASK = '<unused0>'
# SENT = '<unused1>'


class KoGPT2Chat(nn.HybridBlock):
    def __init__(self, kogpt2, prefix=None, params=None):
        super(KoGPT2Chat, self).__init__(prefix=prefix, params=params)
        self.kogpt2 = kogpt2

    def hybrid_forward(self, F, inputs):
        # (batch, seq_len, hiddens)
        output, _ = self.kogpt2(inputs)
        return output


if mx.context.num_gpus() > 0:
    # 0번 GPU를 사용하겠다(해당 그래픽카드의 메모리를 지정)
    ctx = mx.gpu()
else:
    ctx = mx.cpu()

def chat(model, sentence):
    q = sentence
    #q = input('사용자 입력 : ').strip()
    if q == 'quit':
        return
    q_tok = tok(q)
    #print('q_tok : ', q_tok)
    a = ''
    a_tok = []
    ##############
    #Predict 과정#
    ##############
    while 1:
        input_ids = mx.nd.array([vocab['<Question>']] + vocab[q_tok] +
                                vocab['</s>', '<unused1>'] + 
                                #vocab[sent_tokens] +
                                vocab['</s>', '<Answer>'] +
                                vocab[a_tok]).expand_dims(axis=0)
        pred = model(input_ids.as_in_context(ctx))
            
        gen = vocab.to_tokens(
            mx.nd.argmax(
                pred,
                axis=-1).squeeze().astype('int').asnumpy().tolist())[-1]
        #print('gen', gen)
        if gen == EOS:
            break
        a += gen.replace('▁', ' ')
        #print('lsa : ', a)
        a_tok = tok(a)
        #print('a_tok : ', tok(a))
    #print("챗봇 응답 : {}".format(a.strip()))    
    return a.strip()

#모델 파라미터 다운로드(현재는 git에서 다운로드)
def reload():
    url = 'https://kogpt2test.s3.ap-northeast-2.amazonaws.com/'
    for i in range(1, 4):
        savename = str(i)+'.params'
        download.urlretrieve(url+savename, savename)      

if __name__ == "__main__":
    if opt.reload:
        reload()

    tok_path = get_tokenizer()
    model, vocab = get_mxnet_kogpt2_model(ctx=ctx)
    tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)

    #칫챗 모델 List
    chatbot1 = KoGPT2Chat(model)
    chatbot2 = KoGPT2Chat(model)
    chatbot3 = KoGPT2Chat(model)

    #모델별 파라미터 Load
    chatbot1.load_parameters('1.params', ctx=ctx)
    chatbot2.load_parameters('2.params', ctx=ctx)
    chatbot3.load_parameters('3.params', ctx=ctx)

    app.run(host='0.0.0.0')