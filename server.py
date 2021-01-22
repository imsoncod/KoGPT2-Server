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
import boto3

app = Flask(__name__)

#API
@app.route('/<string:name>')
def answer(name):
    q = request.args.get('q')
    if name == 'lamama':
        return jsonify({name : chat(lamama, q)})
    elif name == 'panmingming':    
        return jsonify({name : chat(panmingming, q)})
    elif name == 'pulipy':
        return jsonify({name : chat(pulipy, q)})  

# 딥러닝에 필요한 파라미터들 정의
parser = argparse.ArgumentParser(description='KoGPT2')

# 훈련 데이터 세트로 학습
parser.add_argument('--update',
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
def update():
    name_list = ['lamama', 'panmingming', 'pulipy']
    url = 'https://kogpt2test.s3.ap-northeast-2.amazonaws.com/'
    for name in name_list:
        savename = name + '.params'
        download.urlretrieve(url+savename, savename)

def role_switch():
    sts_client = boto3.client('sts')

    assumed_role_object=sts_client.assume_role(
        RoleArn="arn:aws:iam::248239598373:role/DeveloperRole",
        RoleSessionName="RoleSession1"
    )

    credentials=assumed_role_object['Credentials']

    s3=boto3.client(
        's3',
        aws_access_key_id=credentials['AccessKeyId'],
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )

    return s3

if __name__ == "__main__":
    if opt.update:
        s3 = role_switch()
        name_list = ['lamama', 'panmingming', 'pulipy']
        for name in name_list:
            s3.download_file('kogpt2test', name+'.params', name+'.params')

    tok_path = get_tokenizer()
    model, vocab = get_mxnet_kogpt2_model(ctx=ctx)
    tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)

    #칫챗 모델 List
    lamama = KoGPT2Chat(model)
    panmingming = KoGPT2Chat(model)
    pulipy = KoGPT2Chat(model)

    #모델별 파라미터 Load
    lamama.load_parameters('DataSet/lamama.params', ctx=ctx)
    panmingming.load_parameters('DataSet/panmingming.params', ctx=ctx)
    pulipy.load_parameters('DataSet/pulipy.params', ctx=ctx)

    app.run(host='0.0.0.0')