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

from urllib import request as download

# 딥러닝에 필요한 파라미터들 정의
parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

# 전체 데이터 반복학습 횟수
parser.add_argument('--num-epoch',
                    type=int,
                    default=1,
                    help='number of iterations to train (default: 2)')

#입력으로 들어오는 문자의 최대 길이
parser.add_argument('--max-seq-len',
                    type=int,
                    default=32,
                    help='max sentence length on input (default: 32)')

# 1 iteration에 사용되는 데이터 개수
parser.add_argument('--batch-size',
                    type=int,
                    default=64,
                    help='batch size for training (default: 64)')

# 사용자의 요청에 응답
parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

# 감정을 나타내는 번호
# parser.add_argument('--sentiment',
#                     type=str,
#                     default='0',
#                     help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

# 채팅시작을 위한 모델 파라미터
parser.add_argument('--model_params',
                    type=str,
                    default='kogpt2_chat.params',
                    help='model binary for starting chat')

# 훈련 데이터 세트로 학습
parser.add_argument('--reload',
                    action='store_true',
                    default=False)                    

# 배치 사이즈가 커져도 동일한 성능을 내기 위한 gradient accumulate 설정
parser.add_argument('--accumulate',
                    type=int,
                    default=1,
                    help='accumulate gradient to achieve the same result with a large batch size')

opt = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<Question>'
S_TKN = '<Answer>'
BOS = '<s>'
EOS = '</s>'
MASK = '<unused0>'
# SENT = '<unused1>'


class ChatDataset(gluon.data.Dataset):
    def __init__(self, chats, tok_path, vocab, max_len=32):
        self._data = chats
        self._tok_path = tok_path
        self.tokenizer = None
        self.first = True
        self.q_token = '<Question>'
        self.a_token = '<Answer>'
        # self.sent_token = '<unused1>'
        self.bos = '<s>'
        self.eos = '</s>'
        self.maskt = '<unused0>'
        self.vocab = vocab
        self.max_len = max_len
        self.padder = nlp.data.PadSequence(
            max_len, pad_val=self.vocab[self.vocab.padding_token])

    def _activate_sp(self):
        self.tokenizer = nlp.data.SentencepieceTokenizer(self._tok_path, 0, 0)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self.tokenizer is None:
            self._activate_sp()
        turn = self._data.iloc[idx]
        # turn : Q : 내 마음을 알아줬으면 A : 말을 해야 알거예요. label : 0
        q = turn['Q'] # 내 마음을 알아줬으면
        a = turn['A'] # 말을 해야 알거예요
        # sentiment = str(turn['label']) # 0
        q_toked = [
            self.q_token,
        ] + self.tokenizer(q) + [
            self.eos,
        ] + [
            self.eos,
        ]
        #[self.sent_token] + self.tokenizer(sentiment)
        
        # q_toked : ['<Question>', '▁이별', '▁4', '일', '차', '</s>', '<unused1>', '▁1', '</s>']
        q_len = len(q_toked)
        a_toked = [
            self.a_token,
        ] + self.tokenizer(a) + [
            self.eos,
        ]
        # a_toked : ['<Answer>', '▁아직', '▁실', '감이', '▁안나', '겠어요', '.', '</s>']
        a_len = len(a_toked)
        # 요청 길이 + 응답 길이 > max 길이 일 경우
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [<mask>, <mask>, ...., <mask>, ..., A.. <eos>, <pad>....]
        labels = [
            self.maskt,
        ] * q_len + a_toked[1:]
        if self.first:
            #logging.info("contexts : {}".format(q))
            #logging.info("toked ctx: {}".format(q_toked))
            #logging.info("response : {}".format(a))
            #logging.info("toked response : {}".format(a_toked))
            #logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        return (self.padder(self.vocab[q_toked + a_toked]), nd.array(mask),
                self.padder(self.vocab[labels]))


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


def train(name):
    tok_path = get_tokenizer()
    # /root/kogpt2/kogpt2_news_wiki_ko_cased_818bfa919d.spiece
    model, vocab = get_mxnet_kogpt2_model(ctx=ctx)
    # model : 실제로 사용되는 tokenizer 모델
    # 참조하는 단어집합

    # tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)

    loadname = name + '.csv'
    savename = name + '.params'
    # 학습데이터(.csv) 파일 Read
    data = pd.read_csv(loadname)

    # default : 32
    max_len = opt.max_seq_len
    train_set = ChatDataset(data, tok_path, vocab, max_len=max_len)
    batch_size = opt.batch_size

    # batch_size : 일반적으로 32,64
    # num_workers : 일반적으로 코어개수의 절반
    train_dataloader = mx.gluon.data.DataLoader(train_set,
                                                batch_size=batch_size,
                                                num_workers=5,
                                                shuffle=True)
    kogptqa = KoGPT2Chat(model)
    kogptqa.hybridize()

    # softmax cross entropy loss for classification
    loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
    loss_function.hybridize()

    num_epochs = opt.num_epoch
    lr = 5e-5
    trainer = gluon.Trainer(kogptqa.collect_params(), 'bertadam', {
        'learning_rate': lr,
        'epsilon': 1e-8,
        'wd': 0.01
    })
    # LayerNorm과 Bias에는 Weight Decay를 적용하지 않는다.
    for _, v in kogptqa.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    params = [
        p for p in kogptqa.collect_params().values() if p.grad_req != 'null'
    ]
    # learning rate warmup
    accumulate = opt.accumulate
    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_examples = len(train_set)
    num_train_steps = int(num_train_examples / step_size * num_epochs)
    warmup_ratio = 0.1
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0
    all_model_params = kogptqa.collect_params()

    log_interval = opt.batch_size
    neg = -1e18
    # Set grad_req if gradient accumulation is required
    if accumulate and accumulate > 1:
        for p in params:
            p.grad_req = 'add'

    for epoch_id in range(num_epochs):
        step_loss = 0
        for batch_id, (token_ids, mask, label) in enumerate(train_dataloader):
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                non_warmup_steps = step_num - num_warmup_steps
                offset = non_warmup_steps / (num_train_steps -
                                             num_warmup_steps)
                new_lr = lr - offset * lr
            trainer.set_learning_rate(new_lr)
            with mx.autograd.record():
                # load data to GPU or GPU
                token_ids = token_ids.as_in_context(ctx)
                mask = mask.as_in_context(ctx)
                label = label.as_in_context(ctx)
                # forward computation
                out = kogptqa(token_ids)
                masked_out = nd.where(
                    mask.expand_dims(axis=2).repeat(repeats=out.shape[2],
                                                    axis=2), out,
                    neg * nd.ones_like(out))
                # loss for responses exincluding MASK and PAD
                ls = loss_function(masked_out, label).sum() / mask.sum()
            # backward computation
            ls.backward()
            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(accumulate if accumulate else 1)
                step_num += 1
                if accumulate and accumulate > 1:
                    # set grad to zero for gradient accumulation
                    all_model_params.zero_grad()
            step_loss += ls.asscalar()
            if step_num % log_interval == 0 and step_num > 0:
                print(
                    '훈련중... [Epoch {} Batch {}/{}] loss={:.4f}, lr={:.10f}, train ppl={:.3f}'
                    .format(epoch_id + 1, batch_id + 1, len(train_dataloader),
                            step_loss / log_interval, trainer.learning_rate,
                            math.exp(step_loss / log_interval)))
                step_loss = 0
    logging.info('모델 생성 완료 {}'.format(savename))
    kogptqa.save_parameters(savename)

if __name__ == "__main__":
    name_list = ['lamama', 'panmingming', 'pulipy']
    for name in name_list:
        train(name)