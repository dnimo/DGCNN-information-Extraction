from __future__ import print_function
import json
import numpy as np
from random import choice
from tqdm import tqdm
import pyhanlp
from gensim.models import KeyedVectors
import re,os, codecs
import keras

mode = 0
char_size = 128
maxlen = 512

word2vec = KeyedVectors.load_word2vec_format('data/financial.word.txt')

id2word = {i+1:j for i,j in enumerate(word2vec.wv.index2word)}
word2id = {j:i for i,j in id2word.items()}
word2vec = word2vec.wv.vectors
word_size = word2vec.shape[1]
word2vec = np.concatenate([np.zeros((1,word_size)),word2vec])

def tokenize(s):
    return [i.word for i in pyhanlp.HanLP.segment(s)]


def seq_padding(X,padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x,[padding]*(ML-len(x))]) if len(x) < ML else x for x in X
        ])

def sent2vec(S):
    V = []
    for s in S:
        V.append([])
        for w in s:
            for _ in w:
                V[-1].append(word2id.get(w,0))
    V = seq_padding(V)
    V = word2vec[V]
    return V

total_data = json.load(open('data/train_data_me.json'))
id2predicate, predicate2id = json.load(open('data/all_50_schemas_me.json'))
id2predicate = {int(i):j for i,j in id2predicate.items()}
id2char, char2id = json.load(open('data/all_chars_me.json'))
num_classes = len(id2predicate)

# 随机打乱数据集
if not os.path.exists('data/random_order_vote.json'):
    random_order = list(range(len(total_data)))
    with open('data/random_order_vote.json', 'w', encoding='utf-8') as f:
        json.dump(random_order, f, indent=4, ensure_ascii=False)
    # json.dump(
    #     random_order,
    #     open('data/random_order_vote.json', 'w', encoding='utf-8'),
    #     indent = 4
    # )
else:
    random_order = json.load(open('data/random_order_vote.json'))

# 创建训练集以及测试集
train_data = [total_data[j] for i,j in enumerate(random_order) if i % 8 != mode]
# dev_data = [total_data[j] for i,j in enumerate(random_order) if i % 8 == mode]

predicates = {}
# predicates字典格式，{predicate:[(subject, predicate, object)]}

# 添加抽取的目标关系类型
for d in train_data:
    for sp in d['spo_list']:
        if sp[1] not in predicates:
            predicates[sp[1]] = []
        predicates[sp[1]].append(sp)

# 随机生成输入格式
def random_generate(d, spo_list_key):
    r = np.random.random()
    if r > 0.5:
        return d
    else:
        k = np.random.randint(len(d[spo_list_key]))
        spi = d[spo_list_key][k]
        k = np.random.randint(len(predicates[spi[1]]))
        spo = predicates[spi[1]][k]
        F = lambda s:s.replace(spi[0], spo[0]).replace(spi[2], spo[2])
        text = F(d['text'])
        spo_list = [(F(sp[0]), sp[1], F(sp[2])) for sp in d[spo_list_key]]
        return {'text': text, spo_list_key: spo_list}

class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            T1, T2, S1, S2, K1, K2, O1, O2 = [], [], [], [], [], [], [], []
            for i in idxs:
                spo_list_key = 'spo_list'
                d = random_generate(self.data[i], spo_list_key)
                text = d['text'][:maxlen]
                text_words = tokenize(text)
                text = ''.join(text_words)
                items = {}
                for sp in d[spo_list_key]:
                    subjectid = text.find(sp[0])
                    object = text.find(sp[2])
                    if subjectid != -1 and object != -1:
                        key = (subjectid, subjectid+len(sp[0]))
                        if key not in items:
                            items[key] = []
                        items[key].append(
                            (object, object+len(sp[2]), predicate2id[sp[1]])
                        )
                if items:
                    T1.append([char2id.get(c, 1) for c in text])
                    T2.append(text_words)
                    s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1]-1] = 1
                    k1, k2 = np.array(items.keys()).T
                    k1 = choice(k1)
                    k2 = choice(k2[k2 >= k1])
                    o1, o2 = np.zeros((len(text), num_classes)), np.zeros((len(text)), num_classes)
                    for j in items.get((k1, k2), []):
                        o1[j[0]][j[2]] = 1
                        o2[j[1]-1][j[2]] = 1
                    S1.append(s1)
                    S2.append(s2)
                    K1.append([k1])
                    K2.append([k2-1])
                    O1.append(o1)
                    O2.append(o2)
                    if len(T1) == self.batch_size or i == idxs[-1]:
                        T1 = seq_padding(T1)
                        T2 = sent2vec(T2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        O1 = seq_padding(O1, np.zeros(num_classes))
                        O2 = seq_padding(O2, np.zeros(num_classes))
                        K1, K2 = np.array(K1), np.array(K2)
                        yield [T1, T2, S1, S2, K1, K2, O1, O2], None
                        T1, T2, S1, S2, K1, K2, O1, O2, = [], [], [], [], [], [], [], []

train_D = data_generator(train_data)
