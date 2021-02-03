#! -*- coding:utf-8 -*-

import json
from tqdm import tqdm
import codecs

all_50_schemas = []

with open('data/example_50.json','r',encoding='utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l)
        all_50_schemas.extend(a['qas'])

# 整理之后的json文档
with codecs.open('data/test.json','w',encoding='utf-8') as f:
    json.dump(all_50_schemas, f, indent=4, ensure_ascii=False)

chars = {}
min_count = 2

# 创建训练数据集
train_data = []
with open('data/sentence_800') as f:
    for l in tqdm(f):
        a = json.loads(l)
        if not a['spo_list']:
            continue
        train_data.append(
            {
                'text':a['text'],
                'spo_list' : [(i['subject'], i['predicate'], i['object']) for i in a['spo_list']]
            }
        )
        for c in a['text']:
            chars[c] = chars.get(c, 0) + 1

with codecs.open('data/train_data_me.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)

# 创建自己的测试数据集 dev_data.json