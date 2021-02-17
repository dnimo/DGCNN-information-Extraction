import json
from tqdm import tqdm
import codecs

train = []

with open('data/example_50.json', 'r', encoding='utf-8') as f:
    for l in tqdm(f):
        a = json.loads(l)
        for doc in a['document']:
            for i in a['qas']:
                tmp = []
                for j in i:
                    for k in j['answers']:
                        tmp.append(k['text'])
            train.append(
                {
                    'text': doc['text'],
                    'spo_list': tmp
                }
            )

print(train)

# 整理之后的json文档
with codecs.open('data/test00.json', 'w', encoding='utf-8') as f:
    json.dump(train, f, indent=4, ensure_ascii=False)