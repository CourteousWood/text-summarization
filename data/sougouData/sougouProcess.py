import jieba
from string_repalce import pyltp_model, preprocess

titles = []
contents = []
title = ''
content = ''
model = pyltp_model()

with open('/Users/didi/Downloads/corpus.txt', 'r', encoding='utf-8') as lines:
    for num, line in enumerate(lines):
        if len(titles) % 1000 == 0:
            print(num, len(titles))
        cur_num = num % 6
        if cur_num == 3:
            title = line[14:-16]
            title = title.replace('\u3000', ' ')
            title = preprocess(title, model)
        if cur_num == 4:
            content = line[9:-11]
            content = content.replace('\u3000', ' ')
            content = preprocess(content, model)
        if cur_num == 0:
            if len(title) == 0 or len(content) == 0:
                continue
            else:
                titles.append(title)
                contents.append(content)
                title = ''
                content = ''
        if len(titles) > 500000:
            break
total_datas = titles + contents
vocab = dict()
for i in total_datas:
    for j in i:
        if j not in vocab:
            vocab[j] = 0
        else:
            vocab[j] += 1
del_keys = []
for i, j in vocab.items():
    if j < 5:
        del_keys.append(i)
for i in del_keys:
    vocab.pop(i)

str2id = {i: num + 4 for num, i in enumerate(vocab)}
str2id['PAD'] = 0
str2id['UNK'] = 1
str2id['GO'] = 2
str2id['EOS'] = 3
id2str = {j: i for i, j in str2id.items()}
import pickle

temp = {}
temp['str2id'] = str2id
temp['id2str'] = id2str

with open('save.pkl', 'wb') as f:
    pickle.dump(temp, f)

titles_id = [[str2id[j] if j in str2id else str2id['UNK'] for j in i]for i in titles]
contents_id = [[str2id[j] if j in str2id else str2id['UNK'] for j in i] for i in contents]

with open('../datas/train/titles.txt', 'w', encoding='utf-8') as f:
    for i in titles:
        f.write(" ".join(i) + '\n')

with open('../datas/train/titles_id.txt', 'w', encoding='utf-8') as f:
    for i in titles_id:
        print(i)
        i = [str(m) for m in i]
        f.write(" ".join(i))
        f.write('\n')

with open('../datas/train/contents_id.txt', 'w', encoding='utf-8') as f:
    for i in contents_id:
        i = [str(m) for m in i]
        f.write(" ".join(i) + '\n')

with open('../datas/train/contents.txt', 'w', encoding='utf-8') as f:
    for i in contents:
        f.write(" ".join(i) + '\n')
