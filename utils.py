from nltk.tokenize import word_tokenize
import re
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import random
import nltk
#nltk.download('punkt')

train_article_path = "data/train/contents_id.txt"
train_title_path = "data/train/titles_id.txt"
valid_article_path = "data/train/valid.article.filter.txt"
valid_title_path = "data/train/valid.title.filter.txt"
article_max_len = 250
summary_max_len = 30

def clean_str(sentence):
    sentence = re.sub("[#.]+", "#", sentence)
    return sentence


def get_text_list(data_path, toy):
    with open (data_path, "r", encoding="utf-8") as f:
        if not toy:
            return [clean_str(x.strip()) for x in f.readlines()]
        else:
            return [clean_str(x.strip()) for x in f.readlines()][:50000]

def shuffle(x, y, shuffle=True):
    '''
    import random
    train_x = list(range(100))
    train_y = list(range(100))
    # 打乱方式 一
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(train_x)
    random.seed(randnum)
    random.shuffle(train_y)
    # for i,j in zip(train_x,train_y):
    #     print(i,j)

    # 打乱方式 二
    zipList = [i for i in zip(train_x, train_y)]
    random.shuffle(zipList)
    train_x[:], train_y[:] = zip(*zipList)
    # for i,j in zip(train_x,train_y):
    #     print(i,j)
    '''
    if shuffle == True:
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(x)
        random.seed(randnum)
        random.shuffle(y)

def get_text_list1(flag='dev'):
    # 50w， 训练集 设置为 49w
    #       测试集 为 1w
    print(flag)
    if flag == "train":
        line = (0, 1000)
        contentFile, titleFile = train_article_path, train_title_path
    else:
        line = (1000, 2000)
        contentFile, titleFile  = train_article_path, train_title_path
    contents = []
    titles = []
    # encode ,decode数据之间不同的处理逻辑， decode的 input,output,都会加一个特殊标志符，
    # 而encode则不需要。
    with open(contentFile, 'r', encoding='utf-8') as f:
        for num, i in enumerate(f):
            if (num >= line[0]) and (num <= line[1]):
                # print(i)
                i = i.strip()
                val = i.split(' ')
                val = [int(i) for i in val]
                cur_len = article_max_len if len(val) > article_max_len else len(val)
                # valu = val[:article_max_len]+[0]*(article_max_len-cur_len)
                valu = val[:cur_len] + [0]*(article_max_len-cur_len)
                contents.append(valu)
    with open(titleFile, 'r', encoding='utf-8') as f:
        for num, i in enumerate(f):
            if (num >= line[0]) and (num <= line[1]):
                # print(i)
                i = i.strip()
                val = i.split(' ')
                val = [int(i) for i in val]
                cur_len = summary_max_len-1 if len(val) > summary_max_len else len(val)
                valu = val[:cur_len]
                titles.append(valu)
    return contents, titles


def build_dict():

    word_dict = dict()

    with open('data/vocab', 'r') as f:
        for num, i in enumerate(f):
            word_dict[i.strip()] = num

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    return word_dict, reversed_dict, article_max_len, summary_max_len

def build_dataset(step, word_dict, article_max_len, summary_max_len, toy=False):
    if step == "train":
        article_list = get_text_list(train_article_path, toy)
        title_list = get_text_list(train_title_path, toy)
    elif step == "valid":
        article_list = get_text_list(valid_article_path, toy)
    else:
        raise NotImplementedError

    x = [word_tokenize(d) for d in article_list]
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
    
    if step == "valid":
        return x
    else:        
        y = [word_tokenize(d) for d in title_list]
        y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
        y = [d[:(summary_max_len - 1)] for d in y]
        return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    # inputs = np.array(inputs)
    # outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        shuffle(inputs, outputs, shuffle=True)
        for batch_num in range(num_batches_per_epoch):
            if batch_num % 10 == 0:
                print("当前程序运行到：第{}轮  第{}个".format(epoch, batch_num))
            if batch_num == num_batches_per_epoch-1:
                continue
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def get_init_embedding(reversed_dict, embedding_size):
    glove_file = "glove/glove.42B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)
