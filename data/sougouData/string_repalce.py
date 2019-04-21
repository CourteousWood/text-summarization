import re

import os
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer


class pyltp_model():
    def __init__(self, LTP_DATA_DIR='/Users/didi/Desktop/ltp_data_v3.4.0'):
        cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
        pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
        ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        self.segmentor = Segmentor()  # 初始化实例
        self.postagger = Postagger()  # 初始化实例
        self.recognizer = NamedEntityRecognizer()  # 初始化实例

        self.segmentor.load(cws_model_path)  # 加载模型
        self.postagger.load(pos_model_path)  # 加载模型
        self.recognizer.load(ner_model_path)  # 加载模型

    def token(self, sentence):
        words = self.segmentor.segment(sentence)  # 分词
        words = list(words)
        postags = self.postagger.postag(words)  # 词性标注
        postags = list(postags)
        netags = self.recognizer.recognize(words, postags)  # 命名实体识别
        netags = list(netags)
        result = []
        for i, j in zip(words, netags):
            if j in ['S-Nh', 'S-Ni', 'S-Ns']:
                result.append(j)
                continue
            result.append(i)
        return result

    def close(self):
        self.segmentor.release()
        self.postagger.release()
        self.recognizer.release()  # 释放模型


#
# n = pyltp_model()
# s1, s2, s3 = n.token('元芳你怎么看')
# print(s1)
# print(s2)
# print(s3)
# n.close()


def findTime(line, flag='TAGDATE', flag_1='TAGNUM'):
    # 1 时间替换 2019年月日
    pattern = re.compile(r'[\uff10-\uff19]{4}年[\uff10-\uff19]{1,2}月－[\uff10-\uff19]{1,2}月')
    line = re.sub(pattern, flag, line)

    # 1 时间替换 2019年月日
    pattern = re.compile(r'[[\uff10-\uff19]{1,2}月[\uff10-\uff19]{1,2}日.[[\uff10-\uff19]{1,2}日')
    line = re.sub(pattern, flag, line)

    # 1 时间替换 2019年月日
    pattern = re.compile(r'[\uff10-\uff19]{4}年[\uff10-\uff19]{1,2}月[\uff10-\uff19]{1,2}日')
    line = re.sub(pattern, flag, line)

    # 2 时间替换 2019年月
    pattern = re.compile(r'[\uff10-\uff19]{4}年[\uff10-\uff19]{1,2}月')
    line = re.sub(pattern, flag, line)

    # 3 时间替换 2019年
    pattern = re.compile(r'[\uff10-\uff19]{4}年')
    line = re.sub(pattern, flag, line)

    # 4 时间替换 月 日
    pattern = re.compile(r'[\uff10-\uff19]{1,2}月[\uff10-\uff19]{1,2}日')
    line = re.sub(pattern, flag, line)

    # 4 时间替换 日
    pattern = re.compile(r'[\uff10-\uff19]{1,2}日')
    line = re.sub(pattern, flag, line)

    # 1 时间替换 2019年月日
    pattern = re.compile(r'[\uff10-\uff19]{4}－[\uff10-\uff19]{1,2}－[\uff10-\uff19]{1,2}')
    line = re.sub(pattern, flag, line)

    pattern = re.compile(r'[\uff10-\uff19]{1,} ．[\uff10-\uff19]{1,} ％')
    line = re.sub(pattern, flag_1, line)

    pattern = re.compile(r'[\uff10-\uff19]{1,} ．[\uff10-\uff19]{1,}')
    line = re.sub(pattern, flag_1, line)

    pattern = re.compile(r'[\uff10-\uff19]{1,} ％')
    line = re.sub(pattern, flag_1, line)

    pattern = re.compile(r'[\uff10-\uff19]{1,}％')
    line = re.sub(pattern, flag_1, line)

    pattern = re.compile(r'[\uff10-\uff19]{1,}')
    line = re.sub(pattern, flag_1, line)



    return line


def preprocess(line, model):
    line = findTime(line)
    line = model.token(line)
    return line


if __name__ == '__main__':
    s = [
        '２０１１年７月５日出生的他，患有先天性心脏病、疝气，一出生便被遗弃。２０１２年１月８日，才５个月大的永康被发现呼吸困难，随后送往医院进行抢救治疗，病情稳定后于１月２８日出院。＃玻埃保材辏苍拢保澈牛永康在思源焦点公益基金的帮助下在医院接受手术治疗，术后仅８天']
    s = preprocess(s)
    print(s)
