# -*- coding: utf-8 -*-
'''
该脚本用于将搜狗语料库新闻语料
转化为按照URL作为类别名、
content作为内容的txt文件存储
'''
import os
import re

'''生成原始语料文件夹下文件列表'''
def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
                listdir(file_path, list_name)
        else:
                list_name.append(file_path)

'''字符数小于这个数目的content将不被保存'''
threh = 30
'''获取所有语料'''
list_name = []
listdir('data/',list_name)

'''对每个语料'''
for path in list_name:
    print(path)
    file = open(path, 'rb').read().decode("utf8")

    '''
    正则匹配出url和content
    '''
    patternURL = re.compile(r'<url>(.*?)</url>', re.S)
    patternCtt = re.compile(r'<content>(.*?)</content>', re.S)

    classes = patternURL.findall(file)
    contents = patternCtt.findall(file)

    '''
    # 把所有内容小于30字符的文本全部过滤掉
    '''
    for i in range(contents.__len__())[::-1]:
        if len(contents[i]) < threh:
            contents.pop(i)
            classes.pop(i)

    '''
    把URL进一步提取出来，只提取出一级url作为类别
    '''
    for i in range(classes.__len__()):
        patternClass = re.compile(r'http://(.*?)/',re.S)
        classi = patternClass.findall(classes[i])
        classes[i] = classi[0]

    '''
    按照RUL作为类别保存到samples文件夹中
    '''
    for i in range(classes.__len__()):
        file = 'samples/' + classes[i] + '.txt'
        f = open(file,'a+',encoding='utf-8')
        f.write(contents[i]+'\n')   #加\n换行显示