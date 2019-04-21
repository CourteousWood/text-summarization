# 基于中文的文本摘要
	基于seq2seq+attention+beamsearch，进行搜狗新闻数据的文本摘要。
	本电脑的环境为 TensorFlow 1.10.1
	Gpu: p40
## 1，数据的准备。
   采用了搜狗实验室的文本数据，下载到本地，如果操作系统是Ubuntu 或者 mac系统，要对下载的数据进行重新编码，否则数据的格式总是 乱码。
	cat news_sohusite_xml.dat | iconv -f gbk -t utf-8 -c  > corpus.txt

### 1.1 数据的清洗。
		`/data/sougouData` 数据处理，包括几个部分。采用正则化或者ltp工具包进行处理。
			- 1，时间的替换，
			- 2，数字的替换，
			- 3，数据类型的替换（人名，地名，组织名）。
### 1.2 数据替换后，会生成两个文件 `/data/train/content.txt和 data/train/title.txt`，然后对这两个数据进行str to id的操作， 同时 生成 字典。
		/data/下 执行 gen_vocab.py 和process.py 完成上述操作。在train/下面生成 content_id.txt 和title_id.txt 文件。

## 2，模型训练
   执行python train.py 进行训练。
## 3，模型预测
   执行python test.py 完成预测。

## 4，模型说明。
	由于新闻的文本数据特别长，所以在训练模型时，模型的训练速度实在是太慢了。
	之前，认识一个人，他也是做文本摘要的，在做生成式摘要时，训练花费的时间是特别长的， 半个月，一个月之类的。

## 5，模型参考
	- 1，大神的英文文本摘要吧，由于文本长度很短，这个特别容易实现的，我也是主要借鉴的。
[连接](https://github.com/dongjun-Lee/text-summarization-tensorflow)
	- 2, 本人数据集主要是 来源于搜狗的数据平台。 
   http://www.sogou.com/labs/resource/ca.php
