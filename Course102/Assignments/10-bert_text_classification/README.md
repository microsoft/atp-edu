# 知识点：

## 0.SimpleDemo
    1. 可以从transformers的model zoo中找到各种各样的模型，如果恰好和你的需求吻合，那么直接拿来使用即可。 
    2. 用法非常简单

## 1. FinetuneBertForClassfication.ipynb
    0. 使用了头条新闻数据集进行模型训练
    1. 使用了tfhub的BERT模型，使用起来差不多，但是还是有些区别，比如tokenizer部分。
    2. 使用了dropout层
    3. 和实战7任务类似，但是使用了BERT模型。