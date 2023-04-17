# 知识点：

## 0.BertTextSimilarity
1. 介绍下什么是文本相似度，句子相似度是文本相似度中最常用的（段落太难，词没太大义意）。
2. 回顾最简单的句子相似度计算方法， Bert+mean pooling+ l2 distance， 从例子来看效果还可以。
3. 这种方式的缺陷：距离只有相对义意，没有绝对义意。 
4. 嵌入是语言层级的，在特定任务上效果不一定好
4. 使用transormers里的预训练模型计算，这种方式可以快速上手！ tranformer种embedding模型都不支持tf，这里找了个pytorch的例子
5. sentence_bert 使用cosine距离代替l2距离，这里的距离更具解释性。
6. 介绍阿里问答对数据集，后面都会使用这个数据集进行模型调优。训练集，测试集种的正负例数量
7. 计算sentence_bert在阿里数据集上的效果

## 1.0TrainTextSimilarityModelBasic，1.1，1.2
1. 通过阿里的训练集对模型进行fine-tune。这里1.0 1.1 1.2 都是对模型进行fine-tune，但是采用了不同的方式，对比结果。

### 1.0
    1. 不会直接fine-tune BERT模型本身，而是在bert模型后再加一层dense层从新计算embedding。这样可以train的模型量较少，训练速度快，占用内存少。
    2. 使用cosine+sigmoid作为分类的head，这部分没有参数可调。注意cosine进入sigmoid时`*10`的义意（让模型能更好的收敛）
    3. 介绍如何构建自定义的keras layer，loss
    4. 注意transformers TFBERTModel调用参数顺序，bert_model(input_ids,attention_mask,token_type_ids),当然可以使用kwargs的方式

### 1.1 
    1. 1.1 和1.0的区别是不填将dense层，直接使用mean pooling作为embedding，相应的fine-tune时会调整BERT模型本身
    2. 比 1.0 的learning-rate更小，因为在调整已经训练模型时应更谨慎。
    3. 效果会更好一些

### 1.2
    1. 使用sentence bert的分类方法，这种方法被证明比较有效。
    2. 讲解下模型结构concat（embed1，embed2，embed1-embed2）+ dense
    3. 这里介绍的都是双塔模型，为什么？（因为embedding可以缓存，如果时但模型每次都要从新用bert inference）


### TrainTextSimilarityModelSiamese
    1. 这个模型时模拟只用正例的情况（实践中常见），可以作为课程可选内容。