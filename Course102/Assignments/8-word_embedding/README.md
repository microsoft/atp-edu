# 词嵌入知识点

## word2vec
1. 讲解skip-gram的任务，目标数据集的形式 
2. 模型的结构（很简单），学员可以自行尝试更负杂的模型结构
3. 模型很容易收敛，但效果一般，主要原因是训练数据集较小较片面(wordembedding通常用海量通用语料进行训练)


## bertembedding
1. transformers 的简单介绍
2. 如何从本地读取模型(先保存再读取)
3. BERT模型首次实战，讲解模型，重点是模型的输入和输出。
4. 模型输出的含义，哪部分可以作为字embedding
5. 词embedding可以用字embedding做平均。
6. bert的输出是上下文相关的，这点和word2vec非常不同