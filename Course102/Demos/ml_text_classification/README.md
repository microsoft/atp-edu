# 机器学习文本分类

## 目标：

1. 学员掌握基础的文本处理方法，利用vsm模型等方法进行向量化
2. 学员利用sklearn训练文本分类模型，调整超参数比较

## 使用：

1. 讲解 feature_engineering.py 中的encoder思路，由学员自行实现encoder
2. 学员使用 model_training.py 中的model_train进行调参（调试encoding方式，调试模型和模型的参数）

## Feature Engineering
1. N_gram or 分词
2. 去除停用词
3. TfIdf or one-hot 
4. 学员自行实现其他encoder方法

## Model Training
1. 分词结果好于ngram
2. 通过PCA对高维数据降维至4096维训练效果较好
3. logistic regression 效果最好
4. decision tree 在训练集上极易过拟合，需要指定最大深度，泛化能力不强
5. 还可以尝试Naive bayes，GDBT，etc