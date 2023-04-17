# CLUENER 细粒度命名实体识别

数据分为10个标签类别，分别为: 
地址（address），
书名（book），
公司（company），
游戏（game），
政府（goverment），
电影（movie），
姓名（name），
组织机构（organization），
职位（position），
景点（scene）

数据详细介绍、基线模型和效果测评，见 https://github.com/CLUEbenchmark/CLUENER

技术讨论或问题，请项目中提issue或PR，或发送电子邮件到 ChineseGLUE@163.com

测试集上SOTA效果见榜单：www.CLUEbenchmark.com

## 1.0 
1. 使用bert做字分类任务。
2. NER的数据集构建比较复杂，需要详细讲解
3. BERT的embedding本身就是上下文embedding，所以效果还不错

## 1.1
1. 按词进行分类仍可能出现一些无效的token，可以使用crf学习标签之间的约束(crf会建模yi对yi+1的影响，转移概率)
2. crf的细节这里不讲，但是这里引入了tf手动模型训练（计算），这是由于tf标准模型的限制无法满足这里的loss计算
3. 这里使用了tfa的CRF层何损失计算函数，这个开发不是很完备，所以需要手动构建训练流程。
4. 似然的理解，为什么用似然的相反数做loss  https://tensorflow.google.cn/addons/api_docs/python/tfa/text/crf_log_likelihood?hl=zh-cn
5. 如果对代码不理解可以参考https://github.com/luozhouyang/keras-crf/blob/main/keras_crf/crf_model.py