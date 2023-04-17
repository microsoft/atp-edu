1. 项目主文件为chatbot.py,启动使用 `python chatbot.py`
2. 项目使用sentenceBERT作为语义相似度匹配模型,需要安装依赖 `pip install sentence-transformers`
3. 聊天预料见`语料库.xlsx` ,其中包含标准问,相似问和回复, 例子中只是用了标准问和回复, 相似问可以留给学员做拓展
4. 默认使用 ["uer/sbert-base-chinese-nli"模型](https://huggingface.co/uer/sbert-base-chinese-nli), 该模型默认不会做向量归一化,所以需要在`get_embedding`中做向量归一化(只有单位向量的点成结果才等于cosine距离)
5. 可以将"使用Faiss"加速相似度匹配作为课程可选内容