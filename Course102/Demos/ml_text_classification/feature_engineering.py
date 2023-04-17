# -*- coding: utf-8 -*-
import pandas as pd
from collections import Counter
import jieba
import numpy as np

SAMPLED_LABEL = [item for item in list(range(100, 108))]
AP = 'word_segment'


# 准备数据集，从train.json中抽取出8个类作为任务数据
def prepare_data(train_data_origin_path, train_data_path, sampled_label=None):
    if sampled_label is None:
        sampled_label = SAMPLED_LABEL
    df = pd.read_json(train_data_origin_path, lines=True)
    df = df.loc[df['label'].isin(sampled_label)]
    df.to_csv(train_data_path, encoding='utf-8')
    return df


# 从train data中得到 term 的统计信息, terms支持分词、unigram和bigram
def get_terms_list(train_data_path, raw_terms_path, ap=AP):
    df = pd.read_csv(train_data_path)
    documents = df['sentence'].values
    counter = Counter()

    documents = doc_to_term_list(documents, ap=ap)

    for document in documents:
        counter.update(document)

    output_lines = []
    for (key, freq) in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        output_lines += [[key, str(freq)]]
    raw_terms = pd.DataFrame(output_lines)
    raw_terms.to_csv(raw_terms_path, header=False, index=False, encoding='utf-8')

    return raw_terms


# document => ngram list
def ngram(document, n=1):
    return [document[i:i+n] for i in range(len(document)-n+1)]


# terms中去除停用词 & 特殊符号
def remove_stop_words(raw_terms_path, terms_path, stop_words_dict_path):
    stop_words = pd.read_csv(stop_words_dict_path).values.T[0].tolist()
    punctuations = list(r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~（）… ·')
    raw_terms = pd.read_csv(raw_terms_path, encoding='utf-8', names=['word', 'freq'])
    terms = raw_terms.loc[~raw_terms['word'].isin(stop_words+punctuations)]
    terms.to_csv(terms_path, header=False, index=False, encoding='utf-8')
    return terms


# 文本 切分为 分词/ngram
# e.g.
# document =['你好，你今天怎么样。 你好'] ap='word_segment' => ['你好', '，', '你', '今天', '怎么样', '。', ' ', '你好']
def doc_to_term_list(documents, ap=AP):
    if ap == 'word_segment':
        return [jieba.lcut(document) for document in documents]
    elif ap == 'unigram':
        return [ngram(document, 1) for document in documents]
    elif ap == 'bigram':
        return [ngram(document, 2) for document in documents]


# 文本（列表） => 向量
# e.g. document = ['你好', '，', '你', '今天', '怎么样', '。', ' ', '你好']  ap='word_segment' terms=['你好', '我们']
# => one_hot_vec = [1,0]
class Encoder:
    """
    Bag of words Represent
    Vector Space Model
    例子：
    terms = ['你好', '我们']
    """
    def __init__(self, terms):
        self.terms = terms
        self.idf_dict = None

    # e.g.
    # document = ['你好', '，', '你', '今天', '怎么样', '。', ' ', '你好']
    # self.terms = ['你好', '我们']
    # => res = ['你好', '你好']
    def doc_to_self_term_list(self, document):
        res = []
        for term in self.terms:
            for i in range(document.count(term)):
                res.append(term)
        return res

    # documents => one_hot_bog based on self.terms
    def one_hot_encoder(self, documents):
        documents = [self.doc_to_self_term_list(document) for document in documents]
        return np.array([[1 if term in document else 0 for term in self.terms] for document in documents],dtype='float16')

    # documents => tf_idf_bog based on idf_dict terms
    def tf_idf_encoder(self, documents, min_freq=2):
        #idf_dict = self.idf(documents, min_freq)
        idf_dict = self.idf_dict
        term_lists = [self.doc_to_self_term_list(document) for document in documents]
        return np.array([[self.tf(term, term_list) * idf_dict[term] for term in idf_dict.keys()] for term_list in term_lists],dtype='float16')

    # The term frequency of the target term in the term_list
    # '+1' to handle divided by zero issue
    def tf(self, term_target, term_list):
        return term_list.count(term_target) / (len(term_list)+1)

    # documents => term_idf_dict
    def idf(self, documents, min_freq=2):
        res = {}
        D = len(documents)

        for term in self.terms:
            freq = 0
            for document in documents:
                if term in document:
                    freq += 1
            if freq >= min_freq:
                res[term] = np.log(D/(1+freq))
        self.idf_dict = res
        return res


def test_case():
    #prepare_data('./data/train.json', './data/train.csv')
    #get_terms_list('./data/train.csv', './data/raw_counter.csv', ap='word_segment')
    #get_terms_list('./data/train.csv', './data/raw_unigram_counter.csv', ap='unigram')
    #get_terms_list('./data/train.csv', './data/raw_bigram_counter.csv', ap='bigram')
    #terms = remove_stop_words('./data/raw_counter.csv', './data/terms.csv', './data/cn_stopwords.txt')

    '''terms = ['你好', '我们']
    encoder = Encoder(terms)
    documents = ['你好，你今天怎么样。 你好', '你好，我们今天去吃饭吧']

    documents = doc_to_term_list(documents, ap='word_segment')
    print(documents)

    res = encoder.one_hot_encoder(documents)
    print('one hot encoder: {}'.format(res))
    res = encoder.tf_idf_encoder(documents, min_freq=1)
    print('tf-idf encoder: {}'.format(res))'''


#test_case()
