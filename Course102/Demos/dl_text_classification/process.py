import tensorflow as tf
import numpy as np
import pandas as pd
import jieba

path = './embedding_models/tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'


def load_embeddings(path):
    f = open(path, encoding='utf-8', errors='ignore', mode='r')
    first_line = f.readline()
    first_line = first_line.rstrip().split()
    size = int(first_line[0])
    dim = int(first_line[1])
    # word:embeddings
    vocab = {}
    for line in f.readlines():
        parse_line = line.rstrip().split()
        word = parse_line[0]
        embedding = [float(i) for i in parse_line[1:]]
        vocab[word] = embedding
    f.close()
    print("word embedding vocab loaded")
    return vocab, size, dim


def vectorize(sentence, length, padding, oov, vectors):
    sentence = jieba.lcut(sentence)
    sentence = handle_padding(sentence, length, padding)
    sentence = handle_oov(sentence, vectors.keys(), oov)
    #print(sentence)
    return np.array([np.array(vectors[word]) for word in sentence])


def handle_padding(sentence_segment, length, padding):
    if len(sentence_segment) >= length:
        return sentence_segment[0:length]
    else:
        return sentence_segment + (length - len(sentence_segment)) * [padding]


def handle_oov(sentence_segment, word_dict, oov):
    return [word if word in word_dict else oov for word in sentence_segment]


def test():
    vocab, size, dim = load_embeddings(path)

    OOV = np.zeros(dim).tolist()
    PAD = np.zeros(dim).tolist()
    length = 20

    vocab['oov'] = OOV
    vocab['pad'] = PAD
    test_sentence = '你好，你今天吃什么呢？'

    res = vectorize(test_sentence, length, 'pad', 'oov', vocab)

    print(res)

#test()