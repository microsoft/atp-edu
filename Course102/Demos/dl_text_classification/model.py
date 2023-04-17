import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import process
import pandas as pd

lens = 23


def text_cnn_model(cls = 15):
    k = 100  # word embedding
    n = lens  # length of a sentence
    inputs = tf.keras.Input(shape=(n, k, 1), name='input_data')

    pool_outputs = []
    for filter_size in [4, 5, 6, 7]:
        conv = layers.Conv2D(kernel_size=(filter_size, k), filters=32, activation='relu')(inputs)
        conv = layers.Dropout(0.3)(conv)
        pool = layers.MaxPool2D(pool_size=(n - filter_size + 1, 1))(conv)
        pool_outputs.append(pool)

    pool_outputs = layers.concatenate(pool_outputs, axis=-1, name='concatenate')
    pool_outputs = layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)

    outputs = layers.Dense(64, activation='relu')(pool_outputs)
    outputs = layers.Dropout(0.6)(outputs)
    outputs = layers.Dense(cls)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def main():
    path = './embedding_models/tencent-ailab-embedding-zh-d100-v0.2.0-s/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'

    train_data = pd.read_csv('./data/train.csv')
    dev_data = pd.read_csv('./data/dev.csv')

    print(" == loading word embedding")
    vectors, size, dim = process.load_embeddings(path)
    vectors['OOV'] = np.random.rand(dim)
    vectors['PAD'] = np.zeros(dim)

    print("== sentence embedding ==")

    sentences = train_data['sentence'].values.tolist()
    sentences_embedded = [process.vectorize(sentence=sentence, length=lens, padding='PAD', oov='OOV', vectors=vectors)
                          for sentence in sentences]

    sentences_embedded = np.array(sentences_embedded)
    sentences_embedded = sentences_embedded.reshape(len(train_data), lens, dim, 1)

    labels = train_data['label'].values.tolist()
    label_set = set(labels)
    label_map = {}
    for i, key in enumerate(label_set):
        label_map[key] = i
    labels_mapped = np.array([label_map[label] for label in labels])
    
    # load dev data
    dev_sentences = dev_data['sentence'].values.tolist()
    dev_sentences_embedded = [process.vectorize(sentence=sentence, length=lens, padding='PAD', oov='OOV', vectors=vectors)
                          for sentence in dev_sentences]

    dev_sentences_embedded = np.array(dev_sentences_embedded)
    dev_sentences_embedded = dev_sentences_embedded.reshape(len(dev_data), lens, dim, 1)

    dev_labels = dev_data['label'].values.tolist()
    dev_labels_mapped = np.array([label_map[label] for label in dev_labels])
    

    model = text_cnn_model()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    print("== begin training ==")
    model.fit(sentences_embedded, labels_mapped,
              validation_data=(dev_sentences_embedded, dev_labels_mapped),
              epochs=30,
              batch_size=256)

    print("== save model")
    model.save_weights("text_checkpoint")

    #model = text_cnn_model()
    #model.load_weights("text_checkpoint")
    #print(model.summary())


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus)>1:
        tf.config.set_visible_devices(gpus[1], 'GPU')
        print('use gpu1')
    main()


