import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import IncrementalPCA
from feature_engineering import Encoder


TRAIN_DATA_PATH = './data/train.csv'
TEST_DATA_PATH = './data/dev.csv'
TERMS_PATH = './data/terms.csv'
MODEL_PATH = './model/'


def get_data(path):
    #data_pd = pd.read_csv(path)[0:1000]
    data_pd = pd.read_csv(path)
    documents = data_pd['sentence'].values
    labels = data_pd['label'].values
    print(len(documents))
    return documents, labels


def model_train(model, X_train, y_train, X_test, y_test, model_name):
    print("train model: {}".format(model_name))
    model.fit(X_train, y_train)
    print("precision on \n== training set: {0} \n== test set: {1}".format(model.score(X_train, y_train),
                                                                          model.score(X_test, y_test)))
    model_save(model, MODEL_PATH+'{}.pickle'.format(model_name))
    return model


def model_save(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def model_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    terms = pd.read_csv(TERMS_PATH, names=['term', 'freq'], encoding='utf-8')['term'].values[0:8192*2]
    print("== terms size {} ==".format(len(terms)))
    train_documents, train_labels = get_data(TRAIN_DATA_PATH)
    test_documents, test_labels = get_data(TEST_DATA_PATH)

    models = []
    models += [LinearSVC(penalty='l2', loss='squared_hinge',  dual=True, tol=0.0001, C=1.0, multi_class='ovr', max_iter=800),
               LogisticRegression(multi_class="ovr", max_iter=200),
               DecisionTreeClassifier(max_depth=25),
               GaussianNB()
               ]

    encoder = Encoder(terms)
    encoder.idf(train_documents, min_freq=2)

    train_documents_encoded = encoder.tf_idf_encoder(train_documents)
    test_documents_encoded = encoder.tf_idf_encoder(test_documents)

    import gc
    del encoder
    gc.collect()
    print("== delete encoder ==")
    # 降维
    pca = IncrementalPCA(n_components=4096)
    pca.fit(train_documents_encoded)

    '''X = pca.transform(documents_encoded)
    y = labels'''

    X_train = pca.transform(train_documents_encoded)
    X_test = pca.transform(test_documents_encoded)

    del pca
    gc.collect()
    print("== delete pca ==")

    y_train = train_labels
    y_test = test_labels
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15)

    for i in range(len(models)):
        model_train(models[i], X_train, y_train, X_test, y_test, 'model_{}'.format(i))

    '''test = model_load(MODEL_PATH+'model_0.pickle')
    print("precision on \n== training set: {0} \n== test set: {1}".format(test.score(X_train, y_train),
                                                                          test.score(X_test, y_test)))'''


if __name__ == '__main__':
    main()



