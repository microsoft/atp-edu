from sentence_transformers import SentenceTransformer
from sentence_transformers import  util
import pandas as pd
import numpy as np

model = SentenceTransformer("uer/sbert-base-chinese-nli")


def get_embedding(text):
    embedding = model.encode([text])[0]
    return  embedding/np.sqrt(np.sum(embedding**2))  # convert to unit vector


def initialize():
    df = pd.read_excel('语料库.xlsx')
    q_a_pairs = df[["标准问","回复"]].to_dict('records')
    question_list = [pair['标准问'] for pair in q_a_pairs]
    embedding_list = [get_embedding(question) for question in  question_list]
    embeddings = np.array(embedding_list)
    return q_a_pairs,embeddings
    
if __name__ == "__main__":
    q_a_pair_list, questions_embedding = initialize()
    
    print("我是虎墩, 你有问题要问我么?(输入回车提交)")
    while True:
        question = input()
        if not question:
            break
        question_embedding = get_embedding(question)
        similarities = np.dot(questions_embedding,question_embedding)
        best_question_index = np.argmax(similarities)
        best_question_score = similarities[best_question_index]
        if best_question_score<0.5:
            print('这道题太难了')
        else:
            matched_question = q_a_pair_list[best_question_index]
            print(f"Log: matched question {matched_question['标准问']} with score {best_question_score}")
            print(matched_question['回复'])
            
            
    print("再见")