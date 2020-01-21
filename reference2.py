import re
import jieba
import pandas as pd
# 引入 word2vec
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText
from gensim.models import word2vec
import gensim
import numpy as np

# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 数据路径
merger_data_path = 'data/merged_train_test_seg_data.csv'
# 模型保存路径
save_model_path='data/wv/word2vec.model'

model_wv = word2vec.Word2Vec(LineSentence(merger_data_path), sg=1,workers=8,min_count=5,size=200)

model_wv.wv.most_similar(['奇瑞'], topn=10)

model_ft = FastText(sentences=LineSentence(merger_data_path), workers=8, min_count=5, size=200)
model_ft.wv.most_similar(['奇瑞'], topn=10)

model_wv.save(save_model_path)

model = word2vec.Word2Vec.load(save_model_path)

model.wv.most_similar(['奇瑞'], topn=10)

vocab = {word:index for index, word in enumerate(model_wv.wv.index2word)}
reverse_vocab = {index: word for index, word in enumerate(model_wv.wv.index2word)}

save_embedding_matrix_path='data/embedding_matrix.txt'

def get_embedding_matrix(wv_model):
    # 获取vocab大小
    vocab_size = len(wv_model.wv.vocab)
    # 获取embedding维度
    embedding_dim = wv_model.wv.vector_size
    print('vocab_size, embedding_dim:', vocab_size, embedding_dim)
    # 初始化矩阵
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    # 按顺序填充
    for i in range(vocab_size):
        embedding_matrix[i, :] = wv_model.wv[wv_model.wv.index2word[i]]
        embedding_matrix = embedding_matrix.astype('float32')
    # 断言检查维度是否符合要求
    assert embedding_matrix.shape == (vocab_size, embedding_dim)
    # 保存矩阵
    np.savetxt('save_embedding_matrix_path', embedding_matrix, fmt='%0.8f')
    print('embedding matrix extracted')
    return embedding_matrix

embedding_matrix=get_embedding_matrix(model_wv)
print(embedding_matrix.shape)

embedding_matrix_wv=model_wv.wv.vectors
embedding_matrix_wv.shape

embedding_matrix==embedding_matrix_wv

(embedding_matrix==embedding_matrix_wv).all()

