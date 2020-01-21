import re
import jieba
import pandas as pd
import numpy as np
# 引入 word2vec
from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import gensim
# 引入日志配置
import logging


# create embedding matrix using vocab index
def create_embedding_matrix(model_path, vocab_path):
    # open vocab file
    vocab_file = open(vocab_path, 'r')
    # generate vocab dict using index as key
    vocab_dict = dict()
    for line in vocab_file.readlines():
        line = line.strip('\n').split(' ')
        key = int(line[1])
        vocab_dict[key] = line[0]
    vocab_file.close()

    # load word2vec model
    model = word2vec.Word2Vec.load(model_path)
    # word_vectors = KeyedVectors.load_word2vec_format('data/wv/vectors.txt', binary=False)  # C text format
    # word_vectors_bin = KeyedVectors.load_word2vec_format('data/wv/vectors.bin.gz', binary=True)  # C binary format

    # generating embedding matrix consider vocab index as key
    embedding_matrix = np.zeros((len(vocab_dict), model.vector_size))
    for i in vocab_dict.keys():
        word = vocab_dict[i]
        if word == '<unk>':
            continue
        embedding_matrix[i] = model.wv[word]

    return embedding_matrix

# training model by gensim word2vec api
def train_model(input_file, output_path, embedding_vectors_path, embedding_bin_path, min_count=20, size=200, iteration=10, worker=8, batch_words=20000):

    # reading merged train and test dialogue question and reports sentences
    merger_df = pd.read_csv(input_file, header=None)
    print('merger_data_path data size {}'.format(len(merger_df)))
    merger_df.head()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # min_count 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5. 
    # size is the output coloum dimension, default is 100
    # worker is the thread number
    # iter is the itration of training
    # batchwords is the batch size of one training step
    model = word2vec.Word2Vec(LineSentence(input_file), workers=worker, min_count=min_count, size=size, iter=iteration, batch_words=batch_words)

    # save model through 3 different way
    model.save(output_path)
    model.wv.save_word2vec_format(embedding_vectors_path, binary=False)
    model.wv.save_word2vec_format(embedding_bin_path, binary=True)  


if __name__ == '__main__':
    
    # merged train and test segmentation datasets
    merger_data_path = 'data/merged_train_test_seg_data.csv'

    # word2vec path, embedding vectros and embedding bin path
    save_model_path='data/wv/word2vec.model'
    embedding_vectors_path = 'data/wv/vectors.txt'
    embedding_bin_path = 'data/wv/vectors.bin.gz'

    # vocab file path
    vocab_path = "data/vocab.txt"

    train_model(merger_data_path, save_model_path, embedding_vectors_path, embedding_bin_path, min_count=10, size=300, iteration=10, worker=8, batch_words=20000)
    embedding_matrix = create_embedding_matrix(save_model_path, vocab_path)
    print(embedding_matrix.shape)
    print(embedding_matrix[0])

