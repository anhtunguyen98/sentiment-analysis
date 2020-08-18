from static import *
import torch
import numpy as np
from gensim.models import KeyedVectors
from main_model import *
import gensim
from pyvi import ViTokenizer
from keras.preprocessing.sequence import pad_sequences
class LSTM_Model:
    def __init__(self):
        self.w2v = KeyedVectors.load_word2vec_format('word_vector')
        self.vocab = self.w2v.vocab
        self.model=Sentiment_Analysic(vocab_size=self.w2v.wv.vectors.shape[0],embedding_dim=self.w2v.wv.vectors.shape[1],num_labels=2)
        self.model=torch.load(LSTM_MODEL_DIR)
        self.model.cpu()
    def result(self,text):
        text = gensim.utils.simple_preprocess(text)
        text = ' '.join(text)
        text = ViTokenizer.tokenize(text)
        X_test = []
        for word in text.split(' '):
            if word not in self.vocab:
                X_test.append(self.vocab.get('UNK').index)
            else:
                X_test.append(self.vocab.get(word).index)
        test_ids = pad_sequences([[word for word in X_test]],
                                 value=self.vocab.get('PAD').index, maxlen=MAX_LENGTH, truncating='post', padding='post',
                                 dtype='long')
        input_id = torch.tensor(test_ids, dtype=torch.long)
        with torch.no_grad():
            output = self.model(input_id)
        output = output.detach().numpy().squeeze()
        return output[1],output[0]
class CNN_Model:
    def __init__(self):
        self.w2v = KeyedVectors.load_word2vec_format('word_vector')
        self.vocab = self.w2v.vocab
        self.model=Cnn_Sentiment_Analysis(vocab_size=self.w2v.wv.vectors.shape[0],embedding_dim=self.w2v.wv.vectors.shape[1])
        self.model=torch.load(CNN_MODEL_DIR)
        self.model.cpu()
    def result(self,text):
        text = gensim.utils.simple_preprocess(text)
        text = ' '.join(text)
        text = ViTokenizer.tokenize(text)
        X_test = []
        for word in text.split(' '):
            if word not in self.vocab:
                X_test.append(self.vocab.get('UNK').index)
            else:
                X_test.append(self.vocab.get(word).index)
        test_ids = pad_sequences([[word for word in X_test]],
                                 value=self.vocab.get('PAD').index, maxlen=MAX_LENGTH, truncating='post', padding='post',
                                 dtype='long')
        input_id = torch.tensor(test_ids, dtype=torch.long)
        with torch.no_grad():
            output = self.model(input_id)
        output = output.detach().numpy().squeeze()
        return output[1],output[0]


