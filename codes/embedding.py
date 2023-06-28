from cmath import nan
from sentence_transformers import SentenceTransformer
import gensim
from nltk.corpus import stopwords
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import operator

import nltk
nltk.download('stopwords')

class DataEmbedding:
    def __init__(self, data:pd.DataFrame, sv_model_path, wv_model_path, max_len = 122, batch_size = 8, epochs = 2):
        # Hyperparameters
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.data = data
        self.sv_model = SentenceTransformer(sv_model_path)
        self.wv_model = gensim.models.KeyedVectors.load_word2vec_format(wv_model_path, binary=False)
        self.stops = set(stopwords.words('english'))

        embeddings = self.sv_model.encode(['This is a test'], device='cpu')
        self.wdim = (self.wv_model['word'].shape)[0]
        self.wdim = self.wdim+embeddings.shape[1]
        self.sentence_emb = np.zeros(self.wdim)+9   #Why +9 added??
        
        sentences = data[['id','body']]
        self.sentences_id = sentences.set_index('id')['body'].to_dict()
        sessions = set(list(data['author']))

        postids = list(set(data["id"].values))
        postids.append("ENDPAD")
        self.n_postids = len(postids); 

        self.post2idx = {w: i for i, w in enumerate(postids)}
        self.idx2post = {i: w for i, w in enumerate(postids)}

        getter = self.PostGetter(data)
        self.sentences = getter.sentences

        moc_tags = list(set(data["momentofchange"].values))
        moc_tags.append("ENDPAD")
        if(moc_tags.count(nan)>0):
            moc_tags.remove(nan)
        print("Moment of change labels: ", moc_tags)
        self.n_tags = len(moc_tags)
        
        self.tag2idx = {t: i for i, t in enumerate(moc_tags)}
        self.idx2tag = {i: t for i, t in enumerate(moc_tags)}
            
    def remove_stopwords(self, sentence):
        words = sentence.split()
        wordsFiltered = []
        for w in words:
            if w not in self.stops:
                wordsFiltered.append(w)
        return wordsFiltered
        
    # Preprocess text (username and link placeholders)
    def preprocess(self, text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    # Get embedding for a post
    def get_sentence_emb(self, text):
        wdim = (self.wv_model['word'].shape)[0]
        text = self.preprocess(text)
        words = self.remove_stopwords(text)
        sentence_emb = np.zeros(wdim)
        for word in words:
            if word in self.wv_model:
                sentence_emb += self.wv_model[word]

        if (sentence_emb == np.zeros(wdim)).all():
            return np.zeros(wdim)
        else:
            sentence_emb = sentence_emb/len(words)
        
        return sentence_emb

    def get_data_representation(self):
        getter = self.PostGetter(self.data)
        sentences = getter.sentences
        
        # Embedding posts
        X = [[self.post2idx[w[2]] for w in s] for s in sentences]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.n_postids - 1)
        XX = []
        
        for sess in X:
            ses_pos = []
            ration_pos = []
            for pid in sess:
                pidd = self.idx2post[pid]

                if pidd != 'ENDPAD':
                    # Embedding sentence vectors
                    sent = self.get_sentence_emb(self.sentences_id[pidd])
                    embeddings = self.sv_model.encode([self.sentences_id[pidd]], device='cpu')
                    sent = np.concatenate((sent, embeddings[0]), axis=None)
                    ses_pos.append(sent)

                else:
                    ses_pos.append(self.sentence_emb)
                    
            ses_pos = np.asarray(ses_pos)
            XX.append(ses_pos)


        XX = np.asarray(XX)
        
        # Embedding labels
        y = [[self.tag2idx[w[6]] for w in s] for s in sentences]
        y = pad_sequences(maxlen = self.max_len, sequences=y, padding="post", value=self.tag2idx["ENDPAD"])
        y = [to_categorical(i, num_classes=self.n_tags) for i in y]
        
        # Embedding rationale labels
        r = [[w[7] for w in s] for s in sentences]
        r = pad_sequences(maxlen = self.max_len, sequences=r, padding="post", value=0)
        # r = [to_categorical(i, num_classes=2) for i in r]
        
        return XX, y, r
    
    def get_wdim(self):
        return self.wdim
    
    def get_n_tags(self):
        return self.n_tags

    def get_idx2tag(self):
        return self.idx2tag

    def get_tag2idx(self):
        return self.tag2idx 

    def get_sentences(self):
        return self.sentences
    
    class PostGetter(object):
        def __init__(self, data):
            self.data = data
            self.empty = False
            # ['author', 'sentenceindex', 'sentences', 'id', 'parent_id', 'post_type', 'mood', 'momentofchange']
            agg_func = lambda s: [(a, s, i, pi, pt, m, moc, r) for a, s, i, pi, pt, m, moc, r in zip(s["author"].values.tolist(),
                                                            s["body"].values.tolist(),
                                                            s["id"].values.tolist(),
                                                            s["parent_id"].values.tolist(),
                                                            s["post_type"].values.tolist(),
                                                            s["mood"].values.tolist(),
                                                            s["momentofchange"].values.tolist(),
                                                            s["rationale"].values.tolist())]
            self.grouped = self.data.groupby("author").apply(agg_func)
            self.sentences = [s for s in self.grouped]
        
        def get_next(self,author):
            try:
                s = self.grouped["Session: {}".format(author)]
                return s
            except:
                return None
