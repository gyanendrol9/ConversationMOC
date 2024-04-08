import sys
sys.path.append('codes')

import os
import pickle
import argparse

import numpy as np
from nltk.corpus import stopwords
import gensim
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the models')
    parser.add_argument('--source_dir', help='work directory')
    parser.add_argument('--wv_model', help='word emb path')
    args = parser.parse_args()
    return args

args = parse_args()
source_data_dir = args.source_dir  
wv_model_path = args.wv_model  


wv_model = gensim.models.KeyedVectors.load_word2vec_format(wv_model_path, binary=False) 
sv_model = SentenceTransformer("sentence-transformers/nli-roberta-large")   
stops = set(stopwords.words('english'))

def remove_stopwords(sentence):
    words = sentence.split()
    wordsFiltered = []
    for w in words:
        if w not in stops:
            wordsFiltered.append(w)
    return wordsFiltered

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = t.lower()
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = '' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_sentence_emb(text):
    wdim = (wv_model['word'].shape)[0]
    text = preprocess(text)
    words = remove_stopwords(text)
    sentence_emb = np.zeros(wdim)
    for word in words:
        if word in wv_model:
            sentence_emb += wv_model[word]
    sentence_emb = sentence_emb/len(words)
    return sentence_emb

def get_task_specific_scores(model, tokenizer, text):
    encoded_input = tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    return scores  

def get_task_specific_model(task):
    # tasks = ['emoji','emotion','irony','offensive','sentiment']

    ##task='emoji' # change this for emoji & remove any existing directory with similar name to run the task suceessfully.
    if task != 'hate':
        MODEL = f"twitter-roberta-base-{task}"
    else:
        MODEL = f"twitter-roberta-base-{task}-latest"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)
    return model, tokenizer   

def TwitterRobertbaseEncoding(text, model_tokenizer):
    task='emoji' # change this for emoji & remove any existing directory with similar name to run the task suceessfully.
    emoji_score = get_task_specific_scores(model_tokenizer[task][0], model_tokenizer[task][1], text)

    task='emotion'
    emotion_score = get_task_specific_scores(model_tokenizer[task][0], model_tokenizer[task][1], text)

    task= 'hate'
    hate_score = get_task_specific_scores(model_tokenizer[task][0], model_tokenizer[task][1], text)

    task= 'irony'
    irony_score = get_task_specific_scores(model_tokenizer[task][0], model_tokenizer[task][1], text)

    task= 'offensive'
    offensive_score = get_task_specific_scores(model_tokenizer[task][0], model_tokenizer[task][1], text)

    task= 'sentiment'
    sentiment_score = get_task_specific_scores(model_tokenizer[task][0], model_tokenizer[task][1], text)

    Final_score=np.concatenate((emoji_score,emotion_score,irony_score,offensive_score,sentiment_score), axis=None)
    return Final_score

def get_post_emb(post,model_tokenizer):
    wv_sent = get_sentence_emb(post)
    
    sentences = [str(i) for i in nlp(post).sents]
    embeddings = sv_model.encode(sentences, device='cpu')
    embeddings = np.mean(embeddings, axis=0)
    
    score = []
    for sent in sentences:
        Score=TwitterRobertbaseEncoding(str(sent), model_tokenizer)
        score.append(Score)
    score = np.asarray(score)
    score = np.mean(score, axis=0)
    
    sent_emb = np.concatenate((wv_sent, embeddings,score), axis=None)
    
    return sent_emb

tasks = ['emoji','emotion','hate','irony','offensive', 'sentiment']
model_tokenizer = {}

for task in tasks:
    model_tokenizer[task] = get_task_specific_model(task)

f = open(f"{source_data_dir}/processed_datasets_comments_v2.pkl",'rb')
(user_timeline,all_posts,author_posts,topic_posts,timeline_sequence_post) = pickle.load(f)
f.close()

tot_posts = len(all_posts)
all_posts_keys = list(all_posts.keys())

fs = open(f"{source_data_dir}/skipped-post-id.txt",'w')
f = open(f"{source_data_dir}/nan-emb-post.txt",'w')

error_count = {}
patiences = 5
i = 0 
pid = 0
print('\n---------------\nStarting from:',pid)

while(i<tot_posts):
    i = pid
    try:
        post = all_posts_keys[pid]
        print(f"{pid}/{tot_posts} {all_posts[post]['sentences']}")

        embb = get_post_emb(all_posts[post]['sentences'], model_tokenizer)

        if sum(np.isnan(embb)) == 0:
            all_posts[post]['embedding'] = embb
            pid+=1
        else:
            print(f"Nan emb: {pid}/{tot_posts} ID: {post} Text:{all_posts[post]['sentences']} ")
            if pid not in error_count:
                error_count[pid] = 0
                os.system(f'mv cardiffnlp cardiffnlp-error-{pid}')
            else:
                os.system(f'rm -r cardiffnlp')
            
            error_count[pid] += 1
            if error_count[pid]>5:
                f.write(f"{post}\n")
                pid+=1
        
    except:
        print('\n---------------\nError in:',pid)
        if pid not in error_count:
            error_count[pid] = 0
            os.system(f'mv cardiffnlp cardiffnlp-error-{pid}')
        else:
            os.system(f'rm -r cardiffnlp')
        
        error_count[pid] += 1
        if error_count[pid]>patiences:
            fs.write(f"{post}\n")
            pid+=1
            
        print('\n---------------\nStarting from:',pid)

f.close()
fs.close()

f = open(f"{source_data_dir}/processed_post_embedding_all-faster.pkl",'wb')
pickle.dump(all_posts,f)
f.close()