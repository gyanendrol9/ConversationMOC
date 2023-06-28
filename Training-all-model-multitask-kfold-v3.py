import networkx as nx
import tensorflow as tf
tf.config.run_functions_eagerly(True)

import sys
sys.path.append('codes')
from data_reader import *
from plots import *
from models_v4 import *

# Prepare the function to load the data
from utils.job import get_job_config
import os
# Load Dataset
import pickle
# from data_reader import DataReader
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np

def read_yamlconfig(path):
    job = get_job_config(path)
    params = job['PARAMS']

    base_dir = params['base_dir']
    print("working base_dir: ", base_dir )    
    
    source_data_dir = params['source_data_dir']
    print("source_data_dir: ", source_data_dir)

    return params

# Prepare the configuration
yaml_path = os.path.join("configs", "job_reddit.yaml")


# args = parse_job_args()
config = read_yamlconfig(yaml_path)

# Train test split
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

np.random.seed(config['shuffle_seed'])
tf.random.set_seed(config['shuffle_seed'])
random.seed(config['shuffle_seed'])

processed_dir = f"{config['base_dir']}/data_processed"

f = open(f"{processed_dir}/processed_training_padded_considered_final_all_emb.pkl",'rb')
(timeline_pids, timeline_node_feat_seq,timeline_node_moc_seq,timeline_network,timeline_users_flag,timeline_users, timeline_topics) = pickle.load(f)
f.close()

#Fix label positions task-1
moc_labels = {}
moc_labels['O'] = 0
moc_labels['IE'] = 1
moc_labels['IS'] = 2
moc_labels[' '] = 3 #non-target user
moc_labels['padded'] = 4

idx2moc = {moc_labels[t]: t for t in moc_labels}

n_tags = len(moc_labels)

topic_timeline  = {}
for t, topic in enumerate(timeline_topics):
    if topic not in topic_timeline:
        topic_timeline[topic] = []
    topic_timeline[topic].append(t)
    
topic_labels = {i:t for t, i in enumerate(topic_timeline)}
idx2topic = {topic_labels[t]: t for t in topic_labels}
n_tags2 = len(topic_labels)

max_len = max([feat.shape[0] for feat in timeline_node_feat_seq])
timeline_padded_users_flag = pad_user_flag(timeline_users_flag, max_len)

moc_count = {}
tot_moc_count = {}

timeline_moc_seq_id = []
timeline_topic_id = []

for t,tmoc in enumerate(timeline_node_moc_seq):
    timeline_moc = []
    moc_count[t] = {} 
    for moc in tmoc:
        if moc not in moc_count[t]:
            moc_count[t][moc]=0
        moc_count[t][moc]+=1
        
        if moc not in tot_moc_count:
            tot_moc_count[moc]=0
        tot_moc_count[moc]+=1
        
        timeline_moc.append(moc_labels[moc])
    timeline_moc_seq_id.append(np.asarray(timeline_moc))
    timeline_topic_id.append(topic_labels[timeline_topics[t]])

y = [to_categorical(i, num_classes=n_tags) for i in timeline_moc_seq_id]
y2 = [to_categorical(i, num_classes=n_tags2) for i in timeline_topic_id]

y_padded = []
for i in y:
    padded_y = padding_node_feat(i, max_len,0)
    for j in range(len(i),max_len):
        padded_y[j][4] = 1 #padded vector label
    y_padded.append(padded_y)

x_seq_all = np.asarray(timeline_node_feat_seq).astype(np.float32)
x_net_all = np.asarray(timeline_network).astype(np.float32)
y_train_all = np.asarray(y_padded)
y2_train_all = np.asarray(y2)
users_flag_all = np.asarray(timeline_padded_users_flag)
tu_x_train_all, tu_y_train_all, max_len, tu_x_train_pids_all = get_tuser_only_data(x_seq_all, y_train_all, users_flag_all, timeline_pids)

from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Concatenate, Bidirectional, Flatten

mask = np.sum(y_train_all[..., 1:3], axis=-1)
train_test_ids = []
for idx,i in enumerate(mask):
    if sum(i)>0:
        train_test_ids.append(idx)
    else:
        print(idx,mask[idx],'\n')


train_split_path = f"{processed_dir}/train-test-k-fold-topicwise-split-filter-NoMOC.pkl"
if not os.path.exists(train_split_path):    
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    topic_k_fold = {}
    for topic in topic_timeline:
        random.shuffle(topic_timeline[topic])
        if topic not in topic_k_fold:
            topic_k_fold[topic] = []
        for train_index, test_index in kf.split(topic_timeline[topic]):
            train_index = [topic_timeline[topic][i] for i in train_index]
            test_index = [topic_timeline[topic][i] for i in test_index]
            topic_k_fold[topic].append((train_index, test_index))
            
    f = open(train_split_path,'wb')
    pickle.dump(topic_k_fold,f)
    f.close()
else:
    print(f'Loading: {train_split_path}')
    f = open(train_split_path,'rb')
    topic_k_fold = pickle.load(f)
    f.close()

k_fold = {}
for topic in topic_k_fold:
    for i in range(10):
        train_ids, test_ids = topic_k_fold[topic][i]
        if i not in k_fold:
            k_fold[i] = {}
            k_fold[i]['Train'] = []
            k_fold[i]['Test'] = []
        
        k_fold[i]['Train'] += list(train_ids)
        k_fold[i]['Test'] += list(test_ids)

for i in range(10):
    print(len(k_fold[i]['Train']),len(k_fold[i]['Test']))

model_path = f"{config['base_dir']}/model_GCN_LSTM_multitask"
if not os.path.exists(model_path):
    os.mkdir(model_path)

f = open(f"{config['base_dir']}/data_processed/processed_word_embedding_all-faster.pkl",'rb')
all_posts = pickle.load(f)
f.close()

emotion_label = {}
emotion_label[0] = 'anger'
emotion_label[1] = 'joy'
emotion_label[2] = 'optimism'
emotion_label[3] = 'sadness'

sentiment_label = {}
sentiment_label[0] = 'negative'
sentiment_label[1] = 'neutral'
sentiment_label[2] = 'positive'


# #### GCN
max_len = x_seq_all.shape[1]

# Network 2 (Target/Non-target)
users_flag_net = []
for users_flag in timeline_users_flag:
    A = get_adjacency_matrix_userflag(users_flag)
    padded_A = padding_adj_matrix(A, max_len)
    users_flag_net.append(padded_A)

emotion_net = []
sentiment_net = []
for pids in timeline_pids:
    EA, SA = get_adjacency_matrix_emotion_sentiment(pids, all_posts)
    padded_EA = padding_adj_matrix(EA, max_len)
    padded_SA = padding_adj_matrix(SA, max_len)
    emotion_net.append(padded_EA)
    sentiment_net.append(padded_SA)

epochs = config['epochs']
batch_size = config['batch_size']

for kfold in reversed(range(1)):
    print(f'------------------------------\nTraining Fold: {kfold}\n------------------------------')
    train_index = k_fold[kfold]['Train']
    test_index =  k_fold[kfold]['Test']
    print(len(train_index), len(test_index))
    x_train, x_net_train, y_train, uf_train, x_train_pids = get_dataset(train_index, timeline_pids, x_seq_all, x_net_all, y_train_all, users_flag_all)
    x_test, x_net_test, y_test,uf_test, x_test_pids = get_dataset(test_index, timeline_pids, x_seq_all, x_net_all, y_train_all, users_flag_all)

    # Target user only data
    tu_x_train, tu_x_test = get_split_dataset(tu_x_train_all, train_index, test_index)
    tu_y_train, tu_y_test = get_split_dataset(tu_y_train_all, train_index, test_index)
    y2_train, y2_test = get_split_dataset(y2_train_all, train_index, test_index)

    print(f'------------------------------\nCalculate data statistics:\n------------------------------')
    calculate_statistics_task2(y2_train,topic_labels)
    print('\n')
    calculate_statistics_task2(y2_test,topic_labels)

    calculate_statistics_task1(y_train)
    calculate_statistics_task1(y_test)


    # ### All conversations
    # Define input shapes
    input_posts_timeline, input_features_dim = x_train[0].shape
    _, num_classes = y_train[0].shape
    num_classes2 = y2_train[0].shape[0]

    output_dim_node_features = 100 # Dimension of output node features

    print('Input feature shape:',x_train[0].shape)


    user_flag_net_train, user_flag_net_test = get_split_dataset(users_flag_net, train_index, test_index)
    sentiment_net_train, sentiment_net_test = get_split_dataset(sentiment_net, train_index, test_index)
    emotion_net_train, emotion_net_test = get_split_dataset(emotion_net, train_index, test_index)

    title = 'target_user_LSTM_epochs'
    train_lstm_multitask(epochs, batch_size, tu_x_train, tu_y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_epochs'
    train_lstm_multitask(epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)
    
    # get_lstm_gcn_multiplex_model 4 nets
    title = 'all_user_LSTM_GCN_4multiplex_epochs'
    multiplex_net_train = [emotion_net_train, sentiment_net_train, user_flag_net_train, x_net_train]
    #multiplex_net_test = [emotion_net_test, sentiment_net_test, user_flag_net_test, x_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    # get_lstm_gcn_multiplex_model 2
    # [('e', 's'), ('e', 'u'), ('e', 'p'), ('s', 'u'), ('s', 'p'), ('u', 'p')]
    title = 'all_user_LSTM_GCN_ES_epochs'
    multiplex_net_train = [emotion_net_train, sentiment_net_train]
    #multiplex_net_test = [emotion_net_test, sentiment_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_GCN_EU_epochs'
    multiplex_net_train = [emotion_net_train,user_flag_net_train]
    #multiplex_net_test = [emotion_net_test, user_flag_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_GCN_EP_epochs'
    multiplex_net_train = [emotion_net_train, x_net_train]
    #multiplex_net_test = [emotion_net_test, x_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_GCN_SU_epochs'
    multiplex_net_train = [sentiment_net_train, user_flag_net_train]
    #multiplex_net_test = [sentiment_net_test, user_flag_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_GCN_SP_epochs'
    multiplex_net_train = [sentiment_net_train, x_net_train]
    #multiplex_net_test = [sentiment_net_test, x_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_GCN_UP_epochs'
    multiplex_net_train = [user_flag_net_train, x_net_train]
    #multiplex_net_test = [user_flag_net_test, x_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    # get_lstm_gcn_multiplex_model 3
    #  [('e', 's', 'u'), ('e', 's', 'p'), ('e', 'u', 'p'), ('s', 'u', 'p')]
    title = 'all_user_LSTM_GCN_ESUF_epochs'
    multiplex_net_train = [emotion_net_train, sentiment_net_train, user_flag_net_train]
    #multiplex_net_test = [emotion_net_test, sentiment_net_test, user_flag_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_GCN_ESP_epochs'
    multiplex_net_train = [emotion_net_train, sentiment_net_train, x_net_train]
    #multiplex_net_test = [emotion_net_test, sentiment_net_test, x_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_GCN_EUFP_epochs' #fold 6
    multiplex_net_train = [emotion_net_train, user_flag_net_train, x_net_train]
    #multiplex_net_test = [emotion_net_test, user_flag_net_test, x_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_GCN_SUFP_epochs'
    multiplex_net_train = [sentiment_net_train, user_flag_net_train, x_net_train]
    #multiplex_net_test = [sentiment_net_test, user_flag_net_test, x_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)


    # ### GCN Layer single-net ['e','s','u','p']
    title = 'all_user_LSTM_GCN_UF_epochs'
    multiplex_net_train = [user_flag_net_train]
    #multiplex_net_test = [user_flag_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_GCN_sentiment_epochs'
    multiplex_net_train = [sentiment_net_train]
    #multiplex_net_test = [sentiment_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_GCN_emotion_epochs'
    multiplex_net_train = [emotion_net_train]
    #multiplex_net_test = [emotion_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)

    title = 'all_user_LSTM_GCN_post2post_epochs'
    multiplex_net_train = [x_net_train]
    #multiplex_net_test = [x_net_test]
    train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path)



    