import networkx as nx
import tensorflow as tf

tf.config.run_functions_eagerly(True)

import sys
sys.path.append('codes')


# Prepare the function to load the data
from utils.job import get_job_config
import os
# Load Dataset
import pickle

# from data_reader import DataReader
from data_reader import *
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


processed_dir = f"{config['base_dir']}/data_processed"

f = open(f"{processed_dir}/processed_networks_final.pkl",'rb')
(users_connections,post2post) = pickle.load(f)
f.close()

f = open(f"{processed_dir}/processed_datasets_comments.pkl",'rb')
(user_timeline,all_posts,author_posts,topic_posts,timeline_sequence_post) = pickle.load(f)
f.close()

sequence_pid = {}

for tuser in timeline_sequence_post:
    for topic in timeline_sequence_post[tuser]:
        for t in range(len(timeline_sequence_post[tuser][topic])):
            tline = f'{topic}_{tuser}_{t}'
            sequence_pid[tline] = timeline_sequence_post[tuser][topic][t]['all_users_pid']

post_len = {}
for tline in sequence_pid:
    post_len[tline]=len(sequence_pid[tline])
    
sorted_post_len = dict(sorted(post_len.items(), key=lambda item: item[1], reverse=True))
timeline_keys = list(sorted_post_len.keys())

### Filter timeline based on target user and counts
filepath = f"{processed_dir}/skipped_timelines-1-3conversation.tsv"
f = open(filepath,'w')
f.write(f'Timeline\t#Conversations\t#TargetUser\n')

filepath_c = f"{processed_dir}/considered_timelines.tsv"
fc = open(filepath_c,'w')
fc.write(f'Timeline\t#Conversations\t#TargetUser\n')

considered_timelines  = []
skipped_timelines = []
for timeline in timeline_keys:
    pids  = sequence_pid[timeline]
    tcount = 0
    
    for pid in pids:
        author =  '_'.join(all_posts[pid]['author'].split('_')[1:])
        if all_posts[pid]['Target_user']:
            tcount+=1
    
    if tcount > 1 and len(pids)>2:  #Filter conversation > 2 and target user posts > 1
        considered_timelines.append(timeline)
        fc.write(f'{timeline}\t{sorted_post_len[timeline]}\t{tcount}\n')
    else:
        skipped_timelines.append(timeline)
        print(f'{timeline}\t{sorted_post_len[timeline]}\t{tcount}')
        f.write(f'{timeline}\t{sorted_post_len[timeline]}\t{tcount}\n')
        
f.close()
fc.close()

#Fix label positions
moc_labels = {}
moc_labels['O'] = 0
moc_labels['IE'] = 1
moc_labels['IS'] = 2
moc_labels[' '] = 3 #non-target user
moc_labels['padded'] = 4

idx2moc = {moc_labels[t]: t for t in moc_labels}

n_tags = len(moc_labels)

moc_count = {}
tot_moc_count = {}


#Sanity check if there is any missing labels for target users
filepath = f"{processed_dir}/missing_labels_timeline.tsv"
filepath_c = f"{processed_dir}/skipped_timelines-missing_labels.tsv"

f = open(filepath,'w')
fc = open(filepath_c,'w')

target_user_pids = []
target_user_labels = []
considered_timelines_v2 = []
for tline in considered_timelines:
    pids  = sequence_pid[tline]
    node_labels = []
    node_pids =[]
    tauthor = tline.split('_')
    tauthor =  '_'.join(tauthor[1:-1])
    errr = 0
    for pid in pids:
        if all_posts[pid]['Target_user']:
            author = all_posts[pid]['author'].split('_')
            topic = author[0]
            author =  '_'.join(author[1:])
            if all_posts[pid]['momentofchange'] == ' ':
                print(f"{tline}\t{author}\t{pid}\t'{all_posts[pid]['momentofchange']}'")
                fc.write(f"{topic}\t{author}\t{pid}\t{all_posts[pid]['momentofchange']}\n")
                errr +=1
            else:
                node_labels.append(all_posts[pid]['momentofchange'])
                node_pids.append(pid)

    if len(node_labels)>1 and errr == 0:
        target_user_pids.append(node_pids)
        target_user_labels.append(node_labels)
        considered_timelines_v2.append(tline)
    else:
        f.write(f"{tline}\n")

f.close()
fc.close()

f = open(f"{processed_dir}/processed_training_set_considered-tuser.pkl",'wb')
pickle.dump((target_user_pids,target_user_labels,considered_timelines_v2),f)
f.close()

f = open(f"{config['base_dir']}/data_processed/processed_word_embedding_all-faster.pkl",'rb')
annotator_all_posts = pickle.load(f)
f.close()

tline_pids = []
tline_feature_embedding = []
tline_moc_labels = []
tline_target_users = []
tline_network = []

for tline in considered_timelines_v2:
    pids  = sequence_pid[tline]
    node_labels = [all_posts[pid]['momentofchange'] for pid in pids]
    tuser = [all_posts[pid]['Target_user'] for pid in pids]
    if tline == 'Anxiety__littlemoose_0':
        tline2 == 'Anxiety_7922_0'
    else:
        tline2 = tline
    A, A_idx,users = get_adjacency_matrix(tline2, sequence_pid, post2post, annotator_all_posts)
    
    #Add self-loop
    for i in range(len(A)):
        A[i][i] = 1

    print('Check: ',A.shape)
    
    tline_feature_embedding.append(get_node_features(annotator_all_posts, pids)) 
    tline_moc_labels.append(node_labels)
    tline_target_users.append(tuser)
    tline_pids.append(pids)
    tline_network.append((A, A_idx,users))

f = open(f"{processed_dir}/processed_training_set_considered-all.pkl",'wb')
pickle.dump((tline_pids, tline_feature_embedding, tline_moc_labels, tline_target_users, tline_network, considered_timelines_v2),f)
f.close()

# Create training dataset with padding
timeline_pids = []
timeline_topics = []
timeline_node_feat_seq = []
timeline_node_moc_seq = []
timeline_network = []
timeline_users_flag = []
timeline_users = []

pad_tuser_len = 5

tline_post_counts = [feat.shape[0] for feat in tline_feature_embedding]
max_len = max(tline_post_counts)
pad_tuser_len = 5 #Randomly choosen

for t, timeline in enumerate(considered_timelines_v2):
    topic = timeline.split('_')[0]
    node_feat = tline_feature_embedding[t]
    pids  = tline_pids[t]
    moc_labels = tline_moc_labels[t]
    target_user_flag = tline_target_users[t]
    A, A_idx,users = tline_network[t]
   
    padded_A = padding_adj_matrix(A, max_len)

    # Pad the post embedding with <post-emb>[00000] for non-target user posts
    node_feat_tuser = []
    for i in range(len(target_user_flag)):
        node_feat_tuser.append(padding_vector(node_feat[i],pad_tuser_len,target_user_flag[i]))
    node_feat_tuser = np.asarray(node_feat_tuser)
    
    # Pad the post embedding with max_len
    padded_node_feat  = padding_node_feat(node_feat_tuser, max_len, 9) #Outlier the padded vectors

    timeline_node_moc_seq.append(moc_labels)
    timeline_network.append(padded_A)
    timeline_node_feat_seq.append(padded_node_feat)
    timeline_users_flag.append(target_user_flag)
    timeline_users.append(users)
    timeline_pids.append(pids)
    timeline_topics.append(topic.lower())

f = open(f"{processed_dir}/processed_training_padded_considered_final_all_emb.pkl",'wb')
pickle.dump((timeline_pids, timeline_node_feat_seq,timeline_node_moc_seq,timeline_network,timeline_users_flag,timeline_users, timeline_topics),f)
f.close()

user_stat = {}
user_stat['moc'] = {}
user_stat['topics'] = {}
for t in range(len(timeline_node_moc_seq)):
    tuser = timeline_users[t][0]
    for moc in timeline_node_moc_seq[t]:
        if moc not in user_stat['moc']:
            user_stat['moc'][moc]=[]
        user_stat['moc'][moc].append(tuser)
    
    topic = timeline_topics[t]
    if topic not in user_stat['topics']:
        user_stat['topics'][topic]=[]
    user_stat['topics'][topic].append(tuser)
        

topic_posts_stat = {}

for t in range(len(timeline_node_moc_seq)):   
    topic = timeline_topics[t]
    tuser = timeline_users[t][0]
    if topic not in topic_posts_stat:
        topic_posts_stat[topic] = {}
        topic_posts_stat[topic]['Users'] = 0
    
    topic_posts_stat[topic]['Users'] += len(set(timeline_users[t]))
        
    for moc in timeline_node_moc_seq[t]:
        if moc not in topic_posts_stat[topic]:
            topic_posts_stat[topic][moc] = 0
            
        topic_posts_stat[topic][moc]+=1


print('Topics\t#TargetUsers\tTotalUsers\tO\tIE\tIS\tNon')
for topic in user_stat['topics']:
    print(f"{topic}\t{len(set(user_stat['topics'][topic]))}\t{topic_posts_stat[topic]['Users']}\t{topic_posts_stat[topic]['O']}\t{topic_posts_stat[topic]['IE']}\t{topic_posts_stat[topic]['IS']}\t{topic_posts_stat[topic][' ']}")