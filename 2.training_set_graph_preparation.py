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
f = open(f"{processed_dir}/processed_datasets_comments.pkl",'rb')

(user_timeline,all_posts,author_posts,topic_posts,timeline_sequence_post) = pickle.load(f)
f.close()

f = open(f"{processed_dir}/plots/users_interaction/user_post_statistics.pkl",'rb')
all_user_post_count = pickle.load(f)
f.close()

users_connections = {} #global
post2post = {} #local

for tuser in timeline_sequence_post:
    for topic in timeline_sequence_post[tuser]:
        for t in range(len(timeline_sequence_post[tuser][topic])):
            last_post_pid = ''

            tline = f'{topic}_{tuser}_{t}'
            post2post[tline] = {} #store post2post possible edgelist for timeline t 
            post2post[tline]['actual_connection'] = [] #store post2post based on reply

            node_in_network = {}
            pp_pid = ''
            parent_pid = ''
            
            first_post = timeline_sequence_post[tuser][topic][t]['all_users_pid'][0]
            target_author =  all_posts[first_post]['author']
            
            for ac, pid in enumerate(timeline_sequence_post[tuser][topic][t]['all_users_pid']):
                node_in_network[pid] =  False
                
                user = all_posts[pid]['author']
                user = '_'.join(user.split('_')[1:]) 
                
                if 'parent_id' in all_posts[pid]:
                    parent_pid = all_posts[pid]['parent_id']
                    
                if parent_pid not in all_posts and last_post_pid != '':
                    pp_pid = parent_pid
                    parent_pid = last_post_pid

                elif parent_pid not in all_posts and last_post_pid == '' and ac>0:
                    pp_pid = parent_pid
                    parent_pid = last_post_pid
                    
                elif ac==0:
                    parent_pid = pid
                    
                elif parent_pid not in all_posts:
                    pp_pid = parent_pid
                    parent_pid = last_post_pid
                    
                if user not in users_connections: 
                    users_connections[user] = []
                    
                if target_author == all_posts[pid]['author']:
                    last_post_pid = str(pid)
                                                                        
                author_parent =  all_posts[parent_pid]['author']
                author_parent = '_'.join(author_parent.split('_')[1:])
                if author_parent not in users_connections: 
                    users_connections[author_parent] = []

                users_connections[user].append(author_parent) # add cur_post user connections
                users_connections[author_parent].append(user)

                if pid != parent_pid:
                    post2post[tline]['actual_connection'].append((pid,parent_pid)) # parent to cur_post connection
                    node_in_network[pid] =  True
                    node_in_network[parent_pid] =  True
                else:
                    print('2 parent check',pid,parent_pid,pp_pid,tline)

f = open(f"{processed_dir}/processed_networks_final.pkl",'wb')
pickle.dump((users_connections,post2post),f)
f.close()

