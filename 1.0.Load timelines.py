import sys
sys.path.append('codes')
from plots import *
# Prepare the function to load the data
from utils.job import get_job_config
import os
# Load Dataset
import pickle
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
processed_timeline_path = f"{processed_dir}/processed_timelines.pkl"

f = open(processed_timeline_path,'rb')
timelines = pickle.load(f)
f.close()

user_timeline = {}
all_posts = {}

for author in timelines:
    if author not in user_timeline:
        user_timeline[author] = {}
    
    for topic in timelines[author]:
        if topic not in user_timeline[author]:
            user_timeline[author][topic] = []
        
        for data in timelines[author][topic]:
            comments = get_post(data) 
            print(author,topic,len(data), len(comments))
            all_posts.update(comments)
            user_timeline[author][topic].append(comments)

#Get statistics
author_posts = {}  # index by author name
topic_posts = {} # index by topic name
timeline_sequence_post = {}

for author in user_timeline:   
    users_info = user_timeline[author]
    timeline_sequence_post[author] = {}
    
    for topic in users_info: #get user interaction stats
        if topic not in topic_posts:
            topic_posts[topic] = {}
        if topic not in timeline_sequence_post[author]:            
            timeline_sequence_post[author][topic] = []
                                            
        for comments in users_info[topic]:
            timeline = {'all_users_pid':[],'target_users_pid':[]}
            
            for pid in comments:
                cur_author = comments[pid]['author']
                cur_author = '_'.join(cur_author.split('_')[1:])
                
                timeline['all_users_pid'].append(pid)
                if author in comments[pid]['author']:
                    timeline['target_users_pid'].append(pid)
                
                if cur_author not in topic_posts[topic]:
                    topic_posts[topic][cur_author] = {} # get total author info in topic
                    topic_posts[topic][cur_author]['Target_user'] = False
                    
                if cur_author not in author_posts:
                    author_posts[cur_author] = {}
                    author_posts[cur_author]['Target_user'] = False
                
                if topic not in author_posts[cur_author]:
                    author_posts[cur_author][topic] = {}
                
                if author in cur_author:
                    topic_posts[topic][cur_author]['Target_user'] = True
                    author_posts[cur_author]['Target_user'] = True
                
                author_posts[cur_author][topic][pid] = comments[pid]
                topic_posts[topic][cur_author][pid] = comments[pid]
            timeline_sequence_post[author][topic].append(timeline)

f = open(f"{processed_dir}/processed_datasets_comments.pkl",'wb')
pickle.dump((user_timeline,all_posts,author_posts,topic_posts,timeline_sequence_post),f)
f.close()
