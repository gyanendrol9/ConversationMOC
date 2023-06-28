import sys
sys.path.append('codes')
from plots import *
# Prepare the function to load the data
from utils.job import get_job_config
import os
# Load Dataset
import pickle
from data_reader import *

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


#Plot statistics
import matplotlib.pyplot as plt
plt.style.use("ggplot")


plot_dir = f"{processed_dir}/plots"
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
    os.mkdir(f"{plot_dir}/users_interaction")
    os.mkdir(f"{plot_dir}/conversations")
    os.mkdir(f"{plot_dir}/subreddits")

total_author_topic_count = [len(topic_posts[topic]) for topic in topic_posts]
total_topics_id = [topic for topic in topic_posts]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/subreddits/all_users_per_subreddits.pdf"
plt = plot_gen(total_topics_id,total_author_topic_count,'Subreddits','# of Users',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)

#Target users per topics
total_target_users_topic_count = {}
topic_target_users_post_count = {}
topic_all_users_post_count = {}

for topic in topic_posts:
    total_target_author = 0
    
    for author in topic_posts[topic]:
        if topic_posts[topic][author]['Target_user']:
            total_target_author += 1
            
    total_target_users_topic_count[topic] = total_target_author

total_target_author_topic_count = [total_target_users_topic_count[topic] for topic in topic_posts]
total_topics_id = [topic for topic in topic_posts]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/subreddits/all_target_users_per_subreddits.pdf"
plt = plot_gen(total_topics_id,total_target_author_topic_count,'Subreddits','# of Users',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)

# Post counts per topic
topic_posts_count_all = {}
topic_posts_count_target_user = {}
for topic in topic_posts:
    topic_posts_count_all[topic] = 0
    topic_posts_count_target_user[topic] = 0
    for author in topic_posts[topic]:
        topic_posts_count_all[topic] += len(topic_posts[topic][author])-1
        if topic_posts[topic][author]['Target_user']:
            topic_posts_count_target_user[topic] += len(topic_posts[topic][author])-1

total_topic_post_count = [topic_posts_count_all[topic] for topic in topic_posts]
total_topics_id = [topic for topic in topic_posts]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/subreddits/num_posts_wrt_subreddits-all-user.pdf"
plt = plot_gen(total_topics_id,total_topic_post_count,'Subreddits','# of posts',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)
# plt.show()

total_topic_post_count = [topic_posts_count_target_user[topic] for topic in topic_posts]
total_topics_id = [topic for topic in topic_posts]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/subreddits/num_posts_wrt_subreddits-target-user.pdf"
plt = plot_gen(total_topics_id,total_topic_post_count,'Subreddits','# of posts',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)
# plt.show()

filepath = f"{processed_dir}/plots/subreddits/all_users_per_subreddits.csv"
f = open(filepath, 'w')
f.write('Subreddits\t#Users\t#Posts\t#Target Users\t#Posts\n')
for topic in topic_posts:
    f.write(f'{topic}\t{len(topic_posts[topic])}\t{topic_posts_count_all[topic]}\t{total_target_users_topic_count[topic]}\t{topic_posts_count_target_user[topic]}\n')
f.close()

f = open(f"{processed_dir}/topic_post_statistics.pkl",'wb')
pickle.dump((topic_posts_count_all,topic_posts_count_target_user),f)
f.close()

# Author post statistics
all_user_post_count = {}

for pid in all_posts:
    author = all_posts[pid]['author'].split('_')
    topic = author[0].lower()
    author = '_'.join(author[1:])
    
    if author not in all_user_post_count:
        all_user_post_count[author] = {}
        all_user_post_count[author] = {}
        all_user_post_count[author]['all'] = 0
        all_user_post_count[author]['Target_user'] = 0
    
        all_user_post_count[author]['Conversations'] = {}   
           
    if topic not in all_user_post_count[author]['Conversations']:
        all_user_post_count[author]['Conversations'][topic] = {}
        all_user_post_count[author]['Conversations'][topic]['all'] = 0
        all_user_post_count[author]['Conversations'][topic]['Target_user'] = 0
        
    all_user_post_count[author]['Conversations'][topic]['all'] +=1
    all_user_post_count[author]['all'] +=1

    if all_posts[pid]['Target_user']:
        all_user_post_count[author]['Conversations'][topic]['Target_user'] +=1
        all_user_post_count[author]['Target_user']+=1
        
sorted_all_user_post_count = {}
for author in all_user_post_count:
    sorted_all_user_post_count[author] = all_user_post_count[author]['all']
    
sorted_all_user_post_count = dict(sorted(sorted_all_user_post_count.items(), key=lambda item: item[1], reverse=True))

top_posting_users = list(sorted_all_user_post_count.keys())
total_user_post_count = [all_user_post_count[author]['all'] for author in sorted_all_user_post_count]
total_author_id = [id for id in range(len(all_user_post_count))]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/users_interaction/all_user_posts_count.pdf"
plt = plot_gen(total_author_id,total_user_post_count,'Users','# of posts',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)
# plt.show()

total_user_post_count = [all_user_post_count[author]['Target_user'] for author in sorted_all_user_post_count]
total_author_id = [id for id in range(len(all_user_post_count))]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/Target_users_posts_count.pdf"
plt = plot_gen(total_author_id,total_user_post_count,'Users','# of posts',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)
# plt.show()

sorted_all_user_topic_count = {}
for author in all_user_post_count:
    sorted_all_user_topic_count[author] = len(all_user_post_count[author]['Conversations'])
    
sorted_all_user_topic_count = dict(sorted(sorted_all_user_topic_count.items(), key=lambda item: item[1], reverse=True))

total_user_topics_count = [sorted_all_user_topic_count[author] for author in sorted_all_user_topic_count]
total_author_id = [id for id in range(len(sorted_all_user_topic_count))]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/users_interaction/all_users_topic_interactions_count.pdf"
plt = plot_gen(total_author_id,total_user_topics_count,'Users','# of Subreddits',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)
# plt.show()

total_user_topics_count = {}
for author in sorted_all_user_post_count:
    if author_posts[author]['Target_user']:
        total_user_topics_count[author] = sorted_all_user_topic_count[author]
    

total_user_topics_count = dict(sorted(total_user_topics_count.items(), key=lambda item: item[1], reverse=True))
user_topics_count = [total_user_topics_count[author] for author in total_user_topics_count]
total_author_id = [id for id in range(len(total_user_topics_count))]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/users_interaction/target_users_topic_interactions_count.pdf"
plt = plot_gen(total_author_id,user_topics_count,'Users','# of Subreddits',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)
# plt.show()

f = open(f"{processed_dir}/plots/users_interaction/user_post_statistics.pkl",'wb')
pickle.dump((top_posting_users, all_user_post_count),f)
f.close()

# Number of posts and users per conversations
conversation_count = {}
for tuser in timeline_sequence_post:
    for topic in timeline_sequence_post[tuser]:
        for t in range(len(timeline_sequence_post[tuser][topic])):
            tline = f'{topic}_{tuser}_{t}'
            if tline not in conversation_count:
                conversation_count[tline] = {}
                conversation_count[tline]['Post'] = {}
                conversation_count[tline]['Post']['all'] = {}
                conversation_count[tline]['Post']['Target_user'] = {}
                conversation_count[tline]['Users'] = 0
                
            conversation_count[tline]['Post']['all'] = len(timeline_sequence_post[tuser][topic][t]['all_users_pid'])
            conversation_count[tline]['Post']['Target_user'] = len(timeline_sequence_post[tuser][topic][t]['target_users_pid'])
            
            user = []
            for pid in timeline_sequence_post[tuser][topic][t]['all_users_pid']:
                user.append(all_posts[pid]['author'])
            conversation_count[tline]['Users'] = len(set(user))

f = open(f"{processed_dir}/plots/conversations/conversation_statistics.pkl",'wb')
pickle.dump(conversation_count,f)
f.close()

sorted_all_conversation_count = {}
for tline in conversation_count:
    sorted_all_conversation_count[tline] = conversation_count[tline]['Post']['Target_user']
    
sorted_all_conversation_count = dict(sorted(sorted_all_conversation_count.items(), key=lambda item: item[1], reverse=True))
sorted_conversation_count = list(sorted_all_conversation_count.keys())

total_posts_conversations_count = [conversation_count[tline]['Post']['Target_user'] for tline in sorted_conversation_count]
total_author_id = [id for id in range(len(sorted_conversation_count))]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/conversations/target_users_posts_conversation_count.pdf"
plt = plot_gen(total_author_id,total_posts_conversations_count,'Conversations','# of posts',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)
# plt.show()

total_posts_conversations_count = [conversation_count[tline]['Post']['all'] for tline in sorted_conversation_count]
total_author_id = [id for id in range(len(sorted_conversation_count))]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/conversations/all_users_posts_conversation_count.pdf"
plt = plot_gen(total_author_id,total_posts_conversations_count,'Conversations','# of posts',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)
# plt.show()

total_posts_conversations_count = [conversation_count[tline]['Users'] for tline in sorted_conversation_count]
total_author_id = [id for id in range(len(sorted_conversation_count))]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/conversations/all_users_conversation_count.pdf"
plt = plot_gen(total_author_id,total_posts_conversations_count,'Conversations','# of users',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)
# plt.show()

# Count number of conversations wrt #posts/conversations
conversations_wrt_posts_count = {}

for tline in sorted_conversation_count: 
    numposts = conversation_count[tline]['Post']['all'] 
    if numposts not in conversations_wrt_posts_count:
        conversations_wrt_posts_count[numposts] = 0
    conversations_wrt_posts_count[numposts]+=1

pcounts = set(conversations_wrt_posts_count.keys())

total_conversations_wrt_posts_count = [conversations_wrt_posts_count[numposts] for numposts in pcounts]
total_author_id = [id for id in pcounts]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/conversations/num_conversations_wrt_num_posts_all_users.pdf"
plt = plot_gen(total_author_id,total_conversations_wrt_posts_count,'# of posts/conversation','# of Conversations',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)
# plt.show()

conversations_wrt_posts_count = {}

for tline in sorted_conversation_count: 
    numposts = conversation_count[tline]['Post']['Target_user'] 
    if numposts not in conversations_wrt_posts_count:
        conversations_wrt_posts_count[numposts] = 0
    conversations_wrt_posts_count[numposts]+=1

pcounts = set(conversations_wrt_posts_count.keys())

total_conversations_wrt_posts_count = [conversations_wrt_posts_count[numposts] for numposts in pcounts]
total_author_id = [id for id in pcounts]

fig = plt.figure(figsize=(20, 12))  # Width=8 inches, Height=6 inches
filepath = f"{processed_dir}/plots/conversations/num_conversations_wrt_num_posts_target_users.pdf"
plt = plot_gen(total_author_id,total_conversations_wrt_posts_count,'# of posts/conversation','# of Conversations',axis_fontsize = 10, label_fontsize = 15, rotate = 'right', degree = 45,filepath=filepath, save_fig=True)
# plt.show()