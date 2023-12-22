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

# Load topic specific csv files
only_author_post = False
folder_path = config["source_data_dir"] #Location of reddit directories
topics = {}

dir_files = os.listdir(folder_path)  
print('processed_dir', dir_files)

for topic in dir_files:
    csv_files = []
    file_path = os.path.join(folder_path,topic)
    if os.path.isfile(file_path) and os.path.splitext(file_path)[1] in ('.csv'):
        csv_files.append(file_path)
    if os.path.isdir(file_path):
        csv_files = csv_files + get_csv_path_from_folder(file_path)
    topics[topic] = csv_files

topic_titles = []
for topic in topics.keys():
    topic_titles.append(topic.lower())

processed_dir = f"{config['base_dir']}/data_processed"
processed_timeline_path = f"{processed_dir}/processed_timelines.pkl"

if os.path.exists(processed_timeline_path):
    f = open(processed_timeline_path,'rb')
    timelines = pickle.load(f)
    f.close()
else: 
    print('processed_dir', processed_dir)
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
        
    error_csv = []
    timelines = {}

    for topic in topics:
        for csv in topics[topic]:
            if 'conflicting ' not in csv and 'issue' not in csv:
                data_df = get_data_from_csv(csv, only_author_post, topic)
                author  = data_df.iloc[0,2]
                author='_'.join(author.split('_')[1:])
                data_df = data_df.fillna(method='ffill')
                if author not in timelines:
                    timelines[author] = {}

                if topic not in timelines[author]:
                    timelines[author][topic] = []  

                timelines[author][topic].append(data_df)
            else:
                error_csv.append(csv)
    
    f = open(processed_timeline_path,'wb')
    pickle.dump(timelines,f)
    f.close()

    print('Error CSV files:\n',error_csv)
