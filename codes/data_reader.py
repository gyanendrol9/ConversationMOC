import os
import pandas as pd
import operator
import glob
import numpy as np
from regex import P
from sklearn.metrics import classification_report
import itertools

class DataReader:
    def __init__(self, keyBert_folder_path, clinical_folder_path, dataset_path):    
        # set the paths of the lexicon and dataset
        self.keyBert_folder_path = keyBert_folder_path
        self.clinical_folder_path = clinical_folder_path
        self.dataset_path = dataset_path
        
        # load lexicon, train and test dataset        
        self.lexicon = lexicon_load(self.keyBert_folder_path, self.clinical_folder_path)
        self.dataset = get_data_from_multiple_csvs(self.dataset_path)

    def load_rationales(self):
        self.dataset['rationale'] = self.dataset.apply(
            lambda row: get_rationale_label(row.body, self.lexicon), axis=1)
        return self.dataset
    
    def get_dataset(self):
        return self.dataset
   
# function to get files in a folder
def list_dir(file_dir):
    file_list = []
    dir_list = os.listdir(file_dir)
    
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        
        # if the path is  file then keep the path
        if os.path.isfile(path):
            csv_file = os.path.join(file_dir, cur_file)
            file_list.append(csv_file)
            
    return file_list

# read keyBert dictionary
def read_keyBert(path):
    print("open file: ", path)
    lexicon = []
    f = open(path, 'r')
    line = f.readline()
    while line:
        line = line.strip().split(';')
        line = [s.lower() for s in line]
        line = [x.strip() for x in line if x.strip() != '']	
        lexicon = lexicon + line
        line = f.readline()
        
    f.close()
    return lexicon

# read clinical symptoms dictionary
def read_clinical(path):
    lexicon = []
    f = open(path, 'r')

    try:
        line = f.readline()
    except Exception as e:
        print(e)
        print("error file: ", path)
        line = ""

    while line:
        try:
            if not len(line) or line.startswith('#'):
                line = f.readline()     
                continue    
            line = line.strip('\n')
            lexicon.append(line.lower())
            line = f.readline()
        except Exception as e:
            print(e)
            print("error line: ", line)
            line = f.readline()
        
    f.close()
    return lexicon

# function to load lexicon
def lexicon_load(keyBert_folder_path, clinical_folder_path):
    lexicon = []
    
    # Load keyBert Dictionary
    dic_files = get_keyBert_path_from_folder(keyBert_folder_path)
    print("keyBert files: ", dic_files)
    keyBert_lexicon = []
    clinical_lexicon = []
    
    for f in dic_files:
        keyBert = read_keyBert(f)
        # print("keyBert dictionary: ", keyBert)
        keyBert_lexicon += keyBert

    # Load Clinical Symptoms Dictionary
    dic_files = list_dir(clinical_folder_path)
    for f in dic_files:
        clinical = read_clinical(f)
        # print("clinical symptoms dictionary: ", clinical)
        clinical_lexicon += clinical

    lexicon = keyBert_lexicon + clinical_lexicon
    lexicon = [x for x in lexicon if x != '']
    lexicon = [x for x in lexicon if x != ' ']
    return lexicon

def get_rationale_label(post, lexicon):
    for phrase in lexicon:
        if operator.contains(post.lower(), phrase):
            return 1
    return 0


def get_post(comments):
    posts = {}
    
    t_author = comments.iloc[0]['author']
    for idx in range(len(comments)):
        author = comments.iloc[idx]['author']
        post_type = comments.iloc[idx]['post_type']

        if post_type == 'post':
            post_id = comments.iloc[idx]['post id']
            if post_id not in posts:
                posts[post_id] = {}
                posts[post_id]['post_type'] = comments.iloc[idx]['post_type']
                posts[post_id]['sentences'] = comments.iloc[idx]['sentences']
            else:
                posts[post_id]['sentences'] += '\n'+comments.iloc[idx]['sentences']

        elif post_type == 'comment':
            post_id = comments.iloc[idx]['comment id']
            if post_id not in posts:
                posts[post_id] = {}
                posts[post_id]['post_type'] = comments.iloc[idx]['post_type']
                posts[post_id]['parent_id'] = comments.iloc[idx]['parent_id'].split('_')[1]
                posts[post_id]['sentences'] = comments.iloc[idx]['sentences']
            else:
                posts[post_id]['sentences'] += '\n'+comments.iloc[idx]['sentences']
        else:
            print('error in conversation file',idx)
            
        posts[post_id]['mood'] = comments.iloc[idx]['mood']
        posts[post_id]['momentofchange'] = comments.iloc[idx]['alphabetic maj vote']
        posts[post_id]['author'] = author
        if t_author == author:
            posts[post_id]['Target_user'] = True
        else:
            posts[post_id]['Target_user'] = False
        
    return posts


#Add rationale label for each post
def load_rationales(dataset,lexicon):
    dataset['rationale'] = dataset.apply(
        lambda row: get_rationale_label(row.sentences, lexicon), axis=1)
    return dataset

# Read data from csv file
def get_data_from_csv(csv,only_author_post,topic):
    data_df = pd.read_csv(csv)
    data_df = data_df.fillna(method='ffill')
    # data_df = data_df.dropna(axis=0,how='any')
    author  = data_df.iloc[0,2]

    # clean instances which are not author's posts/comments
    if only_author_post:
        data_df = data_df.loc[data_df['author'].str.lower() == author.lower()]
        
    data_df['author'] = data_df['subreddit']+'_'+data_df['author']
    # create id using post_id and comment_id into one column
    comment_df = data_df.loc[data_df['post_type'] == 'comment']
    
    if len(comment_df) > 0:
        comment_df['id'] = comment_df['post id'].str.strip() +'_C_'+ comment_df['comment id'].str.strip() 

    post_df = data_df.loc[data_df['post_type'] == 'post']
    post_df['id'] =  post_df['post id'].str.strip() 
    data_df = pd.concat([post_df, comment_df])
    data_df = data_df.sort_index(ascending=True)
    return data_df
    

# Read csvs files in a folder
def get_data_from_multiple_csvs(folder_path):
    data_df = pd.DataFrame()
    csv_files = get_csv_path_from_folder(folder_path)
    print("csv files: ", len(csv_files))
    for csv in csv_files:
        try:
            data_df = pd.concat([data_df, get_data_from_csv(csv)])
        except Exception as e:
            print(e)
            print("error file: ", csv)
    return data_df
    
# function to get paths for csv files in the folder
def get_csv_path_from_folder(folder_path):
    csv_files = []
    dir_files = os.listdir(folder_path)  
    for file in dir_files:
        file_path = os.path.join(folder_path,file)
        if os.path.isfile(file_path) and os.path.splitext(file_path)[1] in ('.csv'):
                csv_files.append(file_path)
        if os.path.isdir(file_path):
                csv_files = csv_files + get_csv_path_from_folder(file_path)
    return csv_files

# Read keyBert dictionaries from a folder
def get_keyBert_path_from_folder(folder_path):
    csv_files = []
    dir_files = os.listdir(folder_path)  
    for file in dir_files:
        file_path = os.path.join(folder_path,file)
        if os.path.isfile(file_path):
                csv_files.append(file_path)
        if os.path.isdir(file_path):
                csv_files = csv_files + get_keyBert_path_from_folder(file_path)
    return csv_files
    
#------------------------------new-addition----------------------------#
def get_dataset(train_ids, timeline_pids, x_seq_all, x_net_all, y_train_all, users_flag_all):
    x_train = []
    x_net_train = []
    y_train = []
    uf_train = []
    x_pids = []

    for id in train_ids:
        x_pids.append(timeline_pids[id])
        x_train.append(x_seq_all[id])
        x_net_train.append(x_net_all[id])
        y_train.append(y_train_all[id])
        uf_train.append(users_flag_all[id])
    return np.asarray(x_train), np.asarray(x_net_train), np.asarray(y_train),uf_train, x_pids

def get_tuser_only_data(x_feat, y_label, tu_flag, x_pids, tu_len = 0):
    tuser_only_feat = []
    tuser_only_moc = []
    tuser_only_pid = []
    max_lens = []
    
    for i,uflag in enumerate(tu_flag):
        err = 0
        feats = []
        mocs = []
        pids = []
        for j, flag in enumerate(uflag):
            if flag:
                feats.append(x_feat[i][j])
                mocs.append(y_label[i][j][0:4])
                pids.append(x_pids[i][j])

        tuser_only_feat.append(feats)
        tuser_only_moc.append(mocs)
        tuser_only_pid.append(pids)
        max_lens.append(len(mocs))
        
    if tu_len == 0:
        tu_len = max(max_lens)

    tu_x_train_padded = []
    for i in tuser_only_feat:
        padded_vec = padding_node_feat(np.asarray(i), tu_len,9)#
        tu_x_train_padded.append(padded_vec)
    tu_x_train_padded = np.asarray(tu_x_train_padded)

    tu_y_train_padded = []
    for i in tuser_only_moc:
        padded_vec = padding_node_feat(np.asarray(i), tu_len,0)#
        for j in range(len(i),tu_len):
            padded_vec[j][3] = 1 #padded vector label
        tu_y_train_padded.append(padded_vec)
    tu_y_train_padded = np.asarray(tu_y_train_padded)
        
    return tu_x_train_padded, tu_y_train_padded, tu_len, tuser_only_pid

def get_data_till_tuser_lastpost(x_feat, x_net, y_label, tu_flag, x_pids, tu_len = 0):
    max_lens = []
    for i, uflag in enumerate(tu_flag):
        for idx, val in reversed(list(enumerate(uflag))):
            if val:
                max_lens.append(idx)
                print("Index of last True value:", idx)
                break
    if tu_len == 0:
        tu_len = max(max_lens)

    tuser_only_pid = []
    for i, uflag in enumerate(tu_flag):
        err = 0
        pids = []
        for j in range(len(x_pids[i])):
            if j < tu_len:
                pids.append(x_pids[i][j])
        tuser_only_pid.append(pids)

    tuser_only_feat = []
    tuser_only_moc = []
    tuser_only_net = []

    for id in range(len(x_feat)):
        tuser_only_feat.append(x_feat[id][:tu_len])
        tuser_only_net.append(x_net[id][:tu_len, :tu_len])
        tuser_only_moc.append(y_label[id][:tu_len])
    return np.asarray(tuser_only_feat), np.asarray(tuser_only_net), np.asarray(tuser_only_moc), tu_len, tuser_only_pid


def get_split_dataset(whole_dataset, train_ids, test_ids):
    x_train = []
    x_test = []
    
    for id in train_ids:
        x_train.append(whole_dataset[id])
    
    for id in test_ids:
        x_test.append(whole_dataset[id])

    return np.asarray(x_train), np.asarray(x_test)


def get_senti_emotion_scores(embb):
    emotion = embb[1344:1348]
    sent = embb[-3:]
    return emotion.argmax(), sent.argmax()

# post embedding 
def get_node_features(all_posts, post_ids):
    node_feat = []
    for pid in post_ids:
        author = all_posts[pid]['author'].split('_')
        # author = '_'.join(author[1:])
        emb = all_posts[pid]['embedding']
        node_feat.append(emb) 
    return np.asarray(node_feat)


# Create post2post graph
def get_adjacency_matrix(timeline, sequence_pid, post2post, all_posts):
    post_ids = sequence_pid[timeline] #check post count

    post_net = post2post[timeline]['actual_connection']
    A_id_post_net = {} 
    users = []
    for pid in post_ids:
        A_id_post_net[pid] = len(A_id_post_net)
        author =  '_'.join(all_posts[pid]['author'].split('_')[1:])
        users.append(author)
    
    A = np.zeros((len(A_id_post_net), len(A_id_post_net)))
    
    for n1, n2 in post_net:
        A[A_id_post_net[n1]][A_id_post_net[n2]] += 1
        A[A_id_post_net[n2]][A_id_post_net[n1]] += 1

    return A, A_id_post_net,users

# Create targetuser-nontargetuser graph
def get_adjacency_matrix_userflag(user_flag): 
    A = np.zeros((len(user_flag), len(user_flag)))
    
    post_net = {}
    post_net['Target'] = []
    post_net['NonTarget'] = []
    
    for n1, flag in enumerate(user_flag):
        if flag:
            post_net['Target'].append(n1)
        else:
            post_net['NonTarget'].append(n1)
                
    # generate all possible combinations of 2 items
    combinations = list(itertools.combinations(post_net['Target'], 2))
    for n1, n2 in combinations:
        A[n1][n2] += 1
        A[n2][n1] += 1

    # generate all possible combinations of 2 items
    combinations = list(itertools.combinations(post_net['NonTarget'], 2))
    for n1, n2 in combinations:
        A[n1][n2] += 1
        A[n2][n1] += 1
        
    return A

def get_adjacency_matrix_emotion_sentiment_emb(x_embs): 
    EA = np.zeros((len(x_embs), len(x_embs)))
    SA = np.zeros((len(x_embs), len(x_embs)))

    e_net = {}
    s_net = {}
    for p, embb in enumerate(x_embs):
        e_id, s_id = get_senti_emotion_scores(embb)
        if e_id not in e_net:
            e_net[e_id] = []
        if s_id not in s_net:
            s_net[s_id] = []
        e_net[e_id].append(p)
        s_net[s_id].append(p)

    for c in e_net:
        combinations = list(itertools.combinations(e_net[c], 2))
        for n1, n2 in combinations:
            EA[n1][n2] += 1
            EA[n2][n1] += 1

    for c in s_net:
        combinations = list(itertools.combinations(s_net[c], 2))
        for n1, n2 in combinations:
            SA[n1][n2] += 1
            SA[n2][n1] += 1
        
    return EA, SA

# Create targetuser-nontargetuser graph
def get_adjacency_matrix_emotion_sentiment(pids, all_posts): 
    EA = np.zeros((len(pids), len(pids)))
    SA = np.zeros((len(pids), len(pids)))

    e_net = {}
    s_net = {}
    for p, pid in enumerate(pids):
        embb = all_posts[pid]['embedding']
        e_id, s_id = get_senti_emotion_scores(embb)
        if e_id not in e_net:
            e_net[e_id] = []
        if s_id not in s_net:
            s_net[s_id] = []
        e_net[e_id].append(p)
        s_net[s_id].append(p)

    for c in e_net:
        combinations = list(itertools.combinations(e_net[c], 2))
        for n1, n2 in combinations:
            EA[n1][n2] += 1
            EA[n2][n1] += 1

    for c in s_net:
        combinations = list(itertools.combinations(s_net[c], 2))
        for n1, n2 in combinations:
            SA[n1][n2] += 1
            SA[n2][n1] += 1
        
    return EA, SA


# Padding functions
def padding_adj_matrix(matrix, desired_rows, constant_value = 0):
    # Define the desired padding size
    padding_size = max(0, desired_rows - matrix.shape[0])

    # Pad the matrix only on the right and below
    padded_matrix = np.pad(matrix, pad_width=((0, padding_size), (0, padding_size)), mode='constant', constant_values=constant_value)
    return padded_matrix

def padding_node_feat(matrix, desired_rows, constant_value=0):
    # Calculate the amount of padding needed in terms of rows
    pad_rows = max(0, desired_rows - matrix.shape[0])

    # Pad the matrix with rows of 9's at the bottom
    if pad_rows > 0:
        padded_matrix = np.pad(matrix, ((0, pad_rows), (0, 0)), mode='constant', constant_values=constant_value)
    else:
        padded_matrix = matrix
    return padded_matrix

def padding_vector(vector,pad_length,target_user):
    
    # Pad the vector with zeros at the end
    if target_user:
        padded_vector = np.pad(vector, (0, pad_length), mode='constant', constant_values=1)
    else:
        padded_vector = np.pad(vector, (0, pad_length), mode='constant', constant_values=0)

    return padded_vector

def pad_user_flag(timeline_users_flag, max_len):
    timeline_padded_users_flag = []
    for userflag in timeline_users_flag:
        userflag2 = list(userflag)
        for i in range(len(userflag), max_len):
            userflag2.append(False)
        timeline_padded_users_flag.append(userflag2)
    return timeline_padded_users_flag

def get_pred_labels(tu_y_test, result, tu_pids, fpath, idx2moc):
    y_true = []
    y_pred = []
    
    f = open(fpath, 'w')
    f.write('postID\tTrue Label\tPred Label\n')
    for si, sess in enumerate(tu_y_test):
        si_con = []
        for pi, post in enumerate(sess): 
            idx = int(np.argmax(tu_y_test[si][pi]))
            if idx < 3:
                y_true.append(idx)
                pid = tu_pids[si][pi]
                
                idx = int(np.argmax(result[si][pi]))
                if idx > 2:
                    y_pred.append(0)
                    idx = 0
                else:                    
                    y_pred.append(idx)
                
                true_a = idx2moc[np.argmax(tu_y_test[si][pi])]
                pred_a = idx2moc[idx]
                f.write(f'{tu_pids[si][pi]}\t{true_a}\t{pred_a}\n')
                 
    print(classification_report(y_true, y_pred))
    return np.asarray(y_true), np.asarray(y_pred)

def get_pred_scores(tu_y_test, result):
    y_true = []
    y_pred = []
    
    for si, sess in enumerate(tu_y_test):
        si_con = []
        for pi, post in enumerate(sess): 

            idx = int(np.argmax(tu_y_test[si][pi]))
            if idx < 3:
                y_true.append(idx)
                idx = int(np.argmax(result[si][pi]))
                if idx > 2:
                    y_pred.append(0)
                else:                    
                    y_pred.append(idx)
                 
    print(classification_report(y_true, y_pred))

def get_pred_scores_task2(y2_test, result):
    y_true = []
    y_pred = []
    
    for si, sess in enumerate(y2_test):
        idx = int(np.argmax(y2_test[si]))
        y_true.append(idx)
        idx = int(np.argmax(result[si]))  
        y_pred.append(idx)
                 
    print(classification_report(y_true, y_pred))


# Label statistics
def calculate_statistics_task1(y_train_all):
    total_posts = 0
    for i, moc in enumerate(y_train_all):
        if i == 0:
            # calculate the sum of each column
            col_sums = np.sum(moc, axis=0)
        else:
            col_sums+= np.sum(moc, axis=0)

        total_posts+=len(moc)

    print('\n\nData stats:\nOther\tIE\tIS\tNon-target')
    print(f'{col_sums[0]}\t{col_sums[1]}\t{col_sums[2]}\t{col_sums[3]}\n') 
    print(f'#Conversations: {i}\n')

    
# Label statistics
def calculate_statistics_task2(y2_train_all, topic_labels):
    col_sums = np.sum(y2_train_all, axis=0)
    for i, topic in enumerate(topic_labels):
        print(f'{topic}\t{col_sums[i]}')



def get_pred_scores_dict(tu_y_test, result):
    y_true = []
    y_pred = []
    
    for si, sess in enumerate(tu_y_test):
        si_con = []
        for pi, post in enumerate(sess): 

            idx = int(np.argmax(tu_y_test[si][pi]))
            if idx < 3:
                y_true.append(idx)
                idx = int(np.argmax(result[si][pi]))
                if idx > 2:
                    y_pred.append(0)
                else:                    
                    y_pred.append(idx)
                 
    return classification_report(y_true, y_pred, labels=[0, 1, 2], output_dict=True)

def get_pred_scores_task2_dict(y2_test, result):
    y_true = []
    y_pred = []
    
    for si, sess in enumerate(y2_test):
        idx = int(np.argmax(y2_test[si]))
        y_true.append(idx)
        idx = int(np.argmax(result[si]))  
        y_pred.append(idx)
                 
    return classification_report(y_true, y_pred, output_dict=True)

def get_score_json(scores_dict, scores):
    for label in scores_dict:
        if label not in scores and label != 'accuracy':
            scores[label] = {}
            scores[label]['precision'] = []
            scores[label]['recall'] = []
            scores[label]['f1-score'] = []
        elif label not in scores:
            scores[label] = []

        if label != 'accuracy':
            for score in scores_dict[label]:
                if score != 'support':
                    scores[label][score].append(scores_dict[label][score])
        else:
            scores[label].append(scores_dict[label])
#         print(f'{label}\t{scores[label]}')
    return scores

def kfold_scores(scores, idx2moc):
    results = {}
    for kfold in scores:
        scores_dict = scores[kfold]
        for score_key in scores_dict:
            if score_key != 'accuracy':
                if score_key.isnumeric():
                    score_key_label = idx2moc[int(score_key)]
                else:
                    score_key_label = score_key


                if score_key_label not in results:
                    results[score_key_label] = []

                data = [scores_dict[score_key]['precision'], scores_dict[score_key]['recall'], scores_dict[score_key]['f1-score']]
                results[score_key_label].append(data) 
            else:
                if score_key not in results:
                    results[score_key] = []
                results[score_key].append(scores_dict[score_key])         
    return results

def print_kfold_results(results,outpath=''):
    if len(outpath)>0:
        f = open(outpath,'w')
        r_write = True

    for print_key in results:
        if print_key != 'accuracy':
            if r_write:
                f.write(f'\n---------{print_key}---------\tAccuracy\tPrecision\tRecall\tF-score\n')
            else:
                print(f'\n---------{print_key}---------')
            for kfold in range(len(results[print_key])):
                res = results[print_key][kfold]
                if r_write:
                    f.write(f"Fold {kfold}\t{results['accuracy'][kfold]}\t{res[0]}\t{res[1]}\t{res[2]}\n")
                else:
                    print(f"Fold {kfold}\t{results['accuracy'][kfold]}\t{res[0]}\t{res[1]}\t{res[2]}")
            # print('\n')

