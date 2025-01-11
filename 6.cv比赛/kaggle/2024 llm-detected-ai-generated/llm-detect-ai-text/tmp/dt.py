import os
import re
import pickle
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict

import pickle
from leven_search import LevenSearch, EditCost, EditCostConfig, GranularEditCostConfig


# In your notebook, use the following path:
# `/kaggle/usr/lib/install_levenshtein_search_library/leven_search.pkl'
with open('/kaggle/input/tmp123/leven_search.pkl', 'rb') as file:
    lev_search = pickle.load(file)
    
    
test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
sub = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')
# org_train = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')

train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')

def sentence_correcter(text):
    wrong_words = []
    correct_words = dict()
    word_list = re.findall(r'\b\w+\b|[.,\s]', text)
    
    for t in word_list:
        correct_word = t

        if len(t)>2:
            #result = lev_search.find_dist(t, max_distance=0)
            #result = list(result.__dict__['words'].values())
            if not lev_search.find(t):
                result = lev_search.find_dist(t, max_distance=1)
                result = list(result.__dict__['words'].values())

                if len(result) == 0:
                    result = lev_search.find_dist(t, max_distance=1)
                    result = list(result.__dict__['words'].values())
                if len(result):
                    correct_word = result[0].word
                    wrong_words.append((t, result))

        correct_words[t] = correct_word

    dict_freq = defaultdict(lambda :0)           
    for wrong_word in wrong_words:
        _, result = wrong_word

        for res in result:
            updates = res.updates
            parts = str(updates[0]).split(" -> ")
            if len(parts) == 2:
                from_char = parts[0]
                to_char = parts[1]
                dict_freq[(from_char, to_char)] += 1

    if len(dict_freq):
        max_key = max(dict_freq, key=dict_freq.get)
        count = dict_freq[max_key]
    else:
        count = 0

    if count > 0.06*len(text.split()):
        gec = GranularEditCostConfig(default_cost=10, edit_costs=[EditCost(max_key[0], max_key[1], 1)])

        for wrong_word in wrong_words:
            word, _ = wrong_word
            result = lev_search.find_dist(word, max_distance=9, edit_cost_config=gec)
            result = list(result.__dict__['words'].values())
            if len(result):
                correct_words[word] = result[0].word
            else:
                correct_word = word


    correct_sentence = []
    for t in word_list:
        correct_sentence.append(correct_words[t])

    return "".join(correct_sentence)

def deal_row(row):
    
    return sentence_correcter(row['text'])
    
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    results1 = list(executor.map(deal_row,test.to_dict('records')))
    results2 = list(executor.map(deal_row,train.to_dict('records')))
    
test['correct_text'] = results1
train['correct_text'] = results2

test.to_csv('/kaggle/working/test_essays1.csv', index=False)
train.to_csv('/kaggle/working/train_essays1.csv', index=False)
