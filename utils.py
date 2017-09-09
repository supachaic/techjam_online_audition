

import os
import json
import datetime
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score
from sklearn.metrics import precision_score, f1_score, roc_auc_score


def write_result(predict, label_list, timestamp):
    
    save_path = 'results/'   
    filename = os.path.join(save_path, timestamp + '.txt')
    
    results = []    
    for result in predict:
        results.append(label_list[result])

    results = pd.DataFrame(results)
    results.to_csv(filename, header=None, index=False)
    print('Saved result file: %s' % filename)



def write_params(estimator_name, best_params, 
                 best_score, val_score, 
                 timestamp, num):
    save_params = {
            'timestamp': timestamp,
            'estimator': estimator_name,
            'best_score': best_score,
            'val_score': val_score,
            'best_params': best_params
    }
    
    filename = 'params/%s%.2d_params.json' % (estimator_name, num)
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            json.dump(save_params, f)
            print('Saved params: %s at %s' % (timestamp, filename))
    else:
        with open(filename, 'r') as f:
            params = json.load(f)                
        
        if params['val_score'] < save_params['val_score']:
            with open(filename, 'w') as f:
                json.dump(save_params, f)
                print('Saved params: %s at %s' % (timestamp, filename)) 

def write_features(features, estimators_name):
    df = pd.DataFrame(features.transpose(), columns=estimators_name)
    df['avg'] = df.mean(axis=1)
    
    try:
        col_file = 'data/col_name.csv'
        df_col = pd.read_csv(col_file, header=None)
        df_col.columns = ['features']
        df = pd.concat([df_col, df], axis=1)
    except: FileNotFoundError
    
    timestamp = get_timestamp()
    feature_file = os.path.join('features/', timestamp +'.csv')
    df.to_csv(feature_file)    
    print('write features score at %s' % feature_file)

def get_timestamp():
    return f"{datetime.datetime.now():%Y%m%d%H%M}"

def get_params(estimator_name, num):
    filename = 'params/%s%s_params.json' % (estimator_name, num)
    with open(filename, 'r') as f:        
        params = json.load(f)
    return params['best_params'], params['val_score'], params['timestamp']


def get_nums(estimator_name):
    nums = []
    for filename in os.listdir('params/'):
        if 'params.json' in filename:
            head = filename.split('_')
            if estimator_name in head[0]:
                num = head[0].replace(estimator_name,'')
                try:
                    int(num)
                    nums.append(num)
                except: ValueError    
    return nums



def matric_score(y_val, y_pred, score_name):
    if score_name == 'accuracy':
        return accuracy_score(y_val, y_pred)
    elif score_name == 'precision':
        return precision_score(y_val, y_pred)
    elif score_name == 'recall':
        return recall_score(y_val, y_pred)
    elif score_name == 'f1':
        return f1_score(y_val, y_pred)
    elif score_name == 'roc_auc' or score_name == 'roc':
        return roc_auc_score(y_val, y_pred)
    elif score_name == 'cohen_kappa' or score_name == 'kappa':
        return cohen_kappa_score(y_val, y_pred)
    


class write_logs(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)



def all_subs(idx_list):
    if idx_list == []:
        return [[]]

    indices = all_subs(idx_list[1:])

    return indices + [[idx_list[0]] + i for i in indices]



def search_combination(predict_list, y_val, score_name):
    idx_list = list(range(len(predict_list)))
    combi_idx = all_subs(idx_list)
    combi_idx.pop(0)
    
    best_combi = None
    score = 0.0
    
    for combi in combi_idx:
        pred_list = [predict_list[i] for i in combi]
        y_pred = np.mean(pred_list, axis=0)
        y_pred = np.argmax(y_pred, axis=1)
        pred_score = matric_score(y_val, y_pred, score_name)
        if pred_score >= score:
            score = pred_score
            best_combi = combi
    
    return best_combi

