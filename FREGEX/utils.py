import copy
import re
import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import gc
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, PredefinedSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GBC
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from gensim.models import TfidfModel, FastText
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from nltk.util import ngrams
import time
import platform
import random
#pip install editdistance==0.6.1
import editdistance
#http://lingpy.org/docu/align/multiple.html
#pip install lingpy==2.6.9
from lingpy.align.multiple import mult_align
from nltk.stem import SnowballStemmer
import shutil
from scipy.stats import entropy
import ast
#https://skranger.readthedocs.io/en/stable/ranger_forest_classifier.html
import math
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier as DTC
#https://raw.githubusercontent.com/xiamx/node-nltk-stopwords/master/data/stopwords/spanish
#from nltk.corpus import stopwords
#pip install xgboost==1.7.3
from xgboost import XGBClassifier as XGB
import itertools
from tqdm import tqdm
from itertools import combinations
import unicodedata
import ast

SEED = 42
F_OUTLIERS = 1.5
F_EXT_OUTLIERS = 2.5
NGRAM_MIN = 3
WINDOW_MIN = 5
BATCH = 64
THR_PRB = 0.9
THR_CLASS = 2/3
THR_CONF = 0.95

pnumbers = r'\d+(?:[\.\,]\d+)?'
punctuation = r'[^a-zA-Z\d\s\+\-]' #r'[\.\,\:\=\;\'\"\(\)\[\]\{\}]'#r'[\.\,\¡\!\¿\?\:\=\;\'\"\(\)\[\]\{\}]'
gap = r'(?:\w)?'
gaps =  r'(?:\w+)?'
nonalpha =  r'[^a-zA-Z\d\s]'
words = r'[a-zA-Z]{3,}'
whitespaces = r'[\s]*'  #r'[\s%s]*' %r'\.\,' #r'[\s]*' 
gap_cmb = r'[\s\S]*'
ptimes = r'[^\S]*'
digit_mask = 'DIGIT'
gap_mask = 'GAP'
gap_sw = r'XYZ'

lexicon = {
    'FUMADOR': ['fum','tab', 'cig', 'caj'],
    'OBESIDAD': ['obes',
                 'peso', 'normopes', 'sobrepes',
                 'imc'],
    'OBESIDAD_TIPOS': ['obes',
                        'imc'],
    }

HYPERPARAMS = defaultdict(dict)
HYPERPARAMS['bert']  = {
            'scheduler_opt': True,
            'early_stopping': False,
            'validation_split': 0.0,
            'val_loss_min': None,
            'patience': None,
            'batch_size': 8,
            'epochs': 4,
            'dropout': 0.2,
            'MAX_SENT_LEN': 512, #64,
            'lr': 2e-5,
            'RUNS': 10,
            #'bert_type': 'albert'
            'bert_type': 'bert'

}
'''
HYPERPARAMS['svm'] = {}
HYPERPARAMS['rf'] = {}
HYPERPARAMS['nb'] = {}
HYPERPARAMS['gbc'] = {}
HYPERPARAMS['xgb'] = {}
HYPERPARAMS['cregex-svm'] = {}
HYPERPARAMS['cregex-rf'] = {}
HYPERPARAMS['cregex-nb'] = {}
HYPERPARAMS['cregex-gbc'] = {}
HYPERPARAMS['cregex-xgb'] = {}
HYPERPARAMS['cregex-regexes'] = {}
HYPERPARAMS['cregex-bert']  = {
        'scheduler_opt': True,
        'early_stopping': False,
        'validation_split': 0.0,
        'val_loss_min': None,
        'patience': None,
        'batch_size': 8,
        'epochs': 4,
        'dropout': 0.2,
        'MAX_SENT_LEN': 512, #64,
        'lr': 2e-5,
        'RUNS': 10,
        #'bert_type': 'albert'
        'bert_type': 'bert'
}
'''


try:
    #https://codebeautify.org/python-formatter-beautifier
    import transformers
    from transformers import get_linear_schedule_with_warmup
    from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer
    from transformers import BertModel, DistilBertModel, AlbertModel    
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
    from torch.nn.utils import clip_grad_norm_
    from torch.optim import SGD, Adam, lr_scheduler, AdamW
    import torch.nn.functional as F
    #import pytorch_lightning as pl
    from keras.utils import np_utils
    from keras.preprocessing.sequence import pad_sequences
    
    def seed_everything(seed=SEED):
        print('seeds pytorch')
        '''
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)   
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
        #os.environ["CUDA_VISIBLE_DEVICES"]= "0"
        #torch.backends.cudnn.enabled = False
        '''
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
except:
    def seed_everything(SEED=SEED):
        np.random.seed(SEED)
        random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
    
def create_paths(FILENAME, root=os.getcwd()):
    if 'out' not in os.listdir( os.path.join( root ) ):
        os.mkdir( os.path.join( root, 'out' ) )
    if 'RESULTS' not in os.listdir( os.path.join( root, 'out') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTS' ) )
    if 'Tables' not in os.listdir( os.path.join( root, 'out') ):
        os.mkdir( os.path.join( root, 'out', 'Tables' ) )
    if 'Figures' not in os.listdir( os.path.join( root, 'out') ):
        os.mkdir( os.path.join( root, 'out', 'Figures' ) )
    if FILENAME not in os.listdir( os.path.join( root, 'out', 'RESULTS') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTS', FILENAME ) )
    if 'RESULTSLC' not in os.listdir( os.path.join( root, 'out') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC' ) )
    if 'PL' not in os.listdir( os.path.join( root, 'out', 'RESULTSLC') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'PL' ) )
    if FILENAME not in os.listdir( os.path.join( root, 'out', 'RESULTSLC', 'PL') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'PL', FILENAME ) )
    if 'AL' not in os.listdir( os.path.join( root, 'out', 'RESULTSLC') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'AL' ) )
    if FILENAME not in os.listdir( os.path.join( root, 'out', 'RESULTSLC', 'AL') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'AL', FILENAME ) )
    if 'SSLAL' not in os.listdir( os.path.join( root, 'out', 'RESULTSLC') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'SSLAL' ) )
    if FILENAME not in os.listdir( os.path.join( root, 'out', 'RESULTSLC', 'SSLAL') ):
        os.mkdir( os.path.join( root, 'out', 'RESULTSLC', 'SSLAL', FILENAME ) )
    shutil.copy( os.path.join( root, 'sw_cpp.cpp' ), os.path.join( os.getcwd(), 'sw_cpp_%s.cpp' %FILENAME ) )
    shutil.copy( os.path.join( root, 'sw_cpp_score.cpp' ), os.path.join( os.getcwd(), 'sw_cpp_score_%s.cpp' %FILENAME ) )

def get_min_max_thr(Z, min_, max_, step=1e-4, seed=SEED):
    clusters_aux = fcluster(Z, t=min_, criterion='distance')
    while len(np.unique(clusters_aux)) == 1:
        min_ += step
        clusters_aux = fcluster(Z, t=min_, criterion='distance')
    clusters_aux = fcluster(Z, t=max_, criterion='distance')
    while len(np.unique(clusters_aux)) == 1:
        max_ -= step
        clusters_aux = fcluster(Z, t=max_, criterion='distance')
    return min_, max_
        
def get_thr_clustering(X, Z, metric='cosine', iterations=10,seed=SEED):
    d = dendrogram(Z)
    h = [y[1] for y in d['dcoord']]
    min_thr, max_thr = get_min_max_thr(Z, min(h), max(h))
    silhouettes_x = np.linspace(min_thr, max_thr, iterations)
    silhouettes_y = []
    silhouette_max = 0
    for h in silhouettes_x:
        clusters_aux = fcluster(Z, t=h, criterion='distance')
        score  = silhouette_score(X, clusters_aux, metric=metric, random_state=seed)
        silhouettes_y.append( score )
        if score>silhouette_max:
            silhouette_max = score
            t = h
    return t

def split_tokens(tokens, N=NGRAM_MIN):
    tokens = list(sorted(tokens))   
    visited = []
    for i in range(len(tokens)):
        visited_aux = []
        if i not in visited:
            visited_aux = [ tokens[i]  ]
            ngramA = tokens[i][:N]
            visited.append(i)
            for j in range(len(tokens)):
                ngramB = tokens[j][:N]
                if j not in visited:
                    if not re.findall(r'%s' %pnumbers, tokens[j]):
                        if ngramA[0] == ngramB[0]:
                            visited_aux.append( tokens[j] )
                            visited.append(j)
        if visited_aux:
            yield visited_aux

def filtering_clusters(tokens_clusters, tokens_freq):
    tokens_aux = list( split_tokens(tokens_clusters) )
    bases, filters = [], []
    for tokens in tokens_aux:
        tokens = list(sorted(tokens))
        max_ = -1
        base = ''
        for token in tokens:
            if tokens_freq[token]>max_:
                max_ = tokens_freq[token]
                base = token
        bases.append(base)
        filters.append(tokens)
    del tokens_aux
    gc.collect()
    return bases, filters
    
def finditer(regex, text):
    return re.finditer( r'\s%s\s' %regex,  ' '+text+' ' )

def match(regex, text, pos=False):
    if not pos:
        f = [ m.strip() if type(m)==str else m for m in re.findall(  r'\s%s\s' %regex,  ' '+text+' ' ) ] 
    else:
        f = set() 
        for m in re.findall(  r'\s%s\s' %regex,  ' '+text+' ' ):
            if type(m)==str:
                f.add( text.index(m.strip())) 
            else:
                for elem in m:
                    f.add( text.index(elem.strip()) )             
    return f
    
def findall(regex, pos_aux, numbers_aux, text, return_numbers=False, pnumbers=pnumbers):
    if len(numbers_aux)==0:
        return match(regex, text)
    else:
        regex_numbers = r'%s' %regex.replace(pnumbers,'('+pnumbers+')')
        find = match(regex_numbers, text)
        findings = []
        if find:
            flag = True
            for f in find:
                if type(f)==str:
                    f = [f]
                f = list(filter(None, f))
                findings.append(f)
                if flag:
                    count = 0
                    for i in range(len(f)):
                        number = float(f[i].replace(',', '.'))
                        min_aux = min(numbers_aux[:,i])
                        max_aux = max(numbers_aux[:,i])
                        if number>=min_aux and number<=max_aux:
                            count+=1                        
                    if count==numbers_aux.shape[1]:
                        flag = False
                        break
            if count==numbers_aux.shape[1]:
                if return_numbers:
                    findings = np.array(findings)
                    return [findings, np.round(np.mean(numbers_aux, axis=0),0,).astype(int) ]
                else:
                    return match(regex, text)
            else:
                return []
        else:
            return []

def get_matrix(tokens, X, regexes, opt=False, idf=True):
    n_x, n_t = len(X), len(tokens)
    matrix = np.zeros((n_x,n_t))
    idf_vector = np.zeros(n_t)
    for t in range(n_t):
        d = 0
        for x in range(n_x):
            if opt:
                pos_aux, numbers_aux, _, __, ___ = regexes[tokens[t]]
                f = len( findall(tokens[t], pos_aux, numbers_aux, X[x]) )
            else:
                f = len( match( re.escape(tokens[t]), X[x]) )
            matrix[x,t] = f
            if f>0:
                d += 1
        if d==0:
            idf_vector[t] = 0
        else:
            idf_vector[t] = np.log10(n_x/d)
    if idf:
        return matrix*idf_vector
    else:
        return matrix

def n_grams(texts, N):
    tokens_aux = []
    for text in texts:
        tokens = re.split(r'\s+', text)
        for token in list(ngrams(tokens, N)):
            tokens_aux.append(" ".join(token))
    tokens_aux = np.array( sorted( list(set(tokens_aux)) ) )
    return tokens_aux

def save_txt(data, path, filename):
    remove( path, filename )
    with open(os.path.join(path, filename), 'w', encoding='utf-8', newline='\n') as a:
        for c in range(len(data)):
            if type(data[c])==list:
                a.write(' '.join( data[c]) )
            elif type(data[c]) in [int, float]:
                a.write( str( data[c] ) )
            else:
                a.write( data[c] )
            if c<len(data)-1:
                a.write('\n')

def remove(path, filename):
    #if filename in os.listdir( path ):
    while filename in os.listdir( path ):
        print(filename, 'was removed')
        os.remove( os.path.join( path, filename ) )

def hashfxn(astring):
    return ord(astring[0])
    
def fasttext( VECTOR_SIZE, NGRAM_SIZE, min_count, sg, corpus, CORPUS_SIZE, epochs,seed=SEED  ):
    model = FastText(vector_size=VECTOR_SIZE, window=NGRAM_SIZE, min_count=min_count, sg=sg, 
                     seed=seed, workers=1, max_vocab_size=None, hashfxn=hashfxn, sorted_vocab=1)  
    model.build_vocab(corpus_iterable=corpus)
    model.train(corpus_iterable=corpus, total_examples=CORPUS_SIZE, epochs=epochs)     
    return model

def boundaries(y_1, y_2):
    std_1 = np.std(y_1)
    std_2 = np.std(y_2)
    if std_1 != 0 or std_2 != 0: 
        p = std_1/(std_1+std_2)
        distance = abs( np.min(y_2)-np.max(y_1) )
    else:
        p = 1.0
        distance = abs( np.max(y_1)-np.min(y_2) )/2
    return distance*p

def get_max_min_numbers(col_numbers_aux, classes, f=F_OUTLIERS):
    number_matches = copy.deepcopy(col_numbers_aux)
    number_matches = [ list(map(float, numbers)) for numbers in number_matches ]
    y_pos, y_sorted = zip( *sorted( enumerate( number_matches ), key= lambda x:np.median(x[1]), reverse=False ) )
    min_max = []
    q1 = np.quantile(y_sorted[0], 0.25)
    q3 = np.quantile(y_sorted[0], 0.75)
    iqr = q3-q1
    min_ = q1-(f*iqr)
    min_ = min([min_, np.min(y_sorted[0])])
    if len(y_sorted)>1:
        for i in range(len(y_sorted)-1):
            d = boundaries(y_sorted[i], y_sorted[i+1])
            max_ = np.max(y_sorted[i])+d
            if not math.isnan(min_) and not math.isnan(max_):
                min_max.append( (min_, max_) )
            else:
                print('Warning: NaN')
                min_ = np.min(y_sorted[i])
                max_ = np.max(y_sorted[i])
                min_max.append( (min_, max_) )
            min_ = max_
    q1 = np.quantile(y_sorted[-1], 0.25)
    q3 = np.quantile(y_sorted[-1], 0.75)
    iqr = q3-q1
    max_ = q3+(f*iqr)
    max_ = max([max_, np.max(y_sorted[-1])])
    min_max.append( (min_, max_) )
    min_max = np.array(min_max)
    y_pos = np.array(y_pos)
    classes_aux = np.array(classes)
    idxs = np.arange(len(y_pos), dtype=int)
    return dict( zip(classes_aux[y_pos], idxs) ), min_max

'''
def replace_outliers(numbers, N=NGRAM_MIN, f=F_EXT_OUTLIERS):
    numbers_aux = np.array( copy.deepcopy(numbers), dtype=float )#.flatten()
    if len(numbers_aux)>=N:
        q1 = np.quantile(numbers_aux, 0.25)
        q3 = np.quantile(numbers_aux, 0.75)
        iqr = q3-q1
        median = np.median(numbers_aux)
        lower_bound = q1-(f*iqr)
        upper_bound = q3+(f*iqr)
        idxs = np.array( list( set( list(np.where(numbers_aux<lower_bound)[0])+list(np.where(numbers_aux>upper_bound)[0]) ) ) )
        if len(idxs)>0:
            numbers_aux[idxs] = median
    numbers_aux = list(numbers_aux)
    return numbers_aux
'''

def replace_outliers (numbers, WINDOW_MIN=WINDOW_MIN, THR=F_OUTLIERS):
    numbers_aux = np.array( sorted(numbers, reverse=False) ).astype(float)
    median = int( np.median(numbers_aux) )
    i = 0
    while i<len(numbers_aux):
        mean = np.mean(numbers_aux[:i+WINDOW_MIN])
        std = np.std(numbers_aux[:i+WINDOW_MIN])
        z_score = (numbers_aux[i]-mean)/std
        i+=1
        if np.abs(z_score)>THR:
            break
    if i==len(numbers_aux):
        i = 0
    j = len(numbers_aux)
    while j>0:
        mean = np.mean(numbers_aux[-WINDOW_MIN+j:j])
        std = np.std(numbers_aux[-WINDOW_MIN+j:j])
        z_score = (numbers_aux[j-1]-mean)/std
        j-=1
        if np.abs(z_score)>THR:
            break
    if i!=0:
        numbers_aux[:i+1] = median
    if j==0:
        j = len(numbers_aux)      
    else:
        numbers_aux[j:] = median
    return numbers_aux
    
def complete_value(y_values, SIZE):
    for i in range(len(y_values)):
        while len(y_values[i])<SIZE:
            y_values[i].append(y_values[i][-1])
    return y_values
     
def select_trad_model(MODEL, HYPERPARAMS):
    seed_everything()
    if 'svm' in MODEL:
        model = SVC(**HYPERPARAMS)            
    elif 'nb' in MODEL:
        model = MNB(**HYPERPARAMS)
    elif 'rf' in MODEL:
        model = RFC(**HYPERPARAMS)
    elif 'gbc' in MODEL:
        model = GBC(**HYPERPARAMS)
    elif 'xgb' in MODEL:
        model = XGB(**HYPERPARAMS)
    return model

def best_model(MODEL, ps, X_train_val, y_train_val, scoring='accuracy', SEED=SEED):
    seed_everything()
    if 'svm' in MODEL:
        best_params = {'random_state':SEED, 'probability':True}
        param_grid = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100, 1000]}
        model = SVC( random_state=SEED )
    elif 'nb' in MODEL:
        best_params = {}
        param_grid = {'alpha': [1e-10, 0.25, 0.75, 1]}
        model = MNB()
    elif 'rf' in MODEL:
        best_params = {'random_state':SEED}
        param_grid = {'criterion':('entropy', 'gini'), 'n_estimators':[10, 100, 500, 1000]}
        model = RFC(random_state=SEED)
    elif 'gbc' in MODEL:
        best_params = {'random_state':SEED}
        param_grid = {'n_estimators':[5,50,250,500],'max_depth':[1,3,5,7,9],'learning_rate':[0.01,0.1,1,10,100]}
        model = GBC( random_state=SEED ) 
    elif 'xgb' in MODEL:
        best_params = {'random_state':SEED}
        param_grid = {'gamma':[0, 0.5, 1, 10],'learning_rate':[0.1, 0.3, 0.8, 1.0], 'n_estimators':[10, 20, 50, 200, 400] }
        model = XGB()
    clf = GridSearchCV( model, param_grid=param_grid, cv=ps, scoring=scoring)
    clf.fit( X_train_val, y_train_val )
    best_params.update(clf.best_params_) 
    del clf
    del model
    gc.collect()
    return best_params

def reduce_regex(regexes,whitespaces=whitespaces, gap_cmb=gap_cmb): 
    keys_regexes = sorted( regexes, 
                          key=lambda x:len( re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), x[0]) ), 
                          reverse=True)    
    filtered_regexes = []
    exclude = set()
    for regexA, posA in keys_regexes:
        tokensA = set( re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), regexA) )
        for regexB, posB in keys_regexes:
            if regexA != regexB:
                tokensB = set( re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), regexB) )
                if len( tokensB.difference(tokensA) )==0:
                    exclude.add(regexB)
        if regexA not in exclude:
            filtered_regexes.append( (regexA, posA) )
    return filtered_regexes

def get_modified_regexes(pos2regex, SIZE_X, regexes, comb=True, reduce=False, MAX_COMB = 4, gap_cmb=gap_cmb):
    new_regexes = {}
    #for i in tqdm( range(SIZE_X) ):
    for i in range(SIZE_X) :
        
        #print(i/SIZE_X)
        
        regex_2_pos = copy.deepcopy(pos2regex[i])
        regex_2_pos = reduce_regex(regex_2_pos)
        for regex, _ in regex_2_pos:
            new_regexes[regex] = regexes[regex]

        '''
        regex_2_pos = copy.deepcopy(regex2pos[i])
        if reduce:
            regex_2_pos = reduce_regex(regex_2_pos)
        regex_2_pos = sorted(regex_2_pos, key = lambda x:min(x[1]), reverse=False)
            
        for regex, _ in regex_2_pos:
            new_regexes[regex] = regexes[regex]
    
        if comb:        
            regexes_aux, _ = list( zip( *regex_2_pos) ) 
            #n-th reegex combination
            if len(regexes_aux)<=MAX_COMB:
                ranges = range(1,len(regexes_aux))
            else:
                ranges = np.linspace(1, len(regexes_aux), MAX_COMB, dtype=int)
            for n in ranges:
                intersect = []
                pos = []
                numbers = []
                R = defaultdict(int)  
                #each-one j-th regex
                for j in range(len(regexes_aux[:n+1])):
                    pos_aux, numbers_aux, pattern2token, pattern2tokens, model = regexes[regexes_aux[:n+1][j]]
                    pos.append(pos_aux)
                    numbers.append(numbers_aux)
                    intersect.append(set(pos_aux))
                    for p in set(list(pos_aux)):
                        count = list(pos_aux).count(p)
                        if count>R[p]:
                            R[p] = count                                
                del pos_aux
                del numbers_aux
                gc.collect()
                intersect = np.array( sorted(list( set.intersection(*intersect) ) ) )
                numbers_aux = []
                pos_aux = []
                if len(intersect)>0:
                    flag = True
                    for p in intersect:
                        pos_aux.append( p )
                        while pos_aux.count(p) < R[p]:
                            pos_aux.append( p )
                        k = -1
                        for j in range(len(numbers)):
                            if len(numbers[j])>0:
                                k+=1
                                if flag:
                                    numbers_aux.append( np.array([]) )
                                idx_intersect = np.where( pos[j] == p)[0] 
                                if numbers[j][idx_intersect,:].shape[0]>0:
                                    aux = numbers[j][idx_intersect,:]
                                    while aux.shape[0] < R[p]:
                                        aux = np.vstack( (aux, aux[-1,:]) )
                                    if len(aux.shape)<2:
                                        aux = aux.reshape(-1,1)
                                    if len(numbers_aux[k])==0:
                                        numbers_aux[k] = aux
                                    else:
                                        values = copy.deepcopy(numbers_aux[k])
                                        values = np.vstack((values, aux))
                                        numbers_aux[k] = values           
                        flag = False
                    if sum([v.shape[0] for v in numbers_aux])==0:
                        numbers_aux = np.array([])
                    else:
                        numbers_aux = np.concatenate(numbers_aux, axis=1)
                    pos_aux = np.array(pos_aux)
                    regex = r' '.join(regexes_aux[:n+1])
                    regex = re.sub(r'\s+', r'%s'  %gap_cmb.replace('\\', '\\\\'), regex)
                    new_regexes[regex] = [pos_aux, numbers_aux, pattern2token, pattern2tokens, model]
        '''            
    return new_regexes

#def get_class_conf(regexes, y, kw, whitespaces=whitespaces, gap_cmb=gap_cmb, THR_CLASS=THR_CLASS, THR_CONF=THR_CONF):
def get_filtered_regexes(regexes, y, kw, pattern2token, regex2class, THR_CONF=THR_CONF, whitespaces=whitespaces, gap_cmb=gap_cmb): #, THR_CLASS=THR_CLASS):
    #print(THR_CONF, THR_CONF, THR_CONF, type(THR_CONF))
    a = open('classes_regexes.txt', 'w')
    b = open('classes_regexes_filteed_out.txt', 'w')
    c = open('classes_regexes_filteed_out_out.txt', 'w')
    keys_regexes = list( regexes.keys() )
    labeled_regexes = {}
    labeled_regexes_filtered = {}
    labeled_regexes_all = {}
    i = 0
    while i<len(keys_regexes):
        label = -1
        conf = -1
        key_i = keys_regexes[i]
        label, conf = regex2class[key_i]
        flag = False
        key_i_aux = re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), key_i)
        for token in key_i_aux:
            if token in pattern2token:
                tokenA = pattern2token[token]
            else:
                tokenA = copy.deepcopy(token)
            for tokenB in kw:
                if tokenB in tokenA:
                    flag = True
                    break
            if flag:
                break
        if flag: #kw
            if label != -1 and conf>THR_CONF: 
                labeled_regexes[key_i] = [label, conf]
                labeled_regexes_filtered[key_i] = [label, conf]
                labeled_regexes_all[key_i] = [label, conf]
                a.write('*'+key_i+'->'+str(label)+','+str(conf)+'\n')      
            else:
                b.write('*'+key_i+'->'+str(label)+','+str(conf)+'\n') 
                labeled_regexes_filtered[key_i] = [label, conf]
                labeled_regexes_all[key_i] = [label, conf]
        else: #no kw
            c.write('*'+key_i+'->'+str(label)+','+str(conf)+'\n')
            labeled_regexes_all[key_i] = [label, conf]
        i+=1
    a.close()
    b.close()
    c.close()
    return labeled_regexes, labeled_regexes_filtered, labeled_regexes_all

def sw_pre_processing(x, 
                      regexes,
                      token2pattern,
                      stopwords,
                      replace_numbers = False,
                      stop_words = False,
                      mask_numbers = True,
                      pnumbers=pnumbers, 
                      digit_mask=digit_mask,
                      nonalpha=nonalpha,
                      punctuation=punctuation,
                      whitespaces=whitespaces, gap_cmb=gap_cmb
                      ):
    
    text_aux = ' '+ x +' '
    
    if replace_numbers:
        keys = sorted( list(regexes.keys()), 
                    key = lambda x: len( re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), x) ),
                    reverse = True )
        visited = []
        for regex in keys:          
            _, numbers_aux, _, _, _ = regexes[regex]
            f = findall(regex, [], numbers_aux, text_aux, True)
            if len(f)>0 and len(numbers_aux)>0:
                f_matches, f_mean = copy.deepcopy( f )
                for i in range(f_matches.shape[0]):
                    for j in range(f_matches.shape[1]):
                        if f_matches[i][j] not in visited:
                            text_aux = re.sub(' '+f_matches[i][j]+' ',' '+ str(f_mean[j])+' ', ' '+text_aux+' ').strip()
                            visited.append( f_matches[i][j] )

    if mask_numbers:
        text_aux = re.sub(pnumbers, digit_mask, text_aux)         
    
    text_aux = re.sub(r'(%s\s*)\1+' %nonalpha, r'\1', text_aux)
    text_aux = re.sub(r'(%s)\s*' %nonalpha, r'(?:\\\1\\s*)+ ', text_aux)
    text_aux = re.sub(r'(\(\?\:\\%s\\s\*\))\+' %punctuation, r'%s' %punctuation.replace('\\', '\\\\'), text_aux)  
    text_aux = re.sub(r'(%s\s*)\1+' %re.escape(punctuation), r'\1', text_aux)  
    text_aux = re.sub(r'(%s)' %re.escape(punctuation), r'(?:\1\\s*)*', text_aux)  
    
    if mask_numbers:
        text_aux = re.sub(digit_mask, pnumbers.replace('\\', '\\\\'), text_aux) 
        
    text_aux = text_aux.strip()
    corpus_aux = text_aux.split(' ')
    for t in range(len(corpus_aux)):
        if stop_words:
            if corpus_aux[t] in stopwords:
                corpus_aux[t] = r'(?:%s)?' %corpus_aux[t]
        if corpus_aux[t] in token2pattern:
            corpus_aux[t] = token2pattern[corpus_aux[t]]     
    return ' '.join( corpus_aux )
    
def get_class_similarity( X, y, X_test, regexes, token2pattern, stopwords, FILENAME, proba, sw_pr):
    X_train = copy.deepcopy(X)
    X_test_aux = copy.deepcopy( X_test )

    if sw_pr:
        for i in range(len(X_train)):
            X_train[i] = sw_pre_processing( X_train[i], regexes, token2pattern, stopwords, True, False, False )

        for i in range(len(X_test_aux)):
            X_test_aux[i] = sw_pre_processing( X_test_aux[i], regexes, token2pattern, stopwords, True, False, False )

    y_train = copy.deepcopy( y )
    
    print('PATH-cregex...', os.getcwd())
    
    remove(os.path.join(os.getcwd(), 'out'), 'CLASESU_'+FILENAME+'.txt')
    remove(os.path.join(os.getcwd(), 'out'), 'SCORESU_'+FILENAME+'.txt')
    remove(os.getcwd(), 'sw_cpp_score_%s' %FILENAME)
    remove(os.getcwd(), 'sw_cpp_score_%s.exe' %FILENAME)
    
    save_txt(X_train, os.path.join(os.getcwd(), 'out'), 'DATOSX_'+FILENAME+'.txt')
    save_txt(y_train.astype(str), os.path.join(os.getcwd(), 'out'), 'CLASESX_'+FILENAME+'.txt')
    save_txt(X_test_aux, os.path.join(os.getcwd(), 'out'), 'DATOSU_'+FILENAME+'.txt')
        
    if platform.system() == 'Windows':
        os.system("g++ sw_cpp_score_%s.cpp -o sw_cpp_score_%s.exe" %(FILENAME, FILENAME))
        os.system("sw_cpp_score_%s.exe %s" %(FILENAME, FILENAME))
        remove(os.getcwd(), 'sw_cpp_score_%s.exe' %FILENAME)
    elif platform.system() == 'Linux':
        os.system("g++ sw_cpp_score_%s.cpp -o sw_cpp_score_%s" %(FILENAME,FILENAME))
        os.system("./sw_cpp_score_%s %s" %(FILENAME, FILENAME))
        remove(os.getcwd(), 'sw_cpp_score_%s' %FILENAME)

    with open( os.path.join(os.getcwd(), 'out', 'SCORESU_'+FILENAME+'.txt'), 'r', encoding='utf-8', newline='\n') as a:
        scores = a.read().split("\n")[:-1]
        scores = np.array( scores, dtype=float )/1000 #3 decimals

    with open( os.path.join(os.getcwd(), 'out', 'CLASESU_'+FILENAME+'.txt'), 'r', encoding='utf-8', newline='\n') as a:
        classes = a.read().split("\n")[:-1]
        classes = np.array( classes, dtype=int )

    if proba:
        classes_aux = []
        NCLASES = len(set(y_train))
        for i in range(len(scores)):
            max_conf = copy.deepcopy(scores[i]) 
            pos_aux = copy.deepcopy(classes[i])
            pond = (1-max_conf)/(NCLASES-1)
            classe =  np.ones(NCLASES)*pond
            classe[ pos_aux ] = max_conf
            classes_aux.append(classe)
        del classes
        gc.collect()
        classes = np.array(classes_aux,  dtype=float)

    return classes

def weight_regexes(X, kw,  pattern2token, labeled_regexes, gap_cmb=gap_cmb, whitespaces=whitespaces ):
    m_weight = np.zeros((len(X), len(kw)))
    for x in range(len(X)):
        tokens = re.split(r'\s+', X[x])
        for k in range(len(kw)):
            for t in range(len(tokens)):
                if tokens[t].startswith(kw[k]):
                    m_weight[x][k] = 1
                    break
    kw_weight = {}
    labeled_regexes_aux = {}
    for key in labeled_regexes:
        label, conf = labeled_regexes[key]
        f = 1
        c = 0
        tokens = re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), key)
        for k in range(len(kw)):
            w = m_weight[:,k].sum()/len(X)
            for token in tokens:
                if token in pattern2token:
                    token = pattern2token[token]
                if token.startswith( kw[k] ):
                    f*=w
                    c+=1
                    break
        if c >1:
            labeled_regexes_aux[key] = [label, conf]
        else:
            labeled_regexes_aux[key] = [label, conf*f]
    return labeled_regexes_aux


def get_classes_regexes(regexes, y, tokens2pos, gap_cmb=gap_cmb, whitespaces=whitespaces, THR_CLASS=THR_CLASS):
    keys = sorted( list(regexes.keys()), 
                    key = lambda x: len( re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), x) ),
                    reverse = False
     )
    regex2class = {}
    regexes_aux = {}
    
    for indexA in range(len(keys)):
    #for indexA in range(len(keys)-1):
                
        posA, numbersA, pattern2token, pattern2tokens, model = regexes[keys[indexA]]
        
        pos = tokens2pos[keys[indexA]] #label
        pos = np.array(pos)

        '''
        tokensA = re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), keys[indexA])
        #tokensA = set(tokensA)
        
        labels = y[posA]
        '''
        labels_texts = y[pos]
        labels_training = y[posA]


        '''
        labels_aux = copy.deepcopy(labels)
        posA_aux = copy.deepcopy(posA)
        numbersA_aux = copy.deepcopy(numbersA)
        '''
        
        #posA = set(posA)

        #if keys[indexA] == r'(?:\w)?fumad(?:\w)?or(?:\w)?':
        #    print('--A')
        #    print(labels)

        '''

        for indexB in range(indexA+1, len(keys)):
            posB, numbersB, _, __, ___ = regexes[keys[indexA]]
            tokensB = re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), keys[indexB])
            #tokensB = set(tokensB)
            #posB = set(posB)
            if len( set(tokensA).difference(set(tokensB)) )==0:
                posAB = np.array( list(set(posA).intersection(set(posB))) )
                if len(posAB)>1:
                    classesAB = y[posAB]
                    #if len(set(classesAB))>1:
                    #posA = np.array(list(posA))
                    idxA = np.where(posA==posAB)[0]
                    #print(posA)
                    #print(posA.shape)
                    #print(numbersA)
                    #print(numbersA.shape)
                    #print(idxA)
                    posA = np.delete(posA, idxA)
                    if len(numbersA)>0:
                        numbersA = np.delete(numbersA, idxA, axis=0)
                    labels = y[posA]

                    #if keys[indexA] == r'(?:\w)?fumad(?:\w)?or(?:\w)?':
                    #    print('B')
                    #    print(keys[indexB])
                    #    print(labels)
                    
                    #xyz
                    #break

        '''

        '''
        if len(labels)>0:
            #label_aux, f_aux = Counter(labels).most_common()[0]
            if f_aux/len(labels) >THR_CLASS: #>= THR_CLASS:
                #label = label_aux
                ypred =  np.ones(len(labels), dtype=int)
                ytrue = np.where(labels==label_aux,1,0)
                #print(ypred, ytrue,label)
                conf = precision_score(ytrue, ypred)
                regexes_aux[keys[indexA]] = [posA, numbersA, pattern2token, pattern2tokens, model]
                regex2class[keys[indexA]] = [label_aux, conf]
                #print(keys[indexA], label_aux, labels)
        else:
            labels = copy.deepcopy(labels_aux)
            #posA, numbersA, _, __, ___ = regexes[keys[indexA]]
            #labels = y[posA]
            #print(keys[indexA], keys[indexB])
            #print(posA, posB)
            label_aux, f_aux = Counter(labels).most_common()[0]
            ypred =  np.ones(len(labels), dtype=int)
            ytrue = np.where(labels==label_aux,1,0)
            conf = precision_score(ytrue, ypred)
            regexes_aux[keys[indexA]] = [posA, numbersA, pattern2token, pattern2tokens, model]
            regex2class[keys[indexA]] = [label_aux, conf]
        '''

        #if keys[indexA] == r'(?:\w)?fumad(?:\w)?or(?:\w)?':
        #    print('***A')
        #    print(labels)

        '''
        if len(labels)==0:
            labels = copy.deepcopy(labels_aux)
            posA = copy.deepcopy(posA_aux)
            numbersA = copy.deepcopy(numbersA_aux)
        '''

        label_aux, f_aux = Counter(labels_texts).most_common()[0]
        if f_aux/len(labels_texts) >THR_CLASS: #>= THR_CLASS:
            #label = label_aux
            ypred =  np.ones(len(labels_training), dtype=int)
            ytrue = np.where(labels_training==label_aux,1,0)
            #print(ypred, ytrue)#,label)

            conf = precision_score(ytrue, ypred)
            '''
            if (ypred-ytrue).sum()>0:
                    tn, fp, fn, tp = confusion_matrix(ytrue, ypred).flatten()
            else:
                tp = len(ypred)
            conf = tp
            '''
            
            regexes_aux[keys[indexA]] = [posA, numbersA, pattern2token, pattern2tokens, model]
            regex2class[keys[indexA]] = [label_aux, conf]
            #print(keys[indexA], label_aux, labels)


            #print( keys[indexA], y[posA] )
    return regexes_aux, regex2class

'''
def augmentation(tokens, whitespaces=r'\s+', SEED=SEED):
    tokens_aux = copy.deepcopy(tokens)
    for token in tokens:
        tokens_i = re.split(r'%s' %whitespaces, token)
        tokens_i = shuffle(tokens_i, random_state=SEED)
        new_token = ' '.join(tokens_i)
        if new_token not in tokens_aux:
            tokens_aux.append(new_token)
    return tokens_aux
'''

def augmentation(regexes, regex2class, kw, pattern2token, TOKENS_SIZE=2, whitespaces=whitespaces, gap_cmb=gap_cmb, SEED=SEED):
    #no comb yet: gap_cmb=gap_cmb
    regexes_aux = copy.deepcopy(regexes)
    regex2class_aux = copy.deepcopy(regex2class)

    
    for regex in regexes:
        p_gap = r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces))
        gap = re.findall(r'%s' %p_gap, regex)
        if len(gap)>0:
            gap = gap[0]
        else:
            gap = ''
        tokens_i = re.split( r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), regex)
        tokens_i = shuffle(tokens_i, random_state=SEED)
        new_regex = r'{0}'.join(tokens_i).format(gap)
        if new_regex not in regexes_aux:
            regexes_aux[new_regex] = regexes[regex]
            regex2class_aux[new_regex] = regex2class[regex]


    '''


    classes = set( [l for l,c in list( regex2class.values() ) ] )
    common = defaultdict(lambda: defaultdict(set))
    common_classe = defaultdict(set)

    for classe in classes:
        #common[classe] = {}
        tokens_aux = []
        for regex in regexes:
            if regex2class[regex][0] == classe:
                pos, numbers, _, __, ___ =  regexes[regex]
                if len(numbers)==0:
                    tokens_i = re.split( r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), regex)
                    if len(set(tokens_i))==TOKENS_SIZE:                    
                        tokens_aux.append( tokens_i )
        for tokensA,tokensB in combinations(tokens_aux,2):
            tokensA = set(tokensA)
            tokensB = set(tokensB)
            intersection = tokensA.intersection(tokensB) 
            if len(intersection)==(TOKENS_SIZE-1):
                #print(tokensA, tokensB, intersection)
                if list(intersection)[0] in pattern2token:
                    token = pattern2token[list(intersection)[0]]
                    for k in kw:
                        if k in token:
                            diffA = list(tokensA.difference(intersection))[0] 
                            diffB = list(tokensB.difference(intersection))[0]
                            if ')*' not in diffA:  
                                common[classe][diffA].add(diffA)
                                common_classe[classe].add(diffA)
                            if ')*' not in diffB:  
                                common[classe][diffB].add(diffB)
                                common_classe[classe].add(diffB)
                            if ')*' not in diffA and ')*' not in diffB:  
                                common[classe][diffA] = common[classe][diffA].union([diffA, diffB])
                                common[classe][diffB] = common[classe][diffB].union([diffA, diffB])
                                common_classe[classe] = common_classe[classe].union([diffA, diffB])
                            break

    exclude = set.intersection(*[set(x) for x in common_classe.values()])  

                    
    for classe in classes:

        #print('classe', common[classe])

        for regex in regexes:
            if regex2class[regex][0] == classe:

                p_gap = r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces))
                gap = re.findall(r'%s' %p_gap, regex)
                if len(gap)>0:
                    gap = gap[0]
                else:
                    gap = ''
                tokens_i = re.split( r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), regex)
                
                for i in range(len(tokens_i)):
                    if tokens_i[i] in common[classe]:

                        intersection = exclude.intersection( common[classe][tokens_i[i]] )
                        if len(intersection)>0:
                            common[classe][tokens_i[i]] = common[classe][tokens_i[i]].difference(intersection)

                        tokens_i[i] =  r'(?:'+r'{0}'.join(common[classe][tokens_i[i]]).format('|')+r')'             

                old_regex = r'{0}'.join(tokens_i).format(gap)

                aux = regexes[regex]
                regexes_aux.pop(regex)
                regexes_aux[old_regex] = aux                

                aux = regex2class_aux[regex]
                regex2class_aux.pop(regex)
                regex2class_aux[old_regex] = aux
                    
                tokens_i = shuffle(tokens_i, random_state=SEED)
                #print(tokens_i, gap)
                #new_regex = r'%s'.join(tokens_i) %gap
                new_regex = r'{0}'.join(tokens_i).format(gap)
                if new_regex not in regexes_aux:
                    regexes_aux[new_regex] = regexes[regex]
                    regex2class_aux[new_regex] = regex2class[regex]
    '''

    return regexes_aux, regex2class_aux

def get_sequences(vector, tokens_c):
    tokens = []
    token = ''
    for i in range(len(tokens_c)):
        if vector[i]:
            token+=' '+tokens_c[i]
        elif not vector[i] and token:
            token = token.strip()
            tokens.append(token)
            token=''
    if token:
        token = token.strip()
        tokens.append(token)
    return tokens

def reduce_sequences(sequences, gap=r' '):
    #no comb yet: gap_cmb=gap_cmb
    sequences = [re.split(r'%s' %gap, seq) for seq in sequences]
    #print(sequences)
    sequences = sorted(sequences, key=lambda x:len(x), reverse=True)
    descartar = []
    filtrados = []
    for seqA in sequences:
        for seqB in sequences:
            if gap.join(seqA) != gap.join(seqB) and len(set(seqB).difference(set(seqA)))==0:
                descartar.append(gap.join(seqB))
        if gap.join(seqA) not in descartar:
            filtrados.append(gap.join(seqA))
    return filtrados

'''
def combine_regexes(regexes, whitespaces=whitespaces, gap_cmb=gap_cmb):
    keys = regexes.keys()
    keys = sorted(keys, key= lambda x:len(re.split( r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), x)), reverse=False)
    min_ = len(re.split( r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), keys[0]))
    max_ = len(re.split( r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), keys[-1]))
    pass
'''     

'''
def update_pos(pos2tokens, SIZE_X):
    for i in range(SIZE_X):
        tokens = pos2tokens[i]
        tokens = sorted(tokens, key = lambda x: len(x.split(" ")), reverse = True)
        for indexA in range(len(tokens)-1):
            for indexB in range(indexA+1, len(tokens)):
'''



'''
def get_scores_kw(tokens, X):
    n_x, n_t = len(X), len(tokens)
    tf_vector  = np.zeros(n_t)
    idf_vector = np.zeros(n_t)
    for t in range(n_t):
        d = 0
        for x in range(n_x):
            f = len( match( re.escape(tokens[t]), X[x]) )
            tf_vector[t] += f
            if f>0:
                d += 1
        if d==0:
            idf_vector[t] = 0
        else:
            idf_vector[t] = np.log10(n_x/d)
    return idf_vector
'''

'''
def get_kw(X, y, token2pattern, N=1, pnumbers=pnumbers, NGRAM_MIN=NGRAM_MIN, SEED=SEED):
    with open( os.path.join( os.getcwd(), 'spanish.txt' ), 'r' ) as a:
        stop_esp = a.read().split('\n')[:-1]
        stop_esp = sorted(stop_esp)
    
    tokens = n_grams(X, N)
    
    ###
    tokens = [ token2pattern[token] if token in token2pattern else token for token in tokens]
    
    matrix = get_matrix(tokens, X, {}, False)
    dtc = DTC(criterion='entropy', random_state=SEED)
    dtc.fit(matrix, y)
    scores = copy.deepcopy(dtc.feature_importances_)
    #idxs = np.where(scores>0.0)[0] 
    idxs = np.argsort(scores)[::-1]
    
    ###
    pattern2token = dict( [ (token2pattern[k], k) for k in token2pattern ] )
    tokens = np.array( [ pattern2token[token] if token in pattern2token else token for token in tokens] )

    
    tokens = [token for token in tokens[idxs] \
                                  if token not in stop_esp and\
                                  len(token)>=NGRAM_MIN and \
                                  not re.findall(r'^%s$' %pnumbers, token)]

        
        
    print('tokens', tokens)
    
    ###
    tokens_aux = n_grams(X, N)
    scores_aux = get_scores_kw(tokens_aux, X)
    idxs_aux = np.argsort(scores_aux)[::-1]
    print(tokens_aux[idxs_aux][:20], scores_aux[idxs_aux][:20])
    ##
    
    tokens = ['fuma', 'fumador', 'fumadora', 
              'tabaquico', 'tabaquica', 'tabaco', 'tabaquismo',
              'cig', 'cigarros', 'cigarrilos', 
              'caj', 'cajetilla']
    
    tokens = ['obesidad', 'obeso', 'obesa',
             'peso', 'normopeso',
             'imc']
    
    kw = [ token2pattern[token] if token in token2pattern else token for token in tokens]
    
    del tokens
    del matrix
    gc.collect()    
    return kw
'''