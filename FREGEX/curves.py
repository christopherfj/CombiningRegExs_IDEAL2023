from utils import *
from fregex import FREGEX
from cregex import CREGEX
from bert import *

def scores_regexes(x,gap_cmb=gap_cmb,whitespaces=whitespaces):
    score = 0
    tokens = re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), x)
    score += sum( [ 5 if re.findall(r'(?:%s|^%s)' %(words, re.escape(gap)), token ) else 1 for token in tokens ] )
    return score

def get_texts(X, regexes, 
              punctuation = punctuation,
              gaps = gaps,
              pnumbers = pnumbers,
              whitespaces = whitespaces,
              gap_cmb=gap_cmb,
              digit_mask = 'DIGIT') :
    texts = []
    key_regexes = sorted(regexes.keys(), 
                         key=lambda x:scores_regexes(x), 
                         reverse=True)
    extract_tokens = False
    for text in X:
        text_aux = copy.deepcopy(text)
        matches_aux = ''
        flag = True
        patterns_aux = []
        for regex in key_regexes:
            patterns = re.split(r'(?:%s|%s)' %(re.escape(gap_cmb), re.escape(whitespaces)), regex)
            patterns = [p for p in patterns if re.findall(r'^%s' %re.escape(gap), p)]
            patterns_aux.extend(patterns)
            pos_aux, numbers_aux, pattern2token, pattern2tokens, model = regexes[regex]
            matches = findall(regex, pos_aux, numbers_aux, text) 
            
            if extract_tokens:
                tokens_patterns = sum( pattern2tokens.values(), [] )
                extract_tokens = False
            
            if len(matches)>0:
                
                #replace numbers
                NNUMBERS = 0
                if len(numbers_aux)>0:
                    NNUMBERS = numbers_aux.shape[1]
                                        
                    #medians = np.median(numbers_aux, axis=0).astype(int).astype(str)
                    replacement = []
                    for k in range(numbers_aux.shape[1]):
                        numbers_aux_replace = replace_outliers( copy.deepcopy( numbers_aux[:,k] ) )
                        if all([n.is_integer() for n in numbers_aux_replace ]):
                            replace = Counter( numbers_aux_replace ).most_common()[0][0] 
                        else:
                            replace = np.median( numbers_aux_replace )
                        replace = replace.astype(int).astype(str)
                        replacement.append( replace )
                    replacement = np.array(replacement)
                        
                    #if '12' in replacement:
                    #    print(regex, numbers_aux, text)
                        
                for m in matches:
                    
                    #if 'clase 3' in text:
                    #    print(text, '->', regex, '->', numbers_aux)
                        
                    if m and m not in matches_aux:
                                                
                        #matches_aux += ' '+m 
                        if NNUMBERS>0:
                            
                            matches_aux += ' '+m 
                            
                            regex_numbers = r'%s' %regex.replace(pnumbers,'('+pnumbers+')')
                            numbers_aux = []
                            f_numbers = match(regex_numbers, m) 
                                                        
                            for number in f_numbers:
                                if type(number) == tuple:
                                    numbers_aux.extend( list(filter(None, list(number))) )
                                elif type(number) == str:
                                    numbers_aux.extend( list(set(filter(None, list([number])))) )      
                                    
                                    
                            #if 'imc' in regex and 'imc' in text and '36' in text:
                            #    print(regex, text, numbers_aux)

                            tokens = list( filter(None, re.split(r'\s+', m) ) )
                            
                            #print('----------------')
                            #print(regex, m, replacement, f_numbers, numbers_aux, tokens)
                            
                            
                            j = 0
                            for i in range(len(tokens)):
                                if re.findall(r'^%s$' %pnumbers, tokens[i]) and tokens[i]==numbers_aux[j]:
                                    tokens[i] = replacement[j] #medians[j]
                                    j+=1
                            #text_aux = ' '.join(tokens)
                            #text_aux = re.sub(r'%s' %re.escape(m), ' '.join(tokens), text_aux )       
                            text_aux = text_aux.replace( m, ' '.join(tokens)  )
                            
                            #print(' '.join(tokens), '->', j)
                            #print(m in text_aux)
                            #print(text, text_aux)
                
               
        '''
        #replace patterns
        patterns_aux = sorted(list(set(patterns_aux)))
        if patterns_aux:
            for pattern in patterns_aux:
                
                tokens_aux = match(pattern, text_aux)
                if tokens_aux:
                    #print('__________')
                    #print(tokens_aux)
                    for token in tokens_aux:
                        if token not in tokens_patterns:
                    
                            #text_aux = re.sub(r'\s%s\s' %pattern, ' '+pattern2token[pattern]+' ', ' '+text_aux+' ')
                            text_aux = re.sub(r'\s%s\s' %token, ' '+pattern2token[pattern]+' ', ' '+text_aux+' ')
                            text_aux = text_aux.strip()
        '''
                        
        #replace punctuation
        #text_aux = re.sub(r'(%s\s*)\1+' %punctuation, r'\1', text_aux)
        #text_aux = text_aux.strip()
            
        del matches_aux
        gc.collect()
        text_aux = re.sub(r'\s+', ' ', text_aux)
        text_aux = text_aux.strip()
        texts.append(text_aux)
    return np.array( texts )

def get_tokens(X,y,N, FILENAME):
    regexes = {}
    opt = False
    tokens = []
    if type(N)==int:
        tokens = n_grams(X, N)                
    elif 'fregex' in N:
        mode = N.split('-')[0]
        fregex = FREGEX(X,y, FILENAME, mode)
        fregex.fit()
        regexes = copy.deepcopy( fregex.transform() )
        opt = True
        tokens = list( regexes.keys() )        
    tokens = sorted(tokens)
    return regexes, opt, tokens

class AL(object):
    def __init__(self, X_u, X_l, y_l, clf, N_CLASSES, CURVE, N,FILENAME, MODEL):
        if 'bert' not in MODEL and 'cregex' not in MODEL:
            #print(X_u)
            #print(X_l)
            regexes, opt, tokens = get_tokens(X_l,y_l, N, FILENAME)
            #print(tokens)
            self.X_u = copy.deepcopy( get_matrix(tokens, X_u, regexes, opt) )
        else:
            self.X_u = copy.deepcopy( X_u )

        self.clf = copy.deepcopy(clf)
        self.N_CLASSES = N_CLASSES
    def score_function(self):
        scores = {}
        probs = self.clf.predict_proba( self.X_u )
        scores_aux = []
        for p in probs:
            scores_aux.append( entropy(p, base=2) )
        scores_aux = np.array(scores_aux)
        scores['scores'] = scores_aux
        scores['probs'] = probs   
        return scores
    
class Curves(object):
    def __init__(self, 
                X_train, y_train,
                X_test, y_test, 
                N_CLASSES, CURVE, MODEL, FILENAME,
                PRED_TYPE='', BATCH=BATCH, THR_PRB=THR_PRB, PVAL = 0.5, SEED = SEED
                ):

        self.X_train = copy.deepcopy(X_train)
        self.y_train = copy.deepcopy(y_train)
        self.X_test  = copy.deepcopy(X_test)
        self.y_test = copy.deepcopy(y_test)
        self.N_CLASSES = N_CLASSES
        self.CURVE = CURVE
        self.NGRAM_SIZE = 'None'
        self.GRID_SEARCH = False
        self.HYPERPARAMS = copy.deepcopy(HYPERPARAMS)
        if 'cregex' in MODEL:
            self.MODEL = MODEL
        else:
            if 'bert' not in MODEL:
                self.MODEL, self.NGRAM_SIZE = MODEL.split('-')
                self.NGRAM_SIZE = int(self.NGRAM_SIZE.replace('n',''))
                self.GRID_SEARCH = True
            else:
                self.MODEL = MODEL
                self.HYPERPARAMS = copy.deepcopy(HYPERPARAMS)
        self.FILENAME = FILENAME
        self.PRED_TYPE = PRED_TYPE
        self.BATCH = BATCH
        self.THR_PRB = THR_PRB

        
        #self.MODEL = MODEL
        #self.HYPERPARAMS = HYPERPARAMS.copy()
        
        #self.NGRAM_SIZE = N
        
        #print('......')
        #print(MODEL)

        #if 'cregex' not in MODEL and len(MODEL.split('-'))>1:
        #    self.MODEL, self.NGRAM_SIZE = MODEL.split('-')
        #    self.NGRAM_SIZE = int(self.NGRAM_SIZE)
        
        #self.GRID_SEARCH = GRID_SEARCH
        self.PVAL = PVAL
        self.SEED = SEED      
        self.tokens = []  
        self.results = {}
        
    def search_hyperparams(self, X_l, y_l, X_val, y_val):
        ps = PredefinedSplit( np.array( [0]*len(y_l)+[-1]*len(y_val) ) )
        y_l_val = copy.deepcopy( np.hstack((y_l, y_val)) )
        regexes, opt, tokens = get_tokens(X_l, y_l, self.NGRAM_SIZE, self.FILENAME)
        X_l_aux = copy.deepcopy( get_matrix(tokens, X_l, regexes, opt) )
        X_val_aux = copy.deepcopy( get_matrix(tokens, X_val, regexes, opt) )
        X_l_val = copy.deepcopy( np.vstack((X_l_aux, X_val_aux)) )
        return best_model(self.MODEL, ps, X_l_val, y_l_val)

    def model_selection(self, X_train, y_train, X_test, FILENAME, results=False, return_model=False, probs=True, error=False, X_train_full=[]):
    #def model_selection(self, X_train, y_train, X_test, N, FILENAME, results=False, return_model=False, probs=True, error=False, X_train_full=[]):
        if self.GRID_SEARCH:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                              test_size=0.2, 
                                                              shuffle = False, 
                                                              random_state = self.SEED)
            self.HYPERPARAMS = self.search_hyperparams(X_train, y_train, X_val, y_val)
            self.GRID_SEARCH = False #just once
        
        #print(self.GRID_SEARCH, self.HYPERPARAMS)

        y_l = copy.deepcopy(y_train)
                
        regexes, opt, tokens = get_tokens(X_train,y_train,self.NGRAM_SIZE, self.FILENAME)
        
        seed_everything()
        if 'bert' not in self.MODEL and 'cregex' not in self.MODEL:
            X_l_aux = copy.deepcopy( get_matrix(tokens, X_train, regexes, opt) )
            X_train_aux = copy.deepcopy( get_matrix(tokens, X_train_full, regexes, opt) )
            X_test_aux = copy.deepcopy( get_matrix(tokens, X_test, regexes, opt) )
            model = select_trad_model(self.MODEL, self.HYPERPARAMS)       
        elif 'cregex' in self.MODEL:
            #MODEL_NAME = self.MODEL.split('-')[1]
            #model = CREGEX(self.FILENAME, self.NGRAM_SIZE, self.HYPERPARAMS, MODEL_NAME, self.N_CLASSES)
            #model = CREGEX(self.FILENAME, self.NGRAM_SIZE, self.HYPERPARAMS, self.MODEL, self.N_CLASSES)
            model = CREGEX(self.FILENAME, self.MODEL, self.N_CLASSES, self.CURVE)
            X_l_aux = copy.deepcopy( X_train )
            X_train_aux = copy.deepcopy( X_train_full )
            X_test_aux = copy.deepcopy( X_test )     
        elif 'bert' in self.MODEL:
            #model = BERT(**self.HYPERPARAMS)
            model = BERT(**self.HYPERPARAMS['bert'])
            '''
            if 'fregex' in self.NGRAM_SIZE:
                X_l_aux = get_texts(X_train, regexes)
                X_train_aux = get_texts(X_train_full, regexes)
                X_test_aux = get_texts(X_test, regexes)
                #save_txt(X_l_aux, os.path.join(os.getcwd(), 'out'), 'XTRAIN_'+self.FILENAME+'.txt')
                #save_txt(X_test_aux, os.path.join(os.getcwd(), 'out'), 'XTEST_'+self.FILENAME+'.txt')
            '''
            #else:
            X_l_aux = copy.deepcopy( X_train )
            X_train_aux = copy.deepcopy( X_train_full )
            X_test_aux = copy.deepcopy( X_test )
            
        model.fit(X_l_aux, y_l)
        if probs:
            pred = model.predict_proba(X_test_aux)
            if error:
                pred = [model.predict_proba(X_train_aux), pred]
        else:
            pred = model.predict(X_test_aux)#, self.y_test)
            if error:
                pred = [model.predict(X_train_aux), pred]
        del X_l_aux
        del X_train_aux
        del y_train
        del y_l
        del X_test_aux
        del regexes
        del opt
        del tokens
        gc.collect()
        if return_model:
            return pred, model
        else:
            del model
            gc.collect()
            return pred
        
    def start(self):     
        X_l = np.array([])
        y_l = np.array([])
        VAL_LENGTH = int( np.ceil(self.BATCH*self.PVAL) )
        classes_ = copy.deepcopy(self.y_train[ : VAL_LENGTH ] )
        while len(set(classes_)) != self.N_CLASSES:
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state = self.SEED)
            classes_ = copy.deepcopy(self.y_train[ : VAL_LENGTH ] )
        del classes_
        gc.collect()
        self.X_val = self.X_train[ : VAL_LENGTH ]
        self.y_val = self.y_train[ : VAL_LENGTH ]
        self.X_train = self.X_train[VAL_LENGTH:]
        self.y_train = self.y_train[VAL_LENGTH:]
        if self.GRID_SEARCH:
            self.HYPERPARAMS = self.search_hyperparams(self.X_train, self.y_train, self.X_val, self.y_val)
            self.GRID_SEARCH = False #just once
        classes_ = copy.deepcopy(self.y_train[:self.BATCH])
        while len(set(classes_)) != self.N_CLASSES:
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train, random_state = self.SEED)
            classes_ = copy.deepcopy(self.y_train[:self.BATCH])
        del classes_
        gc.collect()
        X_l = self.X_train[:self.BATCH]
        y_l = self.y_train[:self.BATCH]
        X_u = self.X_train[self.BATCH:]
        y_u = self.y_train[self.BATCH:]
        return X_l, y_l, X_u, y_u, [], []
        
    def valida_examples(self, X_l, y_l, X_u, y_clf, indexes):
        X_l_aux = copy.deepcopy( X_l )
        X_l_aux = np.concatenate( ( X_l_aux, X_u[indexes] ) )
        y_l_aux = copy.deepcopy( y_l )
        y_l_aux = np.concatenate( ( y_l_aux, y_clf[indexes] ) ) 
        pred = self.model_selection(X_l_aux, y_l_aux, self.X_val, self.FILENAME)#self.model_selection(X_l_aux, y_l_aux, self.X_val, self.NGRAM_SIZE, self.FILENAME)
        new_error = 1-accuracy_score( self.y_val, np.argmax(pred, axis=1) )
        del X_l_aux
        del y_l_aux
        gc.collect()
        return new_error, pred
    
    def pge_exp(self, x, r=1,y0=20/100):
        y = y0*(1+r)**x
        return y
        
    def learningCurve(self):    
        scores = []
        samples = [] 
        distribution = [] 
        prob = (1-self.THR_PRB)/(self.N_CLASSES-1)
        THR_ENTROPY = -self.THR_PRB*np.log2(self.THR_PRB)
        for times in range(self.N_CLASSES-1):
            THR_ENTROPY += -prob*np.log2(prob)
        COUNT = -1
        X_PGE = 0
        PBATCH = self.pge_exp(X_PGE)
        X_l, y_l, X_u, y_u, x, y = self.start()    

        #print('start', X_l.shape)

        if 'SSL' in self.CURVE:
            samples.append( self.BATCH )
            distribution.append( ['H']*self.BATCH )
            y_train_semi_aux = copy.deepcopy( y_l )
        if self.CURVE != 'PL':
            scores.append( [] )
        while len(X_u)>=0:   
            COUNT +=1
            x.append( len(y_l) )    
            if self.PRED_TYPE == 'error':
                #print(self.X_train[0])
                pred, clf = self.model_selection(X_l, y_l, self.X_test, self.FILENAME, False, True, True, True, self.X_train) #self.model_selection(X_l, y_l, self.X_test, self.NGRAM_SIZE, self.FILENAME, False, True, True, True, self.X_train)
                #x.append( len(y_l) )    
                y.append( [ np.argmax(pred[0], axis=1), self.y_train, np.argmax(pred[1], axis=1), self.y_test ] )  
            else:
                pred, clf = self.model_selection(X_l, y_l, self.X_test, self.FILENAME, False, True) #self.model_selection(X_l, y_l, self.X_test, self.NGRAM_SIZE, self.FILENAME, False, True)
                y.append( np.argmax(pred, axis=1) )  
                #print( 100*len(y_l)/len(self.X_train) )
                #print( 'acc:', round(100*accuracy_score( self.y_test, y[-1] ),2) )
            if len(X_u)==0:
                break
            indexes = np.array([], dtype = int)
            HALF = 0
            if self.CURVE == 'PL':
                indexes = shuffle( np.arange(len(X_u)), random_state=self.SEED )
            else:
                objeto = AL( X_u, X_l, y_l, clf, self.N_CLASSES, self.CURVE, self.NGRAM_SIZE, self.FILENAME, self.MODEL )
                scores_fx = copy.deepcopy(objeto.score_function())
                scores_aux = copy.deepcopy( scores_fx['scores'] )
                probs_aux = copy.deepcopy( scores_fx['probs'] )        
                y_clf = np.argmax( probs_aux, axis=1 ) #predictions on X_u
                if 'SSLAL' in self.CURVE and (COUNT%2)==0:                    
                    X_PGE+=1
                    indexes_probs_aux = np.where( scores_aux<THR_ENTROPY )[0][:int(self.BATCH*PBATCH)]
                    pred = self.model_selection(X_l, y_l, self.X_val, self.FILENAME) #self.model_selection(X_l, y_l, self.X_val, self.NGRAM_SIZE, self.FILENAME)
                    probs_val_aux = copy.deepcopy(pred)
                    error_base = 1-accuracy_score( self.y_val, np.argmax(pred, axis=1) ) 
                    del pred
                    gc.collect()
                    if 'SSLAL' in self.CURVE:
                        new_error = np.Inf
                        idxs_scores_aux_sorted = indexes_probs_aux[np.argsort(scores_aux[indexes_probs_aux])[::-1]] #entropy +++ to ---
                        while new_error>error_base and len(indexes_probs_aux)>0:    
                            new_error, pred = self.valida_examples(X_l, y_l, X_u, y_clf, indexes_probs_aux)
                            probs_val_aux = copy.deepcopy( pred ) 
                            indexes = copy.deepcopy(indexes_probs_aux) #desordenado
                            del pred
                            gc.collect()
                            val_idx_probs = idxs_scores_aux_sorted[0] #idx mayor entropia
                            idx_probs = np.where( idxs_scores_aux_sorted==val_idx_probs )  
                            idxs_scores_aux_sorted = np.delete( idxs_scores_aux_sorted, idx_probs, axis=0 ) #ok
                            idx_probs = np.where( indexes_probs_aux==val_idx_probs )  
                            indexes_probs_aux = np.delete( indexes_probs_aux, idx_probs , axis=0 )
                    del indexes_probs_aux
                    del probs_val_aux
                    gc.collect()
                    PBATCH = self.pge_exp(X_PGE)
                    if PBATCH>1:
                        PBATCH = 1    
                if self.CURVE != 'PL' and len(indexes)<=self.BATCH: #AL/SSL(par, imcompleto)/SSL(impar)
                    HALF = len(indexes)
                    indexes_entropy = np.argsort( scores_aux )[::-1] # entropy +++ to ---
                    for idx in indexes_entropy:
                        if idx not in indexes:
                            indexes = np.concatenate( (indexes, np.array([idx]) ) )
                    if 'SSLAL' in self.CURVE:
                        samples.append( x[-1]+len(indexes[HALF:self.BATCH]) ) 
                        distribution.append( ['C']*HALF+['H']*len(indexes[HALF:self.BATCH]) )           
            if self.CURVE in ['PL', 'AL']: 
                X_l = np.concatenate((X_l, X_u[indexes[:self.BATCH]] ))
                y_l = np.concatenate((y_l, y_u[indexes[:self.BATCH]] ))
                if self.CURVE == 'AL':
                    scores.append( scores_aux[indexes] )  #all but sorted scores according indexes    
            else: #SSLAL
                scores.append( scores_aux[indexes] )  #all but sorted scores according indexes
                X_l = np.concatenate((X_l, X_u[indexes[:self.BATCH]] ))
                if 'SSLAL' in self.CURVE:
                    y_l_aux = copy.deepcopy( y_clf[indexes[:HALF]] )
                    y_l_aux = np.concatenate((y_l_aux, y_u[indexes[HALF:self.BATCH]] ))
                    y_l = np.concatenate((y_l, y_l_aux ))
                    p_aux = copy.deepcopy( probs_aux[indexes[:HALF]] )
                    p_aux = np.concatenate((p_aux, probs_aux[indexes[HALF:]] )) #all but sorted according indexes
                    del y_l_aux
                    del p_aux
                    gc.collect()
                    y_train_semi_aux = np.concatenate((y_train_semi_aux, y_clf[indexes[:HALF]] ))
                    if HALF>0:
                        y_train_semi_aux[-HALF:] = -1
                    y_train_semi_aux = np.concatenate((y_train_semi_aux, y_u[indexes[HALF:self.BATCH]] ))
            X_u = np.delete(X_u, indexes[:self.BATCH], axis = 0)
            y_u = np.delete(y_u, indexes[:self.BATCH], axis = 0)
            if len(X_u)<=0 and 'SSLAL' not in self.CURVE:
                X_l = copy.deepcopy( self.X_train )
                y_l = copy.deepcopy( self.y_train )
                #del self.X_train
                #del self.y_train
                gc.collect()
            if 'SSLAL' in self.CURVE:  
                indexes_semi_aux = np.where( y_train_semi_aux<0 )[0] #pseudo-labels
                indexes_l_aux = np.where( y_train_semi_aux>=0 )[0] #labels
                if len(indexes_semi_aux)>0:
                    pred = self.model_selection(X_l[indexes_l_aux], y_l[indexes_l_aux], X_l[indexes_semi_aux], self.FILENAME) #self.model_selection(X_l[indexes_l_aux], y_l[indexes_l_aux], X_l[indexes_semi_aux], self.NGRAM_SIZE, self.FILENAME)
                    y_l[indexes_semi_aux] = np.argmax(pred, axis=1)
                    del pred
                del indexes_semi_aux
                del indexes_l_aux
                gc.collect()
            del clf
            gc.collect()
        del X_l
        del y_l
        del X_u
        del y_u
        del self.X_test
        del self.y_test
        del self.X_val
        del self.y_val
        gc.collect()
        x = np.array(x)
        y = np.array(y)
        self.results['x'] = x
        self.results['y'] = y
        self.results['scores'] = np.array(scores)
        self.results['samples'] = np.array(samples)
        self.results['distribution'] = np.array(distribution)
                            