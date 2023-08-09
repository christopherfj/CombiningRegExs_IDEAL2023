from utils import *
from fregex import FREGEX
from bert import BERT

class CREGEX(object):
    def __init__(self, 
                #FILENAME, NGRAM_SIZE, HYPERPARAMS, MODEL_NAMES, N_CLASSES, 
                FILENAME, MODEL_NAMES, N_CLASSES, CURVE,
                NGRAM_MIN=NGRAM_MIN, pnumbers=pnumbers, augmentation=False, 
                gap_cmb=gap_cmb, whitespaces=whitespaces, lexicon=lexicon, HYPERPARAMS=HYPERPARAMS, SEED=SEED):
        self.__metaclass__ = 'CREGEX'
        self.FILENAME = FILENAME
        _,clfs = MODEL_NAMES.split('*')
        clfs = clfs.split('.')
        self.MODEL_NAMES = [clf for clf in clfs]
        self.N_CLASSES = N_CLASSES
        self.CURVE = CURVE
        self.NGRAM_MIN = NGRAM_MIN
        self.pnumbers=pnumbers
        self.augmentation = augmentation
        self.gap_cmb = gap_cmb
        self.whitespaces = whitespaces
        self.lexicon = lexicon
        self.SEED = SEED
        self.HYPERPARAMS = HYPERPARAMS
        self.regexes = {}
        self.labeled_regexes = {}
        self.labeled_regexes_filtered = {}
        self.labeled_regexes_all = {}
        self.tokens = defaultdict(dict)
        self.kw = []
        self.distribution = defaultdict(list)
        self.y = defaultdict(list)
        self.rndm = np.random.RandomState(self.SEED)
        self.models = {}

    def fit(self, X,y):
        fregex = FREGEX(X, y, self.FILENAME)
        fregex.fit()
        self.regexes.update( fregex.transform() )        
        self.kw = copy.deepcopy(self.lexicon[self.FILENAME])
        self.pattern2token = copy.deepcopy( fregex.pattern2token )
        self.token2pattern = copy.deepcopy( fregex.token2pattern )
        self.tokens2pos = copy.deepcopy( fregex.tokens2pos )
        self.stopwords = copy.deepcopy(fregex.stopwords)
        self.regexes, self.regex2class = get_classes_regexes(self.regexes, y, self.tokens2pos)

        if self.augmentation:
            self.regexes, self.regex2class = augmentation(self.regexes, self.regex2class, self.kw, self.pattern2token)

        print(self.kw)

        labeled_regexes, labeled_regexes_filtered, labeled_regexes_all = get_filtered_regexes(self.regexes, y, self.kw, self.pattern2token, self.regex2class )
        self.labeled_regexes.update( labeled_regexes )
        self.labeled_regexes_filtered.update( labeled_regexes_filtered )
        self.labeled_regexes_all.update( labeled_regexes_all )

        del labeled_regexes
        del labeled_regexes_filtered
        del labeled_regexes_all
        gc.collect()

        keys = copy.deepcopy( list( self.regexes.keys() ) )
        for key in keys:
            if key not in self.labeled_regexes:
                self.regexes.pop(key)
                self.regex2class.pop(key)

        for MODEL_NAME in self.MODEL_NAMES:
            tokens = None
            opt = None
            regexes_aux = None
            model = None
            X_train = None
            y_train = None

            if 'sw' not in MODEL_NAME and 'random' not in MODEL_NAME: #clf                
                print('cregex-'+MODEL_NAME+'...fit')
                seed_everything()
                if 'bert' not in MODEL_NAME:
                    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                                    test_size=0.2, 
                                                                    shuffle = False, 
                                                                    random_state = self.SEED)
                    _, NGRAM_SIZE = MODEL_NAME.split('-')
                    NGRAM_SIZE = int(NGRAM_SIZE.replace('n',''))
                    tokens = n_grams(X_train, NGRAM_SIZE)
                    opt = False
                    regexes_aux = {}

                    y_l_aux = copy.deepcopy(y_train)                    
                    X_l_aux = copy.deepcopy( get_matrix(tokens, X_train, regexes_aux, opt) )
                    X_val_aux = copy.deepcopy( get_matrix(tokens, X_val, regexes_aux, opt) )
                    
                    X_train_val = copy.deepcopy( np.vstack((X_l_aux, X_val_aux)) )            
                    ps = PredefinedSplit( np.array( [0]*len(y_train)+[-1]*len(y_val) ) )
                    y_train_val = copy.deepcopy( np.hstack((y_train, y_val)) )
                    HYPERPARAMS = best_model(MODEL_NAME, ps, X_train_val, y_train_val)
                    model = select_trad_model(MODEL_NAME, HYPERPARAMS)
                    self.HYPERPARAMS[MODEL_NAME] = HYPERPARAMS
                    self.tokens[MODEL_NAME] = tokens
                    self.HYPERPARAMS['cregex-'+MODEL_NAME] = HYPERPARAMS
                    self.tokens['cregex-'+MODEL_NAME] = tokens

                else:
                    X_l_aux = copy.deepcopy(X)
                    y_l_aux = copy.deepcopy(y)
                    model = BERT(**self.HYPERPARAMS['bert'])

                model.fit(X_l_aux, y_l_aux)

            elif 'sw' in MODEL_NAME:
                print('cregex-regexes...fit')
                X_train = copy.deepcopy(X)
                y_train = copy.deepcopy(y)

            self.models[MODEL_NAME] = [tokens, opt, regexes_aux, model, X_train, y_train]            
            
    #def predict(self, X, y_true, proba=False):
    def predict(self, X, proba=False):
        #predictions_aux = {}
        for MODEL_NAME in self.MODEL_NAMES:
            print(MODEL_NAME)
            tokens, opt, regexes_aux, model, X_train, y_train = self.models[MODEL_NAME]
            if 'sw' not in MODEL_NAME and 'random' not in MODEL_NAME: #-clf
                if 'bert' not in MODEL_NAME:
                    X_test_aux = copy.deepcopy( get_matrix(tokens, X, regexes_aux, opt) )
                else:
                    X_test_aux = copy.deepcopy(X)
                if proba:
                    predictions = model.predict_proba( X_test_aux )
                else:
                    predictions = model.predict( X_test_aux )                    
            elif 'sw' in MODEL_NAME:
                X_test_aux = copy.deepcopy(X)            
                predictions = get_class_similarity( X_train, y_train, X_test_aux, self.regexes, self.token2pattern, self.stopwords, self.FILENAME, proba, True )
            elif 'random' in MODEL_NAME:
                seed_everything()
                if proba:
                    predictions = []
                    for _ in range(len(X)):
                        #pred = self.rndm.randint(0,100+1, size=self.N_CLASSES) 
                        pred = self.rndm.randint(0, self.N_CLASSES) 
                        pond = 1/(self.N_CLASSES+1)
                        preds = np.ones(self.N_CLASSES)*pond
                        preds[pred] = 1-pond
                        predictions.append(preds)
                    predictions = np.array(predictions)
                    print(np.argmax(predictions, axis=1))
                else:
                    predictions = self.rndm.randint(0, self.N_CLASSES, size=len(X))
                    #predictions = self.rndm.randint(0, self.N_CLASSES+1, size=len(X))
            #predictions_aux[MODEL_NAME] = predictions
            self.y[MODEL_NAME] = copy.deepcopy(predictions) #list(predictions)

        i = -1        

        for text in X:
            i+=1
            flag = False
            labels = []
            confs = []
            regexs = []
            max_conf = []
            idxs_max = []
            regexs_labels = []
            rlc = []

            for regex in self.labeled_regexes:
                label, conf = self.labeled_regexes[regex]
                _, numbers_aux, _, _, _ = self.regexes[regex]
                f = findall(regex, [], numbers_aux, text)
                
                if f:
                    flag = True
                    labels.append(label)
                    confs.append(conf)
                    regexs.append(regex)

            if flag:

                regexs = np.array(regexs)
                labels = np.array(labels)
                confs = np.array(confs)

                rlc = list(zip(regexs, labels, confs))                
                eps = 1e+4
                rlc = sorted( rlc, 
                          key=lambda x:x[2]+len( re.split(r'(?:%s|%s)' %(re.escape(self.gap_cmb), re.escape(self.whitespaces)), x[0]))/eps, 
                          reverse=True)
                
                classe = rlc[0][1]
                
                if proba:
                    max_conf = rlc[0][2]#np.max(confs)
                    pos_aux = copy.deepcopy(classe)
                    #print(predictions.shape)
                    pond = (1-max_conf)/( predictions.shape[1]-1)
                    classe =  np.ones(predictions.shape[1])*pond
                    classe[ pos_aux ] = max_conf

                for MODEL_NAME in self.MODEL_NAMES:
                    if proba:
                        if len(self.y['cregex-'+MODEL_NAME])==0:
                            self.y['cregex-'+MODEL_NAME] = classe
                        else:
                            self.y['cregex-'+MODEL_NAME] = np.vstack(( self.y['cregex-'+MODEL_NAME],  classe))
                    else:
                        self.y['cregex-'+MODEL_NAME].append( classe )

                r_aux, l_aux, c_aux = zip(*rlc)
                self.distribution['predict'].append( ('rex', [list(r_aux), list(l_aux), list(c_aux)]) )

                #if y_true[i] != classe:
                #    print('\n\n', X[i], '\n-', 'true: ', y_true[i], ' - pred: ', classe, '-', rlc, flag)
                
            else:
                
                for MODEL_NAME in self.MODEL_NAMES:
                    
                    #classe = predictions_aux[MODEL_NAME][i]
                    classe = self.y[MODEL_NAME][i]
                
                    if proba:
                        if len(self.y['cregex-'+MODEL_NAME])==0:
                            self.y['cregex-'+MODEL_NAME] = classe
                        else:
                            self.y['cregex-'+MODEL_NAME] = np.vstack(( self.y['cregex-'+MODEL_NAME],  classe))  

                    else:
                        self.y['cregex-'+MODEL_NAME].append( classe )
                 
                    #if y_true[i] != classe:
                    #    print('\n\n', X[i], '\n-', 'true: ', y_true[i], ' - pred: ', classe, '-', rlc, flag)

                self.distribution['predict'].append( ('clf', None) )



        values = self.labeled_regexes.items()
        r_aux = [r for r,lc in values]
        l_aux = [lc[0] for r,lc in values]
        c_aux = [lc[1] for r,lc in values]
        self.distribution['fit'] =  ['rex', [r_aux, l_aux, c_aux] ]

        values = self.labeled_regexes_filtered.items()
        r_aux = [r for r,lc in values]
        l_aux = [lc[0] for r,lc in values]
        c_aux = [lc[1] for r,lc in values]
        self.distribution['fit_filtered'] =  ['rex', [r_aux, l_aux, c_aux] ]

        values = self.labeled_regexes_all.items()
        r_aux = [r for r,lc in values]
        l_aux = [lc[0] for r,lc in values]
        c_aux = [lc[1] for r,lc in values]
        self.distribution['fit_all'] =  ['rex', [r_aux, l_aux, c_aux] ]

        #print(self.y)

        for key in self.y:
            self.y[key] = np.array(self.y[key])

        if self.CURVE == 'RESULTS':
            return self.y
        else:
            return list(self.y.values())[0]

    def predict_proba(self, X):
        return self.predict(X, True)

