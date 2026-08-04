# Probe: rich summary alone vs. summary + curated cells

One Haiku enriched-card call per notebook. For each: read the **summary alone** first, then ask whether the **selected cells** add transferable specifics it missed.



## spooky-author-identification — nb 424628  (score=1.11891, 15 cells)

*selected 4/15 cells; mean position 0.50 (0=start,1=end); summary ~486 tok vs selected cells ~4939 tok*


**Summary (summary-alone condition):**

This notebook tackles multi-class text classification (authorship attribution) using a stacked ensemble approach combining shallow and deep models. The preprocessing pipeline includes text cleaning (lowercasing, removing non-alphabetic characters), author-specific vocabulary extraction via frequency-ratio thresholding (>2.2× relative frequency advantage between authors, minimum 100 occurrences), and engineered features: word counts per author (c_wd_eap/hpl/mws), punctuation ratios per character, stop-word percentage, text statistics (num_words, num_unique_words, num_chars, mean_word_len), and duplicate word ratio. Multiple representation layers feed into base learners: (1) Multinomial Naive Bayes on count-vectorized bigrams/trigrams and TF-IDF character n-grams (1–5), generating 6 probability columns; (2) spaCy-based lemmatization + stopword/punctuation removal producing 300-dim word vectors via spaCy's en_core_web_lg; (3) Doc2Vec trained on cleaned corpus with min_count=1, window=10, vector_size=300, yielding 300-dim embeddings; (4) TruncatedSVD on character-level TF-IDF (20 components, arpack solver); (5) Keras Conv1D RNN with Embedding(10000, 32)→Conv1D(64,5)→MaxPooling→Dense(800)→softmax on padded sequences (max_len=70), 4 epochs, 32 batch size, trained via 4-fold CV; (6) FastText-style Embedding(input_dim, 20)→GlobalAveragePooling1D→softmax on bigram-enhanced sequences (maxlen=300), 28 epochs, 3-fold CV. All base model predictions (6+300+300+20+3+3=632 features total) are stacked and fed to XGBoost with objectives multi:softprob, max_depth=3, eta=0.1, min_child_weight=1, colsample_bytree=0.7, subsample=0.8, num_rounds=2000 with early stopping on validation log loss (30 rounds patience), evaluated via 4-fold log-loss CV. The validation strategy uses stratified K-fold splits across all stages (5 folds for Naive Bayes, 4 for XGBoost, 3 for FastText, single 4-fold for NN) with log_loss as the optimization metric.


**Why cells add value (LLM):** Cell 2 shows the exact author-specific word filtering logic (>2.2× threshold, minimum 100-occurrence filter, derivative features). Cell 3 demonstrates the stacked base-learner engineering (MNB on count/char-TFIDF with 5-fold CV, statistical features like stop-word %). Cell 11 encodes the neural network and FastText architectures (embedding dims, padding lengths, epoch counts, optimizer choices). Cell 12 specifies the final XGBoost hyperparameters (max_depth=3, colsample=0.7, early_stopping=30) and 4-fold validation protocol.


**Selected cells (indices [2, 3, 11, 12]):**


```python
# cell [2] of 15
# Clean data
def clean(X_train,X_test):
    X_train['words'] = [re.sub("[^a-zA-Z]"," ", data).lower().split() for data in X_train['text']]
    X_test['words'] = [re.sub("[^a-zA-Z]"," ", data).lower().split() for data in X_test['text']]
    return X_train,X_test
X_train,X_test = clean(X_train,X_test)
print('Leaning words...',round(time()-start,0))

auth_wds = {'EAP':0,'HPL':0,'MWS':0}

wd = {}
for i, row in X_train.iterrows():
    for a in row['words']:
        if len(a) > 1:
            try: 
                wd[a][row['author']] = wd[a][row['author']] + 1
                auth_wds[row['author']] = auth_wds[row['author']] + 1
            except:
                c_eap = 0
                c_hpl = 0
                c_mws = 0
                try: c_eap = wd[a]['EAP'] 
                except: pass
                try: c_hpl = wd[a]['HPL'] 
                except: pass
                try: c_mws = wd[a]['EAP'] 
                except: pass   
                wd[a] = {'EAP':c_eap,'HPL':c_hpl,'MWS':c_mws}
                wd[a][row['author']] = wd[a][row['author']] + 1
                auth_wds[row['author']] = auth_wds[row['author']] + 1
                
def remove_key(dictionary,key):
    r = dict(dictionary)
    del r[key]
    return r        

for key in list(wd.keys()):
    pass
    if wd[key]['EAP'] + wd[key]['HPL'] + wd[key]['MWS'] < 100: 
        wd = remove_key(wd,key)
        
c_eap = 0
c_hpl = 0
c_mws = 0    
for key in list(wd.keys()): 
    pass
    if not any([(wd[key]['EAP']/auth_wds['EAP'])/((wd[key]['HPL']+1)/auth_wds['HPL'])>2.2,
          (wd[key]['EAP']/auth_wds['EAP'])/((wd[key]['HPL']+1)/auth_wds['HPL'])>2.2,
          (wd[key]['HPL']/auth_wds['HPL'])/((wd[key]['EAP']+1)/auth_wds['EAP'])>2.2,
          (wd[key]['HPL']/auth_wds['HPL'])/((wd[key]['MWS']+1)/auth_wds['MWS'])>2.2,
          (wd[key]['MWS']/auth_wds['MWS'])/((wd[key]['EAP']+1)/auth_wds['EAP'])>2.2,
         ( wd[key]['MWS']/auth_wds['MWS'])/((wd[key]['HPL']+1)/auth_wds['HPL'])>2.2]):
        wd = remove_key(wd,key)

col_wds = {}
for key in list(wd.keys()): 
    col_wds[key]=0

rows = []
for words in X_train['words']:
    line_wds = dict(col_wds)
    for word in words:
        try: line_wds[word] = line_wds[word] + 1
        except: pass
    row = [line_wds[key] for key in list(line_wds.keys())]
    rows.append(row)
pd_df = pd.DataFrame(rows)
    
for column in pd_df: 
    pass
    X_train['wd_'+str(column)] = pd_df[column]

rows = []
for words in X_test['words']:
    line_wds = dict(col_wds)
    for word in words:
        try: line_wds[word] = line_wds[word] + 1
        except: pass
    row = [line_wds[key] for key in list(line_wds.keys())]
    rows.append(row)
pd_df = pd.DataFrame(rows)
    
for column in pd_df: 
    pass
    X_test['wd_'+str(column)] = pd_df[column]

auth_wds = {'EAP':0,'HPL':0,'MWS':0}

wd = {}
for i, row in X_train.iterrows():
    for a in row['words']:
        if len(a) > 1:
            try: 
                wd[a][row['author']] = wd[a][row['author']] + 1
                auth_wds[row['author']] = auth_wds[row['author']] + 1
            except:
                c_eap = 0
                c_hpl = 0
                c_mws = 0
                try: c_eap = wd[a]['EAP'] 
                except: pass
                try: c_hpl = wd[a]['HPL'] 
                except: pass
                try: c_mws = wd[a]['EAP'] 
                except: pass   
                wd[a] = {'EAP':c_eap,'HPL':c_hpl,'MWS':c_mws}
                wd[a][row['author']] = wd[a][row['author']] + 1
                auth_wds[row['author']] = auth_wds[row['author']] + 1
                
def remove_key(dictionary,key):
    r = dict(dictionary)
    del r[key]
    return r        

for key in list(wd.keys()):
    pass
    if wd[key]['EAP'] + wd[key]['HPL'] + wd[key]['MWS'] < 5: 
        wd = remove_key(wd,key)
        
   
e = auth_wds['EAP']
h = auth_wds['HPL']
m = auth_wds['MWS']
eap_wds = []
hpl_wds = []
mws_wds = []
for key in list(wd.keys()): 
    pass
    if (wd[key]['EAP']/e)>((wd[key]['HPL']/h)+(wd[key]['MWS'])/m):
        eap_wds.append(key)
    elif (wd[key]['HPL']/e)>((wd[key]['EAP']/e)+(wd[key]['MWS'])/m):
        hpl_wds.append(key)
    elif (wd[key]['MWS']/e)>((wd[key]['HPL']/h)+(wd[key]['EAP'])/e):
        mws_wds.append(key)
c_wd_rows_eap = []
c_wd_rows_hpl = []
c_wd_rows_mws = []
dup_wds = []
for row in X_train['words']:
    if len(row) == len(set(row)): dup_wds.append(0)
    else: dup_wds.append((len(row)-len(set(row)))/len(row)*10)
    for word in row:
        c_eap = 0
        c_hpl = 0
        c_mws = 0 
        if word in eap_wds: c_eap+=1
        elif word in hpl_wds: c_hpl+=1
        elif word in mws_wds: c_mws+=1
    c_wd_rows_eap.append(c_eap)
    c_wd_rows_hpl.append(c_hpl)
    c_wd_rows_mws.append(c_mws)
X_train['c_wd_eap'] = c_wd_rows_eap  
X_train['c_wd_hpl'] = c_wd_rows_hpl  
X_train['c_wd_mws'] = c_wd_rows_mws 
X_train['dup_wds'] = dup_wds
c_wd_rows_eap = []
c_wd_rows_hpl = []
c_wd_rows_mws = []
dup_wds = []
for row in X_test['words']:
    if len(row) == len(set(row)): dup_wds.append(0)
    else: dup_wds.append((len(row)-len(set(row)))/len(row)*10)
    for word in row:
        c_eap = 0
        c_hpl = 0
        c_mws = 0 
        if word in eap_wds: c_eap+=1
        elif word in hpl_wds: c_hpl+=1
        elif word in mws_wds: c_mws+=1
    c_wd_rows_eap.append(c_eap)
    c_wd_rows_hpl.append(c_hpl)
    c_wd_rows_mws.append(c_mws)
X_test['c_wd_eap'] = c_wd_rows_eap  
X_test['c_wd_hpl'] = c_wd_rows_hpl  
X_test['c_wd_mws'] = c_wd_rows_mws 
X_test['dup_wds'] = dup_wds

print('Characters...',round(time()-start,0))
all_char = set([i for i in str(X_train['text'])])
for char in all_char:
    X_train['punc_'+char] = [(sum([1  for nchar in sentence if nchar == char])/len(sentence)) for sentence in X_train['text']]
    X_test['punc_'+char] = [(sum([1  for nchar in sentence if nchar == char])/len(sentence)) for sentence in X_test['text']]
```

```python
# cell [3] of 15
from gensim.parsing.preprocessing import STOPWORDS

# Feature Engineering
# Stop Words
print('Other columns...',round(time()-start,0))
_dist_train = [x for x in X_train['words']]
X_train['stop_word'] = [len([word for word in sentence if word in STOPWORDS])*100.0/len(sentence) for sentence in _dist_train]

_dist_test = [x for x in X_test['words']]
X_test['stop_word'] = [len([word for word in sentence if word in STOPWORDS])*100.0/len(sentence) for sentence in _dist_test]  

## Number of words in the text ##
X_train["num_words"] = X_train["text"].apply(lambda x: len(str(x).split()))
X_test["num_words"] = X_test["text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
X_train["num_unique_words"] = X_train["text"].apply(lambda x: len(set(str(x).split())))
X_test["num_unique_words"] = X_test["text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
X_train["num_chars"] = X_train["text"].apply(lambda x: len(str(x)))
X_test["num_chars"] = X_test["text"].apply(lambda x: len(str(x)))

## Average length of the words in the text ##
X_train["mean_word_len"] = X_train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
X_test["mean_word_len"] = X_test["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

print('TFIDF...',round(time()-start,0))
### Fit transform the count vectorizer ###
tfidf_vec = CountVectorizer(stop_words=STOPWORDS, ngram_range=(1,3))
tfidf_vec.fit(X_train['text'].values.tolist() + X_test['text'].values.tolist())
train_tfidf = tfidf_vec.transform(X_train['text'].values.tolist())
test_tfidf = tfidf_vec.transform(X_test['text'].values.tolist())

# Feature Engineering
# count - words - nb
def countWords(X_train,X_test):
    count_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))
    count_vec.fit(X_train['text'].values.tolist() + X_test['text'].values.tolist())
    train_count = count_vec.transform(X_train['text'].values.tolist())
    test_count = count_vec.transform(X_test['text'].values.tolist())
    return train_count,test_count
    
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def do_count_MNB(X_train,X_test,Y_train):
    train_count,test_count=countWords(X_train,X_test)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([X_train.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = train_count[dev_index], train_count[val_index]
        dev_y, val_y = Y_train[dev_index], Y_train[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_count)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test /5.
    return pred_train,pred_full_test

pred_train,pred_test = do_count_MNB(X_train,X_test,Y_train)
X_train["count_words_nb_eap"] = pred_train[:,0]
X_train["count_words_nb_hpl"] = pred_train[:,1]
X_train["count_words_nb_mws"] = pred_train[:,2]
X_test["count_words_nb_eap"] = pred_test[:,0]
X_test["count_words_nb_hpl"] = pred_test[:,1]
X_test["count_words_nb_mws"] = pred_test[:,2]


# Feature Engineering
# tfidf - chars - nb
def tfidfWords(X_train,X_test):
    tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,5),analyzer='char')
    full_tfidf = tfidf_vec.fit_transform(X_train['text'].values.tolist() + X_test['text'].values.tolist())
    train_tfidf = tfidf_vec.transform(X_train['text'].values.tolist())
    test_tfidf = tfidf_vec.transform(X_test['text'].values.tolist())
    return train_tfidf,test_tfidf
    
def runMNB(train_X, train_y, test_X, test_y, test_X2):
    model = naive_bayes.MultinomialNB()
    model.fit(train_X, train_y)
    pred_test_y = model.predict_proba(test_X)
    pred_test_y2 = model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model

def do(X_train,X_test,Y_train):
    train_tfidf,test_tfidf = tfidfWords(X_train,X_test)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([X_train.shape[0], 3])
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=88)
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]
        dev_y, val_y = Y_train[dev_index], Y_train[val_index]
        pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("Mean cv score : ", np.mean(cv_scores))
    pred_full_test = pred_full_test /5.
    return pred_train,pred_full_test
pred_train,pred_test = do(X_train,X_test,Y_train)


X_train["tfidf_chars_nb_eap"] = pred_train[:,0]
X_train["tfidf_chars_nb_hpl"] = pred_train[:,1]
X_train["tfidf_chars_nb_mws"] = pred_train[:,2]
X_test["tfidf_chars_nb_eap"] = pred_test[:,0]
X_test["tfidf_chars_nb_hpl"] = pred_test[:,1]
X_test["tfidf_chars_nb_mws"] = pred_test[:,2]
print('SpaCy...',round(time()-start,0))
```

```python
# cell [11] of 15
# Using Neural Networks and Facebook's Fasttext
earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

# NN
def doAddNN(X_train,X_test,pred_train,pred_test):
    X_train["nn_eap"] = pred_train[:,0]
    X_train["nn_hpl"] = pred_train[:,1]
    X_train["nn_mws"] = pred_train[:,2]
    X_test["nn_eap"] = pred_test[:,0]
    X_test["nn_hpl"] = pred_test[:,1]
    X_test["nn_mws"] = pred_test[:,2]
    return X_train,X_test

def initNN(nb_words_cnt,max_len):
    model = Sequential()
    model.add(Embedding(nb_words_cnt,32,input_length=max_len))
    model.add(Dropout(0.3))
    model.add(Conv1D(64,
                     5,
                     padding='valid',
                     activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(800, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    return model

def doNN(X_train,X_test,Y_train):
    max_len = 70
    nb_words = 10000
    
    print('Processing text dataset')
    texts_1 = []
    for text in X_train['text']:
        texts_1.append(text)

    print('Found %s texts.' % len(texts_1))
    test_texts_1 = []
    for text in X_test['text']:
        test_texts_1.append(text)
    print('Found %s texts.' % len(test_texts_1))
    
    tokenizer = Tokenizer(num_words=nb_words)
    tokenizer.fit_on_texts(texts_1 + test_texts_1)
    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)

    xtrain_pad = pad_sequences(sequences_1, maxlen=max_len)
    xtest_pad = pad_sequences(test_sequences_1, maxlen=max_len)
    del test_sequences_1
    del sequences_1
    nb_words_cnt = min(nb_words, len(word_index)) + 1

    # we need to binarize the labels for the neural net
    ytrain_enc = np_utils.to_categorical(Y_train)
    
    kf = model_selection.KFold(n_splits=splits, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([xtrain_pad.shape[0], 3])
    for dev_index, val_index in kf.split(xtrain_pad):
        dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initNN(nb_words_cnt,max_len)
        model.fit(dev_X, y=dev_y, batch_size=32, epochs=4, verbose=1,
                  validation_data=(val_X, val_y),callbacks=[earlyStopping])
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(xtest_pad)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
    return doAddNN(X_train,X_test,pred_train,pred_full_test/splits)

# Fast Text

def doAddFastText(X_train,X_test,pred_train,pred_test):
    X_train["ff_eap"] = pred_train[:,0]
    X_train["ff_hpl"] = pred_train[:,1]
    X_train["ff_mws"] = pred_train[:,2]
    X_test["ff_eap"] = pred_test[:,0]
    X_test["ff_hpl"] = pred_test[:,1]
    X_test["ff_mws"] = pred_test[:,2]
    return X_train,X_test


def initFastText(embedding_dims,input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embedding_dims))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def preprocessFastText(text_docs):
    text_docs = text_docs.replace("' ", " ' ")
    signs = set(',.:;"?!')
    prods = set(text_docs) & signs
    if not prods:
        return text_docs

    for sign in prods:
        text_docs = text_docs.replace(sign, ' {} '.format(sign) )
    return text_docs

def create_docs(df, n_gram_max=2):
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
    docs = []
    for doc in df.text:
        doc = preprocessFastText(doc).split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
    return docs

def doFastText(X_train,X_test,Y_train):
    min_count = 2

    docs = create_docs(X_train)
    tokenizer = Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts(docs)
    num_words = sum([1 for _, v in tokenizer.word_counts.items() if v >= min_count])

    tokenizer = Tokenizer(num_words=num_words, lower=False, filters='')
    tokenizer.fit_on_texts(docs)
    docs = tokenizer.texts_to_sequences(docs)

    maxlen = 300

    docs = pad_sequences(sequences=docs, maxlen=maxlen)
    input_dim = np.max(docs) + 1
    embedding_dims = 20

    # we need to binarize the labels for the neural net
    ytrain_enc = np_utils.to_categorical(Y_train)

    docs_test = create_docs(X_test)
    docs_test = tokenizer.texts_to_sequences(docs_test)
    docs_test = pad_sequences(sequences=docs_test, maxlen=maxlen)
    xtrain_pad = docs 
    kf = model_selection.KFold(n_splits=3, shuffle=True, random_state=2017)
    pred_full_test = 0
    pred_train = np.zeros([xtrain_pad.shape[0], 3])
    for dev_index, val_index in kf.split(xtrain_pad):
        dev_X, val_X = xtrain_pad[dev_index], xtrain_pad[val_index]
        dev_y, val_y = ytrain_enc[dev_index], ytrain_enc[val_index]
        model = initFastText(embedding_dims,input_dim)
        model.fit(dev_X, y=dev_y, batch_size=32, epochs=28, verbose=1,
                  validation_data=(val_X, val_y),callbacks=[earlyStopping])
        pred_val_y = model.predict(val_X)
        pred_test_y = model.predict(docs_test)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
    return doAddFastText(X_train,X_test,pred_train,pred_full_test/3)
print('Other cool methods..',round(time()-start,0))
X_train,X_test = doFastText(X_train,X_test,Y_train)
X_train,X_test = doNN(X_train,X_test,Y_train)
```

```python
# cell [12] of 15
# Final Model
# XGBoost
def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = child
    param['subsample'] = 0.8
    param['colsample_bytree'] = colsample
    param['seed'] = seed_val
    num_rounds = 2000

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=30, verbose_eval=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)
    if test_X2 is not None:
        xgtest2 = xgb.DMatrix(test_X2)
        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    return pred_test_y, pred_test_y2, model

def do(X_train,X_test,Y_train):
    drop_columns=["id","text","words"]
    x_train = X_train.drop(drop_columns+['author'],axis=1)
    x_test = X_test.drop(drop_columns,axis=1)
    y_train = Y_train
    
    kf = model_selection.KFold(n_splits=4, shuffle=True, random_state=2017)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros([x_train.shape[0], 3])
    for dev_index, val_index in kf.split(x_train):
        dev_X, val_X = x_train.loc[dev_index], x_train.loc[val_index]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, x_test, seed_val=0, colsample=0.7)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index,:] = pred_val_y
        cv_scores.append(metrics.log_loss(val_y, pred_val_y))
    print("cv scores : ", cv_scores)
    return pred_full_test/4
result = do(X_train,X_test,Y_train)

result = pd.DataFrame(list(result),columns=['EAP','HPL','MWS'])
result['id'] = X_test['id']
result.to_csv('output_easier_process_version.csv',index=False)
print('Time to completion: ',round(time()-start,0))
```


## jigsaw-toxic-comment-classification-challenge — nb 3871679  (score=0.98702, 15 cells)

*selected 5/15 cells; mean position 0.57 (0=start,1=end); summary ~397 tok vs selected cells ~2605 tok*


**Summary (summary-alone condition):**

This notebook addresses multi-label text classification for toxic comment detection using deep learning. The modeling approach employs a stacked architecture: a frozen pretrained embedding layer (concatenated FastText and GloVe 300-dimensional vectors) feeding into a Bidirectional CuDNNGRU followed by Bidirectional CuDNNLSTM with return_sequences, followed by concatenated GlobalMaxPooling1D and GlobalAveragePooling1D. The hidden representation is refined through two additive residual-style Dense layers (512 units each with ReLU) before sigmoid output for six binary classification targets. Aggressive text normalization precedes tokenization: comments are cleaned via unidecode, asterisk-obfuscated words are restored (e.g., 'f**k' → 'fuck'), toxic words are explicitly isolated via recursive splitting, and multiple dictionary-based corrections are applied (hyphens, misspellings, fasttext-specific variants, acronyms). Tokenization uses a pretrained Keras Tokenizer with TweetTokenizer for splitting, followed by padding to MAX_LEN=220. Validation employs 90/10 stratified train-validation split evaluated on per-label ROC-AUC averaged across six classes. Training uses a novel epoch-wise learning rate schedule (decaying by 0.5 per global epoch, starting at 1e-3) within a multi-seed ensemble strategy (10 random seeds), with each seed trained for 5 epochs and predictions averaged. The model is trained on batches of 128 with binary crossentropy loss on both train and validation data, enabling early stopping-like behavior through learning rate decay without explicit callbacks.


**Why cells add value (LLM):** Cell 4 specifies thresholds and valid character sets for preprocessing; cells 5-6 implement the detailed text normalization pipeline with pattern-based contractions, dictionary lookups, and toxic word isolation; cell 12 defines the exact layer architecture and dimension choices; cell 13 shows the multi-seed ensemble training loop with epoch-wise learning rate scheduling and per-label AUC validation.


**Selected cells (indices [4, 5, 6, 12, 13]):**


```python
# cell [4] of 15
training_samples_count = 149571
validation_samples_count = 10000

length_threshold = 20000 #We are going to truncate a comment if its length > threshold
word_count_threshold = 900 #We are going to truncate a comment if it has more words than our threshold
words_limit = 310000

#We will filter all characters except alphabet characters and some punctuation
valid_characters = " " + "@$" + "'!?-" + "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper()
valid_characters_ext = valid_characters + "abcdefghijklmnopqrstuvwxyz".upper()
valid_set = set(x for x in valid_characters)
valid_set_ext = set(x for x in valid_characters_ext)

#List of some words that often appear in toxic comments
#Sorry about the level of toxicity in it!
toxic_words = ["poop", "crap", "prick", "twat", "wikipedia", "wiki", "hahahahaha", "lol", "bastard", "sluts", "slut", "douchebag", "douche", "blowjob", "nigga", "dumb", "jerk", "wanker", "wank", "penis", "motherfucker", "fucker", "fuk", "fucking", "fucked", "fuck", "bullshit", "shit", "stupid", "bitches", "bitch", "suck", "cunt", "dick", "cocks", "cock", "die", "kill", "gay", "jewish", "jews", "jew", "niggers", "nigger", "faggot", "fag", "asshole"]
astericks_words = [('mother****ers', 'motherfuckers'), ('motherf*cking', 'motherfucking'), ('mother****er', 'motherfucker'), ('motherf*cker', 'motherfucker'), ('bullsh*t', 'bullshit'), ('f**cking', 'fucking'), ('f*ucking', 'fucking'), ('fu*cking', 'fucking'), ('****ing', 'fucking'), ('a**hole', 'asshole'), ('assh*le', 'asshole'), ('f******', 'fucking'), ('f*****g', 'fucking'), ('f***ing', 'fucking'), ('f**king', 'fucking'), ('f*cking', 'fucking'), ('fu**ing', 'fucking'), ('fu*king', 'fucking'), ('fuc*ers', 'fuckers'), ('f*****', 'fucking'), ('f***ed', 'fucked'), ('f**ker', 'fucker'), ('f*cked', 'fucked'), ('f*cker', 'fucker'), ('f*ckin', 'fucking'), ('fu*ker', 'fucker'), ('fuc**n', 'fucking'), ('ni**as', 'niggas'), ('b**ch', 'bitch'), ('b*tch', 'bitch'), ('c*unt', 'cunt'), ('f**ks', 'fucks'), ('f*ing', 'fucking'), ('ni**a', 'nigga'), ('c*ck', 'cock'), ('c*nt', 'cunt'), ('cr*p', 'crap'), ('d*ck', 'dick'), ('f***', 'fuck'), ('f**k', 'fuck'), ('f*ck', 'fuck'), ('fc*k', 'fuck'), ('fu**', 'fuck'), ('fu*k', 'fuck'), ('s***', 'shit'), ('s**t', 'shit'), ('sh**', 'shit'), ('sh*t', 'shit'), ('tw*t', 'twat')]
fasttext_misspelings = {"'n'balls": 'balls', "-nazi's": 'nazis', 'adminabuse': 'admin abuse', "admins's": 'admins', 'arsewipe': 'arse wipe', 'assfack': 'asshole', 'assholifity': 'asshole', 'assholivity': 'asshole', 'asshoul': 'asshole', 'asssholeee': 'asshole', 'belizeans': 'mexicans', "blowing's": 'blowing', 'bolivians': 'mexicans', 'celtofascists': 'fascists', 'censorshipmeisters': 'censor', 'chileans': 'mexicans', 'clerofascist': 'fascist', 'cowcrap': 'crap', 'crapity': 'crap', "d'idiots": 'idiots', 'deminazi': 'nazi', 'dftt': "don't feed the troll", 'dildohs': 'dildo', 'dramawhores': 'drama whores', 'edophiles': 'pedophiles', 'eurocommunist': 'communist', 'faggotkike': 'faggot', 'fantard': 'retard', 'fascismnazism': 'fascism', 'fascistisized': 'fascist', 'favremother': 'mother', 'fuxxxin': 'fucking', "g'damn": 'goddamn', 'harassmentat': 'harassment', 'harrasingme': 'harassing me', 'herfuc': 'motherfucker', 'hilterism': 'fascism', 'hitlerians': 'nazis', 'hitlerites': 'nazis', 'hubrises': 'pricks', 'idiotizing': 'idiotic', 'inadvandals': 'vandals', "jackass's": 'jackass', 'jiggabo': 'nigga', 'jizzballs': 'jizz balls', 'jmbass': 'dumbass', 'lejittament': 'legitimate', "m'igger": 'nigger', "m'iggers": 'niggers', 'motherfacking': 'motherfucker', 'motherfuckenkiwi': 'motherfucker', 'muthafuggas': 'niggas', 'nazisms': 'nazis', 'netsnipenigger': 'nigger', 'niggercock': 'nigger', 'niggerspic': 'nigger', 'nignog': 'nigga', 'niqqass': 'niggas', "non-nazi's": 'not a nazi', 'panamanians': 'mexicans', 'pedidiots': 'idiots', 'picohitlers': 'hitler', 'pidiots': 'idiots', 'poopia': 'poop', 'poopsies': 'poop', 'presumingly': 'obviously', 'propagandaanddisinformation': 'propaganda and disinformation', 'propagandaministerium': 'propaganda', 'puertoricans': 'mexicans', 'puertorricans': 'mexicans', 'pussiest': 'pussies', 'pussyitis': 'pussy', 'rayaridiculous': 'ridiculous', 'redfascists': 'fascists', 'retardzzzuuufff': 'retard', "revertin'im": 'reverting', 'scumstreona': 'scums', 'southamericans': 'mexicans', 'strasserism': 'fascism', 'stuptarded': 'retarded', "t'nonsense": 'nonsense', "threatt's": 'threat', 'titoists': 'communists', 'twatbags': 'douchebags', 'youbollocks': 'you bollocks'}
acronym_words = {} #{"btw":"by the way", "yo": "you", "u": "you", "r": "are", "ur": "your", "ima": "i am going to", "imma": "i am going to", "i'ma":"i am going to", "cos":"because", "coz":"because", "stfu": "shut the fuck up", "wat": "what"}
```

```python
# cell [5] of 15
cont_patterns = [
    (r'(W|w)on\'t', r'will not'),
    (r'(C|c)an\'t', r'can not'),
    (r'(I|i)\'m', r'i am'),
    (r'(A|a)in\'t', r'is not'),
    (r'(\w+)\'ll', r'\g<1> will'),
    (r'(\w+)n\'t', r'\g<1> not'),
    (r'(\w+)\'ve', r'\g<1> have'),
    (r'(\w+)\'s', r'\g<1> is'),
    (r'(\w+)\'re', r'\g<1> are'),
    (r'(\w+)\'d', r'\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

def split_word(word, toxic_words):
    if word == "":
        return ""
    
    lower = word.lower()
    for toxic_word in toxic_words:
        start = lower.find(toxic_word)
        if start >= 0:
            end = start + len(toxic_word)
            result = " ".join([word[0:start], word[start:end], split_word(word[end:], toxic_words)])
            return result.replace("  ", " ").strip()
    return word

tknzr = TweetTokenizer(strip_handles=False, reduce_len=True)
def word_tokenize(sentence):
    sentence = sentence.replace("$", "s")
    sentence = sentence.replace("@", "a")    
    sentence = sentence.replace("!", " ! ")
    sentence = sentence.replace("?", " ? ")
    
    return tknzr.tokenize(sentence)

def replace_url(word):
    if "http://" in word or "www." in word or "https://" in word or "wikipedia.org" in word:
        return ""
    return word

def normalize_by_dictionary(normalized_word, dictionary):
    result = []
    for word in normalized_word.split():
        if word == word.upper():
            if word.lower() in dictionary:
                result.append(dictionary[word.lower()].upper())
            else:
                result.append(word)
        else:
            if word.lower() in dictionary:
                result.append(dictionary[word.lower()])
            else:
                result.append(word)
    
    return " ".join(result)
```

```python
# cell [6] of 15

from spacy.symbols import nsubj, VERB, dobj
import spacy
nlp = spacy.load('en')

def normalize_comment(comment):
    comment = unidecode(comment)
    comment = comment[:length_threshold]
    
    normalized_words = []
    
    for w in astericks_words:
        if w[0] in comment:
            comment = comment.replace(w[0], w[1])
        if w[0].upper() in comment:
            comment = comment.replace(w[0].upper(), w[1].upper())
    
    for word in word_tokenize(comment):
        #for (pattern, repl) in patterns:
        #    word = re.sub(pattern, repl, word)

        if word == "." or word == ",":
            normalized_words.append(word)
            continue
        
        word = replace_url(word)
        if word.count(".") == 1:
            word = word.replace(".", " ")
        filtered_word = "".join([x for x in word if x in valid_set])
                    
        #Kind of hack: for every word check if it has a toxic word as a part of it
        #If so, split this word by swear and non-swear part.
        normalized_word = split_word(filtered_word, toxic_words)
        normalized_word = normalize_by_dictionary(normalized_word, hyphens_dict)
        normalized_word = normalize_by_dictionary(normalized_word, merged_dict)
        normalized_word = normalize_by_dictionary(normalized_word, misspellings_dict)
        normalized_word = normalize_by_dictionary(normalized_word, fasttext_misspelings)
        normalized_word = normalize_by_dictionary(normalized_word, acronym_words)

        normalized_words.append(normalized_word)
        
    normalized_comment = " ".join(normalized_words)
    
    result = []
    for word in normalized_comment.split():
        if word.upper() == word:
            result.append(word)
        else:
            result.append(word.lower())
    
    #apparently, people on wikipedia love to talk about sockpuppets :-)
    result = " ".join(result)
    if "sock puppet" in result:
        result = result.replace("sock puppet", "sockpuppet")
    
    if "SOCK PUPPET" in result:
        result = result.replace("SOCK PUPPET", "SOCKPUPPET")
    
    return result


```

```python
# cell [12] of 15
def build_model(embedding_matrix):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(6, activation='sigmoid')(hidden)
    
    
    model = Model(inputs=words, outputs=result)
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model
```

```python
# cell [13] of 15
EPOCHS = 5
SEEDS = 10

pred = 0

for ii in range(SEEDS):
    model = build_model(embedding_matrix)
    for global_epoch in range(EPOCHS):
        print(global_epoch)
        model.fit(
                    x_train,
                    y_train,
                    validation_data = (x_valid, y_valid),
                    batch_size=128,
                    epochs=1,
                    verbose=2,
                    callbacks=[
                        LearningRateScheduler(lambda _: 1e-3 * (0.5 ** global_epoch))
                    ]
                )
        val_preds = model.predict(x_valid)
        AUC = 0
        for i in range(6):
             AUC += roc_auc_score(y_valid[:,i], val_preds[:,i])/6.
        print(AUC)

    pred += model.predict(x_test, batch_size = 1024, verbose = 1)/SEEDS
    np.save('pred', pred)
    model.save_weights('model_weights_'+str(ii)+'.h5')
    os.system('gzip '+'model_weights_'+str(ii)+'.h5')
```


## new-york-city-taxi-fare-prediction — nb 14124715  (score=2085.39277, 45 cells)

*selected 8/45 cells; mean position 0.60 (0=start,1=end); summary ~338 tok vs selected cells ~1123 tok*


**Summary (summary-alone condition):**

This notebook tackles taxi fare prediction through systematic geographic and temporal feature engineering combined with linear regression. The approach extracts absolute longitude and latitude differences between pickup and dropoff locations, filters outliers (trips with coordinate deltas exceeding 5 degrees), and engineers derived distance metrics using the Haversine formula to compute great-circle distances in miles, including trip distance and distances from both endpoints to a fixed airport reference point (40.6413111, -73.7781391). Temporal features are extracted by parsing the pickup_datetime column to derive weekday information (one-hot encoded into seven binary features) and pickup time encoded as a four-digit numeric value (HHMM format converted to hour*100 + minute). Preprocessing includes null value removal, outlier filtering on coordinate differences, and variance-based normalization applied asymmetrically to latitude and longitude difference features (subtracting mean, dividing by variance). The model uses scikit-learn's LinearRegression with normalization enabled, trained on 99% of the data with a fixed random seed (80) for reproducibility, and predictions are rounded to two decimal places. The validation strategy employs a simple 99%/1% train-test split rather than cross-validation, using R² score as the target metric.


**Why cells add value (LLM):** These cells encode the exact feature engineering operations (Haversine distance calculations with radius 6373 km and 0.621 conversion to miles, weekday extraction via pd.Timestamp.weekday(), pickup time encoding as hour*100+minute, airport reference coordinates 40.6413111/-73.7781391), non-standard normalization approach, and train-test split/model configuration that are not fully conveyable in prose alone.


**Selected cells (indices [7, 14, 23, 26, 29, 34, 38, 39]):**


```python
# cell [7] of 45
train_data['Difference_longitude']=np.abs(np.asarray(train_data['pickup_longitude']-train_data['dropoff_longitude']))
train_data['Difference_latitude']=np.abs(np.asarray(train_data['pickup_latitude']-train_data['dropoff_latitude']))


test_data['Difference_longitude']=np.abs(np.asarray(test_data['pickup_longitude']-test_data['dropoff_longitude']))
test_data['Difference_latitude']=np.abs(np.asarray(test_data['pickup_latitude']-test_data['dropoff_latitude']))
```

```python
# cell [14] of 45
ls1=list(train_data['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][:-4:]
    ls1[i]=pd.Timestamp(ls1[i])
    ls1[i]=ls1[i].weekday()
train_data['Weekday']=ls1


ls1=list(test_data['pickup_datetime'])
for i in range(len(ls1)):
    ls1[i]=ls1[i][:-4:]
    ls1[i]=pd.Timestamp(ls1[i])
    ls1[i]=ls1[i].weekday()
test_data['Weekday']=ls1
```

```python
# cell [23] of 45
ls1=list(train_data['pickuptime'])
for i in range(len(ls1)):
    z=ls1[i].split(':')
    ls1[i]=int(z[0])*100+int(z[1])
train_data['pickuptime']=ls1


ls1=list(test_data['pickuptime'])
for i in range(len(ls1)):
    z=ls1[i].split(':')
    ls1[i]=int(z[0])*100+int(z[1])
test_data['pickuptime']=ls1
```

```python
# cell [26] of 45
R = 6373.0
lat1 =np.asarray(np.radians(train_data['pickup_latitude']))
lon1 = np.asarray(np.radians(train_data['pickup_longitude']))
lat2 = np.asarray(np.radians(train_data['dropoff_latitude']))
lon2 = np.asarray(np.radians(train_data['dropoff_longitude']))

dlon = lon2 - lon1
dlat = lat2 - lat1
ls1=[] 
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c

    
train_data['Distance']=np.asarray(distance)*0.621



lat1 =np.asarray(np.radians(test_data['pickup_latitude']))
lon1 = np.asarray(np.radians(test_data['pickup_longitude']))
lat2 = np.asarray(np.radians(test_data['dropoff_latitude']))
lon2 = np.asarray(np.radians(test_data['dropoff_longitude']))

dlon = lon2 - lon1
dlat = lat2 - lat1
 
a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/ 2)**2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data['Distance']=np.asarray(distance)*0.621
```

```python
# cell [29] of 45
R = 6373.0
lat1 =np.asarray(np.radians(train_data['pickup_latitude']))
lon1 = np.asarray(np.radians(train_data['pickup_longitude']))
lat2 = np.asarray(np.radians(train_data['dropoff_latitude']))
lon2 = np.asarray(np.radians(train_data['dropoff_longitude']))

lat3=np.zeros(len(train_data))+np.radians(40.6413111)
lon3=np.zeros(len(train_data))+np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff=lon3 -lon2
d_lat_dropoff=lat3-lat2
a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_data['Pickup_Distance_airport']=np.asarray(distance1)*0.621

a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

    
train_data['Dropoff_Distance_airport']=np.asarray(distance2)*0.621



lat1 =np.asarray(np.radians(test_data['pickup_latitude']))
lon1 = np.asarray(np.radians(test_data['pickup_longitude']))
lat2 = np.asarray(np.radians(test_data['dropoff_latitude']))
lon2 = np.asarray(np.radians(test_data['dropoff_longitude']))

lat3=np.zeros(len(test_data))+np.radians(40.6413111)
lon3=np.zeros(len(test_data))+np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff=lon3 -lon2
d_lat_dropoff=lat3-lat2
a1 = np.sin(dlat_pickup/2)**2 + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup/ 2)**2
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data['Pickup_Distance_airport']=np.asarray(distance1)*0.621

a2=np.sin(d_lat_dropoff/2)**2 + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff/ 2)**2
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

    
test_data['Dropoff_Distance_airport']=np.asarray(distance2)*0.621
```

```python
# cell [34] of 45
train_data['Difference_latitude']=np.abs(train_data['Difference_latitude']-np.mean(train_data['Difference_latitude']))
train_data['Difference_latitude']=train_data['Difference_latitude']/np.var(train_data['Difference_latitude'])
```

```python
# cell [38] of 45
from sklearn.model_selection import train_test_split
X=train_data.drop(['key','fare_amount'],axis=1)
y=train_data['fare_amount']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.01,random_state=80)
```

```python
# cell [39] of 45
from sklearn.linear_model import LinearRegression
lr=LinearRegression(normalize=True)
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))
```


## nomad2018-predict-transparent-conductors — nb 2037642  (score=0.064, 42 cells)

*selected 7/42 cells; mean position 0.43 (0=start,1=end); summary ~324 tok vs selected cells ~1273 tok*


**Summary (summary-alone condition):**

This notebook tackles multi-target regression (formation energy and bandgap energy prediction) on materials science data using ensemble tree-based methods with extensive hyperparameter tuning. The feature engineering approach derives count features from atomic percentages (e.g., al = number_of_total_atoms * percent_atom_al) and applies one-hot encoding to categorical spacegroup. The modeling strategy employs LightGBM as the primary estimator, carefully tuned through sequential GridSearchCV rounds targeting specific hyperparameters: learning_rate (0.1), tree structure via max_depth (7) and num_leaves (60), regularization through L1 (reg_alpha ~0.042) and tree count reduction (bagging_fraction and feature_fraction both ~0.4), and child node constraints (min_child_samples=20, min_child_weight=0.5) to prevent overfitting. Validation uses 80/20 train-test split with StandardScaler normalization and RMSE on log-transformed targets to handle scale differences across outputs; dual-model submission trains separate LightGBM and SVR (kernel='rbf', C=80, gamma=0.00043) regressors for the two targets. The distinctive regularization strategy combines aggressive feature/bagging subsampling with explicit tree growth constraints, calibrated iteratively to reduce test RMSE from ~0.10 to ~0.098.


**Why cells add value (LLM):** These cells contain the concrete feature engineering pipeline (cell 5), validation framework with log-transform RMSE metric (cell 8), LightGBM cross-validation setup for early stopping (cell 17), and the final tuned hyperparameter configurations across iterations (cells 19, 21, 25, 28) that are directly transferable to similar regression problems.


**Selected cells (indices [5, 8, 17, 19, 21, 25, 28]):**


```python
# cell [5] of 42
def feature_engine(df):
    df['al'] = df['number_of_total_atoms'] * df['percent_atom_al']
    df['ga'] = df['number_of_total_atoms'] * df['percent_atom_ga']    
    df['in'] = df['number_of_total_atoms'] * df['percent_atom_in']    
    df['all'] = df['al'] + df['ga'] + df['in']
    df['spacegroup'] = df['spacegroup'].astype('object')
    
    try:
        df.drop(['formation_energy_ev_natom','bandgap_energy_ev'],axis=1,inplace=True)
    except:
        pass
    return pd.get_dummies(df)

Y1 = df_train['formation_energy_ev_natom']
Y2 = df_train['bandgap_energy_ev']
# Y1 = np.log1p(Y1)
# Y2 = np.log1p(Y2)
df_train = feature_engine(df_train)
df_test = feature_engine(df_test)
df_train.head()
```

```python
# cell [8] of 42
x_train,x_test,y_train,y_test = train_test_split(df_train,Y2,test_size=0.2,random_state=7)
rs = StandardScaler()
x_train = rs.fit_transform(x_train)
x_test = rs.transform(x_test)

def validate_data(model,x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test):
    model.fit(x_train,y_train)
    ss = StandardScaler()
#     x_train = ss.fit_transform(x_train)
#     x_test = ss.transform(x_test)
    y_pred_ = model.predict(x_train)
    y_pred = model.predict(x_test)
    print('train:\n {}'.format(np.sqrt(MSE(np.log1p(y_train),np.log1p(y_pred_)))))
#     print(y_pred)
    print('test :\n {}'.format(np.sqrt(MSE(np.log1p(y_test),np.log1p(y_pred)))))

    
def make_sub(model1,model2):
    model1.fit(df_train,Y1)
    model2.fit(df_train,Y2)
    y_pred_ = model1.predict(df_test)
    y_pred = model2.predict(df_test)
    sub['formation_energy_ev_natom'] = y_pred_
    sub['bandgap_energy_ev'] = y_pred
    sub.to_csv('sub.csv',index=False)
    print('submit finished')

```

```python
# cell [17] of 42
from lightgbm import LGBMRegressor
import lightgbm as lgb


def model_fit_lgb(model,model_params,x_train,y_train,early_stop_rounds=5):
    model_train = lgb.Dataset(x_train,y_train)
    print('cving...')
    cv_result = lgb.cv(model_params,
                       model_train,
                       early_stopping_rounds=early_stop_rounds,
                       nfold=50,
                       stratified=False,#回归问题加stratified=False！
                       shuffle=True,
                       num_boost_round=5000,
                       seed=0,
                       metrics='rmse')
    
    print('cv finished.')
    n_estimators = len(cv_result['rmse-mean'])#这里注意values的长度！
    print(n_estimators)
    model.set_params(n_estimators=n_estimators)   
    
'''
learning_rate=0.1,max_depth=6,subsample=0.8,subsample_freq=1,colsample_bytree=0.8，n_estimators=100
train:
 0.06019847356156149
test :
 0.10068118284715544
 
'''
lgb_params ={
    'learning_rate':0.1,'bagging_fraction':0.8,'feature_fraction':0.8,
    'num_leave':50, #加入num_leave防止过拟合
    'metrics':'rmse','bagging_freq':10
}

lgb1 = LGBMRegressor(**lgb_params)

model_fit_lgb(lgb1,lgb_params,x_train,y_train)
# model_fit(lgb_params,lgb1,x_train,y_train)
validate_data(lgb1)
'''
 train:
 0.06290452900706987
test :
 0.10081715525549317
'''
```

```python
# cell [19] of 42
lgb2 = LGBMRegressor(n_estimators=61,learning_rate=0.1,bagging_fraction=0.8,feature_fraction=0.8,
                    max_depth=7,
                    num_leave=60, #加入num_leave防止过拟合
                    metrics='rmse',bagging_freq=10)
validate_data(lgb2)
'''
{'max_depth': 7, 'num_leaves': 60}
train:
 0.06499574811552297
test :
 0.09924309162427868
'''
```

```python
# cell [21] of 42
# print(grid.best_params_)
lgb3 = LGBMRegressor(n_estimators=61,learning_rate=0.1,bagging_fraction=0.8,feature_fraction=0.8,
                    max_depth=7,
                    num_leave=60, #加入num_leave防止过拟合
                    metrics='rmse',bagging_freq=10,
                    min_child_samples=20,
                    min_child_weight=0.5,
                    )

'''
{'min_child_samples': 20, 'min_child_weight': 0.0001}
train:
 0.06499574811552297
test :
 0.09924309162427868
'''
validate_data(lgb3)
```

```python
# cell [25] of 42
lgb5 = LGBMRegressor(n_estimators=61,learning_rate=0.1,bagging_fraction=0.4,feature_fraction=0.4555555,
                    max_depth=7,
                    num_leave=60, #加入num_leave防止过拟合
                    metrics='rmse',bagging_freq=10,
                    min_child_samples=20,
                    min_child_weight=0.5,
                    reg_alpha= 0.04204081632653062,
                    reg_gamma=0,
                    )
# print(grid.best_score_)
# print(grid.best_params_)
# validate_data(lgb5)

'''
{'reg_alpha': 0.04140816326530612, 'reg_gamma': 0}
train:
 0.0744088692165038
test :
 0.09802256997806044
 
 {'reg_alpha': 0.04204081632653062, 'reg_gamma': 0}
train:
 0.07440965133014965
test :
 0.09802215359344574
 
'''
```

```python
# cell [28] of 42
params = dict(n_estimators=61,learning_rate=0.1,bagging_fraction=0.4,feature_fraction=0.4555555,
                    max_depth=7,
                    num_leave=60, #加入num_leave防止过拟合
                    metrics='rmse',bagging_freq=10,
                    min_child_samples=20,
                    min_child_weight=0.5,
                    reg_alpha= 0.04204081632653062,
                    reg_gamma=0,)
model_fit_lgb(lgb5,params,x_train,y_train)
make_sub(SVR(kernel='rbf',**{'C': 80.0, 'gamma': 0.00043333333333333337}),lgb5)
```


## dogs-vs-cats-redux-kernels-edition — nb 164050  (score=17.269779999999994, 24 cells)

*selected 5/24 cells; mean position 0.50 (0=start,1=end); summary ~286 tok vs selected cells ~640 tok*


**Summary (summary-alone condition):**

This notebook trains a convolutional neural network for binary image classification (cats vs. dogs). The approach uses data stratification with fixed train/validation/test splits (1000/100 training examples per class), resizes all images to 64×64 pixels, and normalizes pixel values to [0,1] by dividing by 255. The model architecture employs two convolutional blocks—the first with 30 filters of kernel size 5×5 followed by 2×2 max pooling, the second with 15 filters of kernel size 3×3 and max pooling—then dropout (rate 0.2) for regularization, flattening, and a dense stack (128→50 units with ReLU, final softmax output). Training uses categorical crossentropy loss, Adam optimizer, batch size 200, and 10 epochs, with validation-set evaluation for error reporting. The key preprocessing transforms are cubic interpolation resizing via OpenCV and grayscale conversion for training (though color images X_train are used for the model), and labels are one-hot encoded via Keras utilities. The validation strategy evaluates on a held-out 100-per-class validation set during training and reports final classification error rate as a percentage.


**Why cells add value (LLM):** Cell 4 shows the exact data partitioning and shuffling scheme; cells 6 and 14 demonstrate the complete image preprocessing (resize, grayscale conversion, normalization); cell 16 encodes the full CNN architecture with filter counts, kernel sizes, dropout rate, and dense layer dimensions; cell 18 specifies training hyperparameters (batch size, epochs, optimizer, loss function) and validation protocol.


**Selected cells (indices [4, 6, 14, 16, 18]):**


```python
# cell [4] of 24
#only taking a subset (less accuracy but faster training)
train_dog = images_dog[:1000]
train_cat = images_cat[:1000]
valid_dog = images_dog[1000:1100]
valid_cat = images_cat[1000:1100]

train_list = train_dog + train_cat
valid_list = valid_dog + valid_cat
test_list  = images_test[0:]

shuffle(train_list)

train = np.ndarray(shape=(len(train_list),ROWS, COLS))
train_color = np.ndarray(shape=(len(train_list), ROWS, COLS, CHANNELS), dtype=np.uint8)
test = np.ndarray(shape=(len(test_list),ROWS, COLS))
test_color = np.ndarray(shape=(len(images_test), ROWS, COLS, CHANNELS), dtype=np.uint8)
valid = np.ndarray(shape=(len(valid_list), ROWS, COLS))
valid_color = np.ndarray(shape=(len(valid_list), ROWS, COLS, CHANNELS), dtype=np.uint8)
```

```python
# cell [6] of 24
labels = np.ndarray(len(train_list))

for i, img_path in enumerate(train_list):
    img_color = cv2.imread(os.path.join(train_path, img_path), 1)
    img_color = cv2.resize(img_color, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    train[i] = img
    train_color[i] = img_color
   
    if "dog" in img_path:
        labels[i] = 0
    else:
        labels[i] = 1
```

```python
# cell [14] of 24
from keras.utils import np_utils

X_train = train_color / 255
X_valid = valid_color / 255
X_test  = test_color  / 255
# one hot encode outputs
y_train = np_utils.to_categorical(labels)
y_valid = np_utils.to_categorical(valid_labels)
num_classes = y_valid.shape[1]
```

```python
# cell [16] of 24
def larger_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(64, 64, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
```

```python
# cell [18] of 24
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_valid, y_valid, verbose=0)
print("Classification Error: %.2f%%" % (100-scores[1]*100))
```


## aerial-cactus-identification — nb 5144562  (score=1.0, 18 cells)

*selected 2/18 cells; mean position 0.53 (0=start,1=end); summary ~371 tok vs selected cells ~1050 tok*


**Summary (summary-alone condition):**

This notebook addresses binary image classification (cactus presence detection) using a custom convolutional neural network trained with stratified k-fold cross-validation and class-weighted loss to handle imbalance (class 0:1 ratio ~3.01:1). The architecture comprises three blocks of three consecutive Conv2D layers (64, 128, and 256 filters respectively) with ReLU activation and BatchNormalization (scale=False), each block followed by MaxPooling2D and Dropout(0.5); a GlobalAveragePooling2D layer reduces spatial dimensions before two dense layers (256 units with ReLU and Dropout, then 1 unit with sigmoid) for binary output. Images are normalized to [0,1] range and augmented on-the-fly using ImageDataGenerator with horizontal flipping and 0.1 height shift. The validation strategy employs 10-fold StratifiedKFold with out-of-fold predictions aggregated for robustness; out-of-fold (OOF) predictions on validation splits and test predictions are averaged across folds. Training uses Adam optimizer with binary crossentropy loss, batch size 64, up to 128 epochs, with callbacks including ModelCheckpoint (best validation loss), ReduceLROnPlateau (factor 0.7, patience 5), and EarlyStopping (patience 15). Class weights are computed inversely proportional to class frequency to penalize misclassification of the minority class. Predictions are clipped to [0,1] range before submission; ambiguous samples (predictions in 0.2–0.8 range) are identified for uncertainty quantification.


**Why cells add value (LLM):** Cell 8 encodes the exact CNN architecture with layer-specific configurations (filter counts, kernel sizes, padding, BatchNorm scale setting, dropout rates); Cell 10 implements the complete stratified k-fold training loop with class weight computation, data augmentation parameters, callback configuration, and OOF/test prediction aggregation strategy.


**Selected cells (indices [8, 10]):**


```python
# cell [8] of 18
def build_model(input_shape):
    model =Sequential()
    model.add(Conv2D(64,(3,3),padding='same',input_shape=(input_shape)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False))    
    model.add(Conv2D(64,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False))        
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    
    model.add(Conv2D(128,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(128,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False)) 
    model.add(Conv2D(128,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False))        
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(256,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False))
    model.add(Conv2D(256,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False)) 
    model.add(Conv2D(256,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False))          
    
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256))
    model.add(Activation('relu'))    
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile('adam',loss='binary_crossentropy',metrics=['accuracy']) 
#     model.add(Dense(2))
#     model.add(Activation('softmax'))    
#     model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy']) 
    
    return model 
```

```python
# cell [10] of 18
%%time
import sklearn
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
import keras.backend as K
from sklearn.metrics import *
from keras.preprocessing.image import ImageDataGenerator
histories = []
oof_pred = np.zeros(len(df_train))
sub_pred = np.zeros(len(df_test))

class_weights = {} 
weights = [3.010082493125573,#3.01,
           1.0]
len(df_train[df_train.has_cactus==0]) 
for i in range(2): 
    class_weights[i] = weights[i] 
print('class_weights:',class_weights)

checkpoint_name = '/checkpoint.file'
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
BATCH_SIZE = 64#128
EPOCHS = 128

for fold_id, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    print(f'fold id: {fold_id}')
    X_tr, y_tr = X_train[train_index], y_train[train_index]
    X_val, y_val = X_train[val_index], y_train[val_index]

    callbacks=[
        keras.callbacks.ModelCheckpoint(
            checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, verbose=1,min_delta=0.00005, ),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
    ]   

    datagen = ImageDataGenerator(
#             width_shift_range=0.1,#2,
            height_shift_range=0.1,
            horizontal_flip=True
    )
    K.clear_session()
    model = build_model(X_train.shape[1:])   
    model.summary()    
    
    histories.append(
#         model.fit(X_tr, y_tr, batch_size=16, epochs=128,#64, 
#                   validation_data=(X_val, y_val), 
#                   class_weight=class_weights,verbose=2, callbacks=callbacks)
        model.fit_generator(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=int(np.ceil(len(X_train) / BATCH_SIZE)), validation_data=(X_val, y_val), 
            epochs=EPOCHS, class_weight=class_weights, 
            callbacks=callbacks, verbose=2)
    )
    
    model = load_model(checkpoint_name)
    oof_pred[val_index] = model.predict(X_val).flatten()
    sub_pred += model.predict(X_test).flatten() / skf.n_splits
    print(roc_auc_score(y_val, oof_pred[val_index]))
    plot_roc_auc(y_val, oof_pred[val_index])
    del callbacks
sub_pred = np.clip(sub_pred,0.0,1.0)    
```