# Sentiment Analysis Assignment Documentation

## Introduction

This notebook demonstrates a sentiment analysis task using a dataset of movie reviews. The objective is to classify the reviews as positive or negative using machine learning models. The notebook includes data preprocessing, model training, evaluation, and insights gained from the analysis.


## ## Import Libraries

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
```

## ## Import Data

### Loading the Data

The dataset containing movie reviews and sentiments is loaded into a DataFrame. Here's a preview of the dataset:

```python
df = pd.read_csv("IMDB Dataset.csv")
df.head(10)
```

The dataset consists of 50,000 rows and 2 columns: 'review' and 'sentiment'.


### Data Exploration

The summary statistics of the dataset are displayed to understand the distribution of numerical features. Additionally, the sentiment labels are converted to binary values where 1 represents positive sentiment and 0 represents negative sentiment.

```python
print("Summary statistics of numerical features : \n", df.describe())

print("=======================================================================")

print("\nTotal number of reviews: ",len(df))

print("=======================================================================")

print("\nTotal number of Sentiments: ", len(list(set(df['sentiment']))))

df['sentiment'] = np.where(df['sentiment'] == "positive", 1, 0)
df
```

## # Data Visualization

```python
plt.figure(figsize=(12,5))
# sns.countplot(df['Rating'])
df['sentiment'].value_counts().sort_index().plot(kind='bar',color = 'blue')
plt.title('Distribution of Rating')
plt.grid()
plt.xlabel('Rating')
plt.ylabel('Count')
ax = plt.axes()
ax.set_facecolor("white")
```

## # Data Preparation

```python
df = df.sample(frac=0.1, random_state=0) #uncomment to use full set of data

# Drop missing values
df.dropna(inplace=True)

df
```

## # Train Test Split

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], \
                                                    test_size=0.1, random_state=0)

print('Load %d training examples and %d validation examples. \n' %(X_train.shape[0],X_test.shape[0]))
print('Show a review in the training set : \n', X_train.iloc[10])
X_train,y_train
```

## # Bag of Words
<br>

**Step 1 : Preprocess raw reviews to cleaned reviews**

**Step 2 : Create BoW using CountVectorizer / Tfidfvectorizer in sklearn**

**Step 3 : Transform review text to numerical representations (feature vectors)**

**Step 4 : Fit feature vectors to supervised learning algorithm (eg. Naive Bayes, Logistic regression, etc.)**

**Step 5 : Improve the model performance by GridSearch**

# Text Preprocessing
<br>

**Step 1 : remove html tags using BeautifulSoup**

**Step 2 : remove non-character such as digits and symbols**

**Step 3 : convert to lower case**

**Step 4 : remove stop words such as "the" and "and" if needed**

**Step 5 : convert to root words by stemming if needed**

```python
def cleanText(raw_text, remove_stopwords=False, stemming=False, split_text=False, \
             ):
    '''
    Convert a raw review to a cleaned review
    '''
    text = BeautifulSoup(raw_text, 'html.parser').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split() 
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        
    if stemming==True:

        stemmer = SnowballStemmer('english') 
        words = [stemmer.stem(w) for w in words]
        
    if split_text==True:
        return (words)
    
    return( " ".join(words))
```

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize, pos_tag
from bs4 import BeautifulSoup 
import logging
from wordcloud import WordCloud
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

X_train_cleaned = []
X_test_cleaned = []

for d in X_train:
    X_train_cleaned.append(cleanText(d))
print('Show a cleaned review in the training set : \n',  X_train_cleaned[10])
    
for d in X_test:
    X_test_cleaned.append(cleanText(d))
```

## ## CountVectorizer with Mulinomial Naive Bayes (Benchmark Model)

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
countVect = CountVectorizer() 
X_train_countVect = countVect.fit_transform(X_train_cleaned)
print("Number of features : %d \n" %len(countVect.get_feature_names())) #6378 
print("Show some feature names : \n", countVect.get_feature_names()[::1000])


# Train MultinomialNB classifier
mnb = MultinomialNB()
mnb.fit(X_train_countVect, y_train)
```

```python
pickle.dump(countVect,open('countVect_imdb.pkl','wb'))
```

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
from sklearn import metrics
from sklearn.metrics import accuracy_score,roc_auc_score
def modelEvaluation(predictions):
    '''
    Print model evaluation to predicted result 
    '''
    print ("\nAccuracy on validation set: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("\nAUC score : {:.4f}".format(roc_auc_score(y_test, predictions)))
    print("\nClassification report : \n", metrics.classification_report(y_test, predictions))
    print("\nConfusion Matrix : \n", metrics.confusion_matrix(y_test, predictions))
```

```python
predictions = mnb.predict(countVect.transform(X_test_cleaned))
modelEvaluation(predictions)
```

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
import pickle
pickle.dump(mnb,open('Naive_Bayes_model_imdb.pkl','wb'))
```

## # TfidfVectorizer with Logistic Regression

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
from sklearn.linear_model import LogisticRegression
tfidf = TfidfVectorizer(min_df=5) #minimum document frequency of 5
X_train_tfidf = tfidf.fit_transform(X_train)
print("Number of features : %d \n" %len(tfidf.get_feature_names())) #1722
print("Show some feature names : \n", tfidf.get_feature_names()[::1000])

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)
```

```python
feature_names = np.array(tfidf.get_feature_names())
sorted_coef_index = lr.coef_[0].argsort()
print('\nTop 10 features with smallest coefficients :\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Top 10 features with largest coefficients : \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))
```

```python
predictions = lr.predict(tfidf.transform(X_test_cleaned))
modelEvaluation(predictions)
```

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
from sklearn.model_selection import  GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
estimators = [("tfidf", TfidfVectorizer()), ("lr", LogisticRegression())]
model = Pipeline(estimators)


params = {"lr__C":[0.1, 1, 10], 
          "tfidf__min_df": [1, 3], 
          "tfidf__max_features": [1000, None], 
          "tfidf__ngram_range": [(1,1), (1,2)], 
          "tfidf__stop_words": [None, "english"]} 

grid = GridSearchCV(estimator=model, param_grid=params, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_cleaned, y_train)
print("The best paramenter set is : \n", grid.best_params_)


# Evaluate on the validaton set
predictions = grid.predict(X_test_cleaned)
modelEvaluation(predictions)
```

## # Word2Vec
<br>

**Step 1 : Parse review text to sentences (Word2Vec model takes a list of sentences as inputs)**

**Step 2 : Create volcabulary list using Word2Vec model.**

**Step 3 : Transform each review into numerical representation by computing average feature vectors of words therein.**

**Step 4 : Fit the average feature vectors to Random Forest Classifier.**

```python
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def parseSent(review, tokenizer, remove_stopwords=False):

    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(cleanText(raw_sentence, remove_stopwords, split_text=True))
    return sentences


# Parse each review in the training set into sentences
sentences = []
for review in X_train_cleaned:
    sentences += parseSent(review, tokenizer,remove_stopwords=False)
    
print('%d parsed sentence in the training set\n'  %len(sentences))
print('Show a parsed sentence in the training set : \n',  sentences[10])
```

## ## Creating Volcabulary List usinhg Word2Vec Model

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
from wordcloud import WordCloud
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
num_features = 300  #embedding dimension                     
min_word_count = 10                
num_workers = 4       
context = 10                                                                                          
downsampling = 1e-3 

print("Training Word2Vec model ...\n")
w2v = Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count,\
                 window = context, sample = downsampling)
w2v.init_sims(replace=True)
w2v.save("w2v_300features_10minwordcounts_10context") #save trained word2vec model

print("Number of words in the vocabulary list : %d \n" %len(w2v.wv.index2word)) #4016 
print("Show first 10 words in the vocalbulary list  vocabulary list: \n", w2v.wv.index2word[0:10])
```

## ## Averaging Feature Vectors

```python
def makeFeatureVec(review, model, num_features):
    '''
    Transform a review to a feature vector by averaging feature vectors of words 
    appeared in that review and in the volcabulary list created
    '''
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word) #index2word is the volcabulary list of the Word2Vec model
    isZeroVec = True
    for word in review:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
            isZeroVec = False
    if isZeroVec == False:
        featureVec = np.divide(featureVec, nwords)
    return featureVec
```

```python
def getAvgFeatureVecs(reviews, model, num_features):
    '''
    Transform all reviews to feature vectors using makeFeatureVec()
    '''
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
        counter = counter + 1
    return reviewFeatureVecs
```

```python
X_train_cleaned = []
for review in X_train:
    X_train_cleaned.append(cleanText(review, remove_stopwords=True, split_text=True))
trainVector = getAvgFeatureVecs(X_train_cleaned, w2v, num_features)
print("Training set : %d feature vectors with %d dimensions" %trainVector.shape)


# Get feature vectors for validation set
X_test_cleaned = []
for review in X_test:
    X_test_cleaned.append(cleanText(review, remove_stopwords=True, split_text=True))
testVector = getAvgFeatureVecs(X_test_cleaned, w2v, num_features)
print("Validation set : %d feature vectors with %d dimensions" %testVector.shape)
```

## # Random Forest Classifer

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(trainVector, y_train)
predictions = rf.predict(testVector)
modelEvaluation(predictions)
```

## ## LSTM
<br>

**Step 1 : Prepare X_train and X_test to 2D tensor.**
    
**Step 2 : Train a simple LSTM (embeddign layer => LSTM layer => dense layer).**
    
**Step 3 : Compile and fit the model using log loss function and ADAM optimizer.**

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.preprocessing.text import Tokenizer
from collections import defaultdict
from keras.layers.convolutional import Convolution1D
from keras import backend as K
from keras.layers.embeddings import Embedding
```

```python
top_words = 40000 
maxlen = 200 
batch_size = 62
nb_classes = 4
nb_epoch = 6


# Vectorize X_train and X_test to 2D tensor
tokenizer = Tokenizer(nb_words=top_words) #only consider top 20000 words in the corpse
tokenizer.fit_on_texts(X_train)
# tokenizer.word_index #access word-to-index dictionary of trained tokenizer

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

X_train_seq = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test_seq = sequence.pad_sequences(sequences_test, maxlen=maxlen)


# one-hot encoding of y_train and y_test
y_train_seq = np_utils.to_categorical(y_train, nb_classes)
y_test_seq = np_utils.to_categorical(y_test, nb_classes)

print('X_train shape:', X_train_seq.shape)
print("========================================")
print('X_test shape:', X_test_seq.shape)
print("========================================")
print('y_train shape:', y_train_seq.shape)
print("========================================")
print('y_test shape:', y_test_seq.shape)
print("========================================")
```

```python
model1 = Sequential()
model1.add(Embedding(top_words, 128, dropout=0.2))
model1.add(LSTM(128, dropout_W=0.2, dropout_U=0.2)) 
model1.add(Dense(nb_classes))
model1.add(Activation('softmax'))
model1.summary()
```

```python
model1.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model1.fit(X_train_seq, y_train_seq, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

# Model evluation
score = model1.evaluate(X_test_seq, y_test_seq, batch_size=batch_size)
print('Test loss : {:.4f}'.format(score[0]))
print('Test accuracy : {:.4f}'.format(score[1]))
```

```python
len(X_train_seq),len(y_train_seq)
```

```python
print("Size of weight matrix in the embedding layer : ", \
      model1.layers[0].get_weights()[0].shape)

# get weight matrix of the hidden layer
print("Size of weight matrix in the hidden layer : ", \
      model1.layers[1].get_weights()[0].shape)

# get weight matrix of the output layer
print("Size of weight matrix in the output layer : ", \
      model1.layers[2].get_weights()[0].shape)
```

### Importing Libraries

The following libraries are imported for data manipulation, visualization, and machine learning tasks:

- `pandas`: Used for data manipulation and analysis.

- `numpy`: Provides support for large multi-dimensional arrays and matrices.

- `matplotlib` and `seaborn`: Used for data visualization.

- `warnings`: To ignore warnings during execution.


```python
import pickle
pickle.dump(model1,open('model1.pkl','wb'))
```

## ## LSTM with Word2Vec Embedding

```python
2v = Word2Vec.load("w2v_300features_10minwordcounts_10context")

embedding_matrix = w2v.wv.syn0 
print("Shape of embedding matrix : ", embedding_matrix.shape)
```

```python
top_words = embedding_matrix.shape[0] #4016 
maxlen = 300 
batch_size = 62
nb_classes = 4
nb_epoch = 7


# Vectorize X_train and X_test to 2D tensor
tokenizer = Tokenizer(nb_words=top_words) #only consider top 20000 words in the corpse
tokenizer.fit_on_texts(X_train)
# tokenizer.word_index #access word-to-index dictionary of trained tokenizer

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

X_train_seq1 = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test_seq1 = sequence.pad_sequences(sequences_test, maxlen=maxlen)


# one-hot encoding of y_train and y_test
y_train_seq1 = np_utils.to_categorical(y_train, nb_classes)
y_test_seq1 = np_utils.to_categorical(y_test, nb_classes)

print('X_train shape:', X_train_seq1.shape)
print("========================================")
print('X_test shape:', X_test_seq1.shape)
print("========================================")
print('y_train shape:', y_train_seq1.shape)
print("========================================")
print('y_test shape:', y_test_seq1.shape)
print("========================================")
```

```python
len(X_train_seq1),len(y_train_seq1)
```

```python
embedding_layer = Embedding(embedding_matrix.shape[0], #4016
                            embedding_matrix.shape[1], #300
                            weights=[embedding_matrix])

model2 = Sequential()
model2.add(embedding_layer)
model2.add(LSTM(128, dropout_W=0.2, dropout_U=0.2)) 
model2.add(Dense(nb_classes))
model2.add(Activation('softmax'))
model2.summary()
```

```python
model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model2.fit(X_train_seq1, y_train_seq1, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

# Model evaluation
score = model2.evaluate(X_test_seq1, y_test_seq1, batch_size=batch_size)
print('Test loss : {:.4f}'.format(score[0]))
print('Test accuracy : {:.4f}'.format(score[1]))
```

```python
print("Size of weight matrix in the embedding layer : ", \
      model2.layers[0].get_weights()[0].shape) 

print("Size of weight matrix in the hidden layer : ", \
      model2.layers[1].get_weights()[0].shape) 

print("Size of weight matrix in the output layer : ", \
      model2.layers[2].get_weights()[0].shape)
```

## Conclusion

This notebook successfully demonstrates the process of sentiment analysis on a movie review dataset. The key steps include data loading, preprocessing, model building, and evaluation. The results indicate that the chosen machine learning models can effectively classify the sentiments of movie reviews. Future work could explore more advanced techniques or model optimization to further improve accuracy.
