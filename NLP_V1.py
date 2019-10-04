# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_excel("C:\ML&AI\TextClassification\RAW DATA(Main Update)#2 (2019-07-02).xlsx", sheet_name="All",skiprows=5)
data.columns.tolist()

#countries_english=['RLTD','RAME','RIND','RPAC','RSAF','RKOR']

filtered_data = data[(data["Filtering\n(Credit/Age/Class)"]=="No") & (data["Work Type"]=="Visit") & (data["Case Type (Related Case)"]=="Complaint") & (data["Country"].str.contains("RLTD|RAME|RIND|RPAC|RSAF|RKOR"))]
                 
filtered_data.drop_duplicates(subset="SWO Number")

#reset all indexes as we dropped rows above
filtered_data = filtered_data.reset_index(drop=True)

#concatenate 3 columns of text
filtered_data['Text']= filtered_data['Case Description'] +' ' + filtered_data['Investigation'] + ' ' + filtered_data['Corrective Action']

# final columns for classification
dataframe=filtered_data.loc[:,['SWO Number','Text','RMED FaultCode L1(New)']]
#,'RMED FaultCode L2(New)', 'RMED FaultCode L3(New)', 'RMED FaultCode L4(New)',]

#If the count of target is less than 5%, combine to others type
classif = dataframe['RMED FaultCode L1(New)'].value_counts(normalize=True)
idx = classif[classif.lt(0.05)].index
dataframe.loc[dataframe['RMED FaultCode L1(New)'].isin(idx),'RMED FaultCode L1(New)'] = 'Others'

#prediction data -  save the rows where tgt is blank from dataframe
pred_X_Y = dataframe.loc[dataframe['RMED FaultCode L1(New)'].isnull(), ['SWO Number','Text','RMED FaultCode L1(New)']]

#get data after filtering where target is blank and text is blank
dataframe = dataframe.loc[dataframe['RMED FaultCode L1(New)'].notnull(), ['SWO Number','Text','RMED FaultCode L1(New)']]
dataframe = dataframe.loc[dataframe['Text'].notnull(), ['SWO Number','Text','RMED FaultCode L1(New)']]
dataframe = dataframe.reset_index(drop=True)

dataframe['Prediction'].value_counts()


######################   down sampling of majority class dint help in improving accuracy  ####################
#df_majority = dataframe[dataframe['RMED FaultCode L1(New)']=='Defect']
#df_majority_downsampled = resample(df_majority, 
#                                 replace=False,    # sample without replacement
#                                 n_samples=300,     # to match minority class
#                                 random_state=123) # reproducible results
#
#dataframe = dataframe[dataframe['RMED FaultCode L1(New)']!='Defect']
#dataframe = pd.concat([dataframe,df_majority_downsampled])

##below is without stemming and before upsampling
#model_name
#LinearSVC                 0.575613
#LogisticRegression        0.573000
#MultinomialNB             0.534379
#RandomForestClassifier    0.430786

#below is with stemming and before upsampling
#model_name
#LinearSVC                 0.653144
#LogisticRegression        0.579271
#MultinomialNB             0.491997
#RandomForestClassifier    0.442122
##################################################################################################################

######################        up sampling of minority class        ####################
from sklearn.utils import resample
df_minority = dataframe[dataframe['RMED FaultCode L1(New)']!='Defect']
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample without replacement
                                 n_samples=4000,     # to match minority class
                                 random_state=123) # reproducible results
df_minority_upsampled['RMED FaultCode L1(New)'].value_counts()

dataframe = dataframe[dataframe['RMED FaultCode L1(New)']=='Defect']
dataframe = pd.concat([dataframe,df_minority_upsampled])

dataframe['RMED FaultCode L1(New)'].value_counts()

#below is the check the imbalance classes
fig = plt.figure(figsize=(8,6))
dataframe.groupby('RMED FaultCode L1(New)').Text.count().plot.bar(ylim=0)
plt.show()


#as translation isnot working we can start to consider only english text
################################################################################

#Google Trans :(
#https://py-googletrans.readthedocs.io/en/latest/
from googletrans import Translator
import time
translator = Translator()   
df=filtered_data.loc[:1000,['SWO Number','Case Description']]
for i in range (0,len(df)):
    translator = Translator()
    time.sleep(20)
    test = df['Case Description'].apply(translator.translate, dest='en').apply(getattr, args=('text',))


fileids = newcorpus.fileids
for f in fileids:
    p = newcorpus.raw(f) 
    p = str(p[:15000])
    translated_text = translator.translate(p)
    print(translated_text)
    sleep(10)


from googletrans import Translator
translator = Translator()
text=translator.translate('안녕하세요.',dest='en').text
#----------------------------------------------
from nltk import sent_tokenize
from nltk import word_tokenize
import nltk
nltk.download('punkt')
from googletrans import Translator

import time

df=filtered_data.loc[0:500,['SWO Number','Case Description']]
for i in range (0,len(df)):
    token = sent_tokenize(str(df.loc[i,['Case Description']]))
    for tt in token:
        translatedText = translator.translate(tt, dest="en")
        data=translatedText.text
    df.loc[i,['Case Description']]=data
    translator = Translator()
    time.sleep(0.4)
    
#below to be used
df=dataframe.iloc[0:500,1]
for i in range (0,len(df)):
    token = sent_tokenize(str(df.loc[i]))
    for tt in token:
        translatedText = translator.translate(tt, dest="en")
        data=translatedText.text
    df.loc[i,['Case Description']]=data
    translator = Translator()
    time.sleep(0.4)
################################################################################

#Multi class NLP Classification
    
#create a column where each class has a unique id called category id
dataframe['category_id'] = dataframe['RMED FaultCode L1(New)'].factorize()[0]
category_id_dataframe = dataframe[['RMED FaultCode L1(New)', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_dataframe.values)
id_to_category = dict(category_id_dataframe[['category_id', 'RMED FaultCode L1(New)']].values)
dataframe.head()

#https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html
#The difference is that stem might not be an actual word whereas, lemma is an actual language word. 
#in lemma, you used WordNet corpus & corpus for stop words to produce lemma which makes it slower than stemming.
import re
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter_stemmer.stem(word) for word in words]
    return words


#below is with stemming(25K words) and after upsampling
#LinearSVC                 0.938311
#LogisticRegression        0.881233
#MultinomialNB             0.802324
#RandomForestClassifier    0.525123

##################################################
#from nltk.corpus import wordnet as wn
#def lemmatize(self, token, tag):
#        tag = {
#            'N': wn.NOUN,
#            'V': wn.VERB,
#            'R': wn.ADV,
#            'J': wn.ADJ
#        }.get(tag[0], wn.NOUN)
#
#        return self.lemmatizer.lemmatize(token, tag)
##################################################




from sklearn.feature_extraction.text import TfidfVectorizer
#min_df = When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
#norm='l2' = The cosine similarity between two vectors is their dot product when l2 norm has been applied.
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english',token_pattern=r'(?u)\b[A-Za-z]+\b', tokenizer=stemming_tokenizer)
features = tfidf.fit_transform(dataframe.Text).toarray()
labels = dataframe.category_id
features.shape

###################################################################################
##########Below is to check the most correlated unigrams and bigrams in a tgt category##########
#apart from tfidf we can find the terms that are the most correlated with each of the target class using chi2
#Compute chi-squared stats between each non-negative feature and class.
from sklearn.feature_selection import chi2
N = 2
for Text, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  #print("# :".format(dataframe['RMED FaultCode L1(New)']))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
###################################################################################
############Naive Bayes only ###################
#To train model - we first transformed the “Text” into a vector of numbers. We checked upon vector representations like TF-IDF weighted vectors.
#Next we can train supervised classifiers to train unseen “Text” and predict the target class on which they fall.

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X = dataframe['Text']
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(X)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, dataframe['RMED FaultCode L1(New)'],test_size = 0.20, random_state = 0)
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc= accuracy_score(y_test,y_pred)

print(clf.predict(count_vect.transform(["Solution Pack Not connecting but after remove and reconnect showing OK but after Some time again Showing Not Connected.Solution Pack Not connecting but after remove and reconnect showing OK but after Some time again Showing Not Connected.Solution Pack Not Inserted properly. Waist Leak Also Found.There is a manufacturing fault in Solution Pack.Door closed properly. and Puted a paper piece to retail."])))

##############################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

cv_df.groupby('model_name').accuracy.mean()

#continue with the best model further 
# may be due to imbalance class - balance it further
# confusion matrix and heat map to see what is predicted incorrectly
# major of the predictions end up on the diagonal (predicted label = actual label)
from sklearn.model_selection import train_test_split
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, dataframe.index, test_size=0.20, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=category_id_dataframe['RMED FaultCode L1(New)'].values, yticklabels=category_id_dataframe['RMED FaultCode L1(New)'].values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


#there are misclassifications, and it it is important to see what caused it:
from IPython.display import display
for predicted in category_id_dataframe.category_id:
  for actual in category_id_dataframe.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 5:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(dataframe.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['RMED FaultCode L1(New)', 'Text']])
      print('')

#check the correlated unigram & bigrams in each target classification
model.fit(features, labels)
N = 10
for dataframe['RMED FaultCode L1(New)'], category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
#  print("# '{}':".format(dataframe['RMED FaultCode L1(New)']))
  print(category_id)
  print("  . Top unigrams:\n      . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))



from sklearn import metrics
#print(metrics.classification_report(y_test, y_pred, target_names=dataframe['RMED FaultCode L1(New)'].unique()))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc2=metrics.accuracy_score(y_test,y_pred)




#Bibiliography

# https://www.analyticsvidhya.com/blog/2015/10/6-practices-enhance-performance-text-classification-model/

#Google translate
#https://pypi.org/project/googletrans/
#https://py-googletrans.readthedocs.io/en/latest/
#https://cloud.google.com/translate/docs/












