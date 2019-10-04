# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:50:29 2019

@author: DIVNA
"""

import pandas as pd
import numpy as np
############# 2 sheets of data ###############
data_2019=pd.read_excel("C:\ML&AI\TextClassification\RAW DATA UPSWO_Translated.xlsx")
#data.columns.tolist()
#countries_english=['RLTD','RAME','RIND','RPAC','RSAF','RKOR']
# "(data["Filtering\n(Credit/Age/Class)"]=="No") & " = 2725 rec if we apply this filter
filtered_data_2019 = data_2019[(data_2019["Filtering\n(Credit/Age/Class)"]=="No") &  (data_2019["Work Type"]=="Visit") & (data_2019["Case Type (Related Case)"]=="Complaint") ]
filtered_data_2019.drop_duplicates(subset="SWO Number")
#reset all indexes as we dropped rows above
filtered_data_2019 = filtered_data_2019.reset_index(drop=True)

#concatenate 3 columns of text
filtered_data_2019['Text']= filtered_data_2019['Case Description Eng'].fillna('').map(str) +' ' + filtered_data_2019['Investigation Eng'].fillna('').map(str) + ' ' + filtered_data_2019['Corrective Action Eng'].fillna('').map(str)
# final columns for classification
#dataframe_2019=filtered_data_2019.loc[:,['SWO Number','Text','RMED FaultCode L1(New)']]
#,'RMED FaultCode L2(New)', 'RMED FaultCode L3(New)', 'RMED FaultCode L4(New)',]
dataframe_2019=filtered_data_2019.loc[:,['SWO Number','Text','RMED FaultCode L1(New)']]

data_2017=pd.read_excel("C:\ML&AI\TextClassification\RAW DATA(Main Update)#2 (2017-07-04)_Translated.xlsx")
filtered_data_2017 = data_2017[(data_2017["Filtering\n(Credit/Age/Class)"]=="No")  ]
filtered_data_2017.drop_duplicates(subset="SWO Number")
#reset all indexes as we dropped rows above
filtered_data_2017 = filtered_data_2017.reset_index(drop=True)
dataframe_2017 = filtered_data_2017.loc[:,['SWO Number','RMED FaultCode L1(New)']]
dataframe_2017['Text'] = filtered_data_2017['INESTIGATION (ENG)'].fillna('').map(str) + ' ' + filtered_data_2017['CORRECTION(ENG)'].fillna('').map(str)

# Combine 2 excel sheets
frame = [dataframe_2017, dataframe_2019]
dataframe = pd.concat(frame)

#Drop rows where there is no text
dataframe['Text'] = dataframe['Text'].replace(' ', np.nan)
dataframe = dataframe.dropna(axis=0, subset=['Text'])
dataframe = dataframe.reset_index(drop=True)
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

dataframe['RMED FaultCode L1(New)'].value_counts()

#Multi class NLP Classification
#create a column where each class has a unique id called category id
dataframe['category_id'] = dataframe['RMED FaultCode L1(New)'].factorize()[0]
category_id_dataframe = dataframe[['RMED FaultCode L1(New)', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_dataframe.values)
id_to_category = dict(category_id_dataframe[['category_id', 'RMED FaultCode L1(New)']].values)
dataframe.head()
#################################################################################################
x =  dataframe.Text
y =  dataframe['RMED FaultCode L1(New)']
y_df =  pd.DataFrame(dataframe['RMED FaultCode L1(New)'])

from sklearn.model_selection import train_test_split
SEED = 2000
x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(x, y, dataframe.index,test_size=.10, random_state=SEED)
x_train, x_test, y_train_df, y_test_df = train_test_split(x, y_df, test_size=.10, random_state=SEED)
#######################################################
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

from sklearn.feature_extraction.text import TfidfVectorizer
#min_df = When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
#norm='l2' = The cosine similarity between two vectors is their dot product when l2 norm has been applied.
tfidf = TfidfVectorizer(sublinear_tf=True,  norm='l2', encoding='latin-1', ngram_range=(1,1),stop_words='english',token_pattern=r'(?u)\b[A-Za-z]+\b', tokenizer=stemming_tokenizer)

tfidf.fit(x_train)

# encode document
xtrain_tfidf = tfidf.transform(x_train).toarray()
# summarize encoded vector
print(xtrain_tfidf.shape)

xtest_tfidf = tfidf.transform(x_test).toarray()

features = tfidf.fit_transform(dataframe.Text).toarray()
labels = dataframe.category_id
features.shape

#########################  over sampling   #########################

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(xtrain_tfidf, y_train)

###########################################

#from sklearn.decomposition import PCA
## if we want 0.9 as variance, what is the n_components that should be in PCA
#pca = PCA(0.99)
#pca.fit(X_train_res)
#PCA(copy=True, iterated_power='auto', n_components=0.9, random_state=None,  svd_solver='auto', tol=0.0, whiten=False)
#svd_components = pca.n_components_
#Out[70]: 1673 for 90% variance
# 4777 for 99% variance

from sklearn.decomposition import TruncatedSVD
svd_200 = TruncatedSVD(n_components = 200)
svd_200.fit(X_train_res)
svd_X_train_res_200 = svd_200.transform(X_train_res)
svd_x_test_tfidf_200 = svd_200.transform(xtest_tfidf)

#######################      conda install -c anaconda py-xgboost   ###########################
import xgboost as xgb
clf = xgb.XGBClassifier(max_depth=7, n_estimators=1000, colsample_bytree=0.8,subsample=0.7, nthread=10, learning_rate=0.01, objective='multi:softprob',silent=1,eta=0.4,num_class=5,num_rounds=15)


clf.fit(svd_X_train_res_200, y_train_res, eval_metric="mlogloss")
y_pred = clf.predict(svd_x_test_tfidf_200)

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=dataframe['RMED FaultCode L1(New)'].unique()))

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)

#                      precision    recall  f1-score   support
#
#              Defect       0.57      0.60      0.59        72
#              Others       0.67      0.77      0.72       277
#             Cleaned       0.53      0.67      0.59       144
#          Adjustment       0.81      0.73      0.77       734
#Assisted Replacement       0.49      0.48      0.49       245
#
#            accuracy                           0.68      1472
#           macro avg       0.62      0.65      0.63      1472
#        weighted avg       0.69      0.68      0.69      1472

#y_pred = pd.DataFrame(y_pred.ravel())
df_out = x_test.reset_index()
df_out["Actual"] = y_test_df.reset_index()['RMED FaultCode L1(New)']
df_out["Prediction"] = y_pred[0] #y_pred.reset_index()[0]
#y_test_df['preds'] = y_pred.reset_index()[0]
#df_out = pd.merge(dataframe,y_test_df[['preds']],how = 'left',left_index = True, right_index = True)
df_out.to_excel("output.xlsx",sheet_name='Sheet1')

#####################Pickle#########################

import pickle

filename='NLP-SVD-XGBM-RandCV-Trans.pkl'
pickle.dump(clf,open(filename,'wb'))

clf_loaded_pickle = pickle.load(open(filename,'rb'))
result = clf_loaded_pickle.score(svd_x_test_tfidf_200, y_test)
print(result)

#print(result)
# 0.6493506493506493

#check the correlated unigram & bigrams in each target classification
clf.fit(features, labels)
N = 10
for dataframe['RMED FaultCode L1(New)'], category_id in sorted(category_to_id.items()):
  indices = np.argsort(clf[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
#  print("# '{}':".format(dataframe['RMED FaultCode L1(New)']))
  print(category_id)
  print("  . Top unigrams:\n      . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))

from IPython.display import display
for predicted in category_id_dataframe.category_id:
  for actual in category_id_dataframe.category_id:
    if predicted != actual and conf_mat[actual, predicted] >= 5:
      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
      display(dataframe.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['RMED FaultCode L1(New)', 'Text']])
      print('')

#####################################################################################
      
#####################################################################################

from sklearn.decomposition import PCA
# if we want 0.9 as variance, what is the n_components that should be in PCA
pca = PCA(0.99)
pca.fit(X_train_res)
PCA(copy=True, iterated_power='auto', n_components=0.9, random_state=None,  svd_solver='auto', tol=0.0, whiten=False)
svd_components = pca.n_components_
# 5503

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = svd_components)
svd.fit(X_train_res)
svd_X_train_res = svd.transform(X_train_res)
svd_x_test_tfidf = svd.transform(xtest_tfidf)

#######################      conda install -c anaconda py-xgboost   ###########################
import xgboost as xgb
clf_pca = xgb.XGBClassifier(max_depth=7, n_estimators=1000, colsample_bytree=0.8,subsample=0.7, nthread=10, learning_rate=0.01, objective='multi:softprob',silent=1,eta=0.4,num_class=5,num_rounds=15)


clf_pca.fit(svd_X_train_res, y_train_res, eval_metric="mlogloss")
y_pred = clf_pca.predict(svd_x_test_tfidf)

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=dataframe['RMED FaultCode L1(New)'].unique()))

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)

#                     precision    recall  f1-score   support
#
#              Defect       0.55      0.24      0.33        72
#              Others       0.66      0.71      0.69       277
#             Cleaned       0.54      0.48      0.51       144
#          Adjustment       0.70      0.82      0.76       734
#Assisted Replacement       0.59      0.39      0.47       245
#
#            accuracy                           0.67      1472
#           macro avg       0.61      0.53      0.55      1472
#        weighted avg       0.65      0.67      0.65      1472
#y_pred = pd.DataFrame(y_pred.ravel())


#####################Pickle#########################

import pickle

filename='NLP-SVD-XGBM-RandCV-Trans.pkl_pca'
pickle.dump(clf_pca,open(filename,'wb'))
#################################################