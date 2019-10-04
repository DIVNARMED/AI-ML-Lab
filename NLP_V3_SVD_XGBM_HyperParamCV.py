# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data=pd.read_excel("C:\ML&AI\TextClassification\RAW DATA(Main Update)#2 (2019-07-02).xlsx", sheet_name="All",skiprows=5)
#data.columns.tolist()

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

dataframe['RMED FaultCode L1(New)'].value_counts()

#Multi class NLP Classification
#create a column where each class has a unique id called category id
dataframe['category_id'] = dataframe['RMED FaultCode L1(New)'].factorize()[0]
category_id_dataframe = dataframe[['RMED FaultCode L1(New)', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_dataframe.values)
id_to_category = dict(category_id_dataframe[['category_id', 'RMED FaultCode L1(New)']].values)
dataframe.head()

x = dataframe.Text
y = dataframe['RMED FaultCode L1(New)']
from sklearn.model_selection import train_test_split
SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

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

xvalid_tfidf =  tfidf.transform(x_validation).toarray()
xtest_tfidf = tfidf.transform(x_test).toarray()

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(xtrain_tfidf, y_train)

##############################################################################
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import LinearSVC
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.model_selection import cross_val_score
#models = [
#    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
#    LinearSVC(),
#    MultinomialNB(),
#    LogisticRegression(random_state=0),
#]
#CV = 5
#cv_df = pd.DataFrame(index=range(CV * len(models)))
#entries = []
#for model in models:
#  model_name = model.__class__.__name__
#  accuracies = cross_val_score(model, X_train_res, y_train_res, scoring='accuracy', cv=CV)
#  for fold_idx, accuracy in enumerate(accuracies):
#    entries.append((model_name, fold_idx, accuracy))
#cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
#
#
#import seaborn as sns
#sns.boxplot(x='model_name', y='accuracy', data=cv_df)
#sns.stripplot(x='model_name', y='accuracy', data=cv_df,size=8, jitter=True, edgecolor="gray", linewidth=2)
#plt.show()
#
#cv_df.groupby('model_name').accuracy.mean()
#
##continue with the best model further 
## may be due to imbalance class - balance it further
## confusion matrix and heat map to see what is predicted incorrectly
## major of the predictions end up on the diagonal (predicted label = actual label)
#from sklearn.model_selection import train_test_split
#model = LinearSVC()
##X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, dataframe.index, test_size=0.20, random_state=0)
#model.fit(X_train_res, y_train_res)
#y_pred = model.predict(xvalid_tfidf)
#from sklearn.metrics import confusion_matrix
#conf_mat = confusion_matrix(y_validation, y_pred)
#fig, ax = plt.subplots(figsize=(6,6))
#sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=category_id_dataframe['RMED FaultCode L1(New)'].values, yticklabels=category_id_dataframe['RMED FaultCode L1(New)'].values)
#plt.ylabel('Actual')
#plt.xlabel('Predicted')
#plt.show()


##there are misclassifications, and it it is important to see what caused it:
#from IPython.display import display
#for predicted in category_id_dataframe.category_id:
#  for actual in category_id_dataframe.category_id:
#    if predicted != actual and conf_mat[actual, predicted] >= 5:
#      print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
#      display(dataframe.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['RMED FaultCode L1(New)', 'Text']])
#      print('')
#
##check the correlated unigram & bigrams in each target classification
#model.fit(features, labels)
#N = 10
#for dataframe['RMED FaultCode L1(New)'], category_id in sorted(category_to_id.items()):
#  indices = np.argsort(model.coef_[category_id])
#  feature_names = np.array(tfidf.get_feature_names())[indices]
#  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
#  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
##  print("# '{}':".format(dataframe['RMED FaultCode L1(New)']))
#  print(category_id)
#  print("  . Top unigrams:\n      . {}".format('\n       . '.join(unigrams)))
#  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))



#from sklearn import metrics
##print(metrics.classification_report(y_test, y_pred, target_names=dataframe['RMED FaultCode L1(New)'].unique()))
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_validation, y_pred)
#acc2=metrics.accuracy_score(y_validation,y_pred)

###########################################

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=200)
svd.fit(X_train_res)
svd_X_train_res = svd.transform(X_train_res)
svd_xvalid_tfidf = svd.transform(xvalid_tfidf)
svd_xtest_tfidf = svd.transform(xtest_tfidf)
#######################      conda install -c anaconda py-xgboost   ###########################

#########################################################################
# tuning hyper parameters, test with CV = 5
########################################################################
# from sklearn.grid_search import GridSearchCV
# number of trees, tree depth and the learning rate as most crucial parameters.
# n_estimators captures the number of trees that we add to the model. A high number of trees can be computationally expensive
# max_depth bounds the maximum depth of the tree
# The square root of features is usually a good starting point
# Subsample sets the fraction of samples to be used for fitting the individual base learners. Values lower than 1 generally lead to a reduction of variance and an increase in bias.
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

p_test3 = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750]}
tuning = GridSearchCV(estimator =GradientBoostingClassifier(max_depth=4, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10),param_grid = p_test3, scoring='accuracy',n_jobs=4,iid=False, cv=5)
tuning.fit(svd_X_train_res, y_train_res)
tuning.best_params_, tuning.best_score_
#    ({'learning_rate': 0.1, 'n_estimators': 1250}, 0.9279073046083356)
#  ({'learning_rate': 0.15, 'n_estimators': 1750}, 0.5295885100008811)
p_test2 = {'max_depth':[2,3,4,5,6,7] }
tuning = GridSearchCV(estimator =GradientBoostingClassifier(learning_rate=0.01,n_estimators=1500, min_samples_split=2, min_samples_leaf=1, subsample=1,max_features='sqrt', random_state=10),param_grid = p_test2, scoring='accuracy',n_jobs=4,iid=False, cv=5)
tuning.fit(svd_X_train_res, y_train_res)
tuning.best_params_, tuning.best_score_
#  ({'max_depth': 7}, 0.935271830117191)

p_test4 = {'min_samples_split':[2,4,6,8,10,20,40,60,100], 'min_samples_leaf':[1,3,5,7,9]}
tuning = GridSearchCV(estimator =GradientBoostingClassifier(learning_rate=0.1, n_estimators=1250,max_depth=7, subsample=1,max_features='sqrt', random_state=10), param_grid = p_test4, scoring='accuracy',n_jobs=4,iid=False, cv=5)
tuning.fit(svd_X_train_res, y_train_res)
tuning.best_params_, tuning.best_score_
#  ({'min_samples_leaf': 9, 'min_samples_split': 40}, 0.9356181161335801)

p_test5 = {'max_features':[6,8,9,10,11,12,13,14,15]}
tuning = GridSearchCV(estimator =GradientBoostingClassifier(learning_rate=0.1, n_estimators=1250,max_depth=7, min_samples_split=40, min_samples_leaf=9, subsample=1, random_state=10), param_grid = p_test5, scoring='accuracy',n_jobs=4,iid=False, cv=5)
tuning.fit(svd_X_train_res, y_train_res)
tuning.best_params_, tuning.best_score_
#   ({'max_features': 8}, 0.9393840867036743)

p_test6= {'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1]}
tuning = GridSearchCV(estimator =GradientBoostingClassifier(learning_rate=0.1, n_estimators=1250,max_depth=7, min_samples_split=40, min_samples_leaf=9,max_features=8 , random_state=10), param_grid = p_test6, scoring='accuracy',n_jobs=4,iid=False, cv=5)
tuning.fit(svd_X_train_res, y_train_res)
tuning.best_params_, tuning.best_score_
#  ({'subsample': 0.85}, 0.9375046259582343)

#########################################################################
# Upon tuning hyper parameters, apply the same to the model training and testing
########################################################################
import xgboost as xgb
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(svd_result, y_train_res, dataframe.index, test_size=0.20, random_state=0)
clf = xgb.XGBClassifier(max_depth=7, n_estimators=1250, colsample_bytree=0.8,subsample=0.85, nthread=10, learning_rate=0.1,  min_samples_split=40, min_samples_leaf=9,max_features=8 ,objective='multi:softprob',silent=1,eta=0.4,num_class=3,num_rounds=15)
clf.fit(svd_X_train_res, y_train_res)
y_pred = clf.predict(svd_xtest_tfidf)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred.ravel())
import seaborn as sns
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=category_id_dataframe['RMED FaultCode L1(New)'].values, yticklabels=category_id_dataframe['RMED FaultCode L1(New)'].values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=dataframe['RMED FaultCode L1(New)'].unique()))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.ravel())
print(metrics.accuracy_score(y_test,y_pred.ravel()))



#                     precision    recall  f1-score   support
#
#              Defect       0.00      0.00      0.00         2
#              Others       0.33      1.00      0.50         1
#    Revisit required       0.75      0.60      0.67         5
#Assisted Replacement       0.82      0.82      0.82        11
#             Cleaned       0.67      0.67      0.67         3
#          Adjustment       1.00      1.00      1.00         1
#
#            accuracy                           0.70        23
#           macro avg       0.59      0.68      0.61        23
#        weighted avg       0.70      0.70      0.69        23
#
#0.6956521739130435
########################################
