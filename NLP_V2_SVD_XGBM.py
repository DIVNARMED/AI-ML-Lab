# -*- coding: utf-8 -*-
"""
Spyder Editor

https://towardsdatascience.com/the-theory-you-need-to-know-before-you-start-an-nlp-project-1890f5bbb793
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

dataframe['RMED FaultCode L1(New)'].value_counts()


##############################################################################

#Multi class NLP Classification
    
#create a column where each class has a unique id called category id
dataframe['category_id'] = dataframe['RMED FaultCode L1(New)'].factorize()[0]
category_id_dataframe = dataframe[['RMED FaultCode L1(New)', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_dataframe.values)
id_to_category = dict(category_id_dataframe[['category_id', 'RMED FaultCode L1(New)']].values)
dataframe.head()


from sklearn.feature_extraction.text import TfidfVectorizer
#min_df = When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
#norm='l2' = The cosine similarity between two vectors is their dot product when l2 norm has been applied.
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, norm='l2', encoding='latin-1', stop_words='english',token_pattern=r'(?u)\b[A-Za-z]+\b',ngram_range=(1, 1))
features = tfidf.fit_transform(dataframe.Text).toarray()
labels = dataframe.category_id
features.shape
#SVD is used for dimensionality reduction
# Apply SVD, I chose 200 components. 120-200 components are good enough for SVM model.
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=200)
svd.fit(features)
svd_result = svd.transform(features)

#######################      conda install -c anaconda py-xgboost   ###########################
import xgboost as xgb
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(svd_result, labels, dataframe.index, test_size=0.20, random_state=0)
clf = xgb.XGBClassifier(max_depth=3, n_estimators=300, colsample_bytree=0.8,
                        subsample=0.8, nthread=10, learning_rate=0.1, objective='multi:softprob',
                        silent=1,eta=0.4,num_class=3,num_rounds=15)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
import seaborn as sns
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=category_id_dataframe['RMED FaultCode L1(New)'].values, yticklabels=category_id_dataframe['RMED FaultCode L1(New)'].values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=dataframe['RMED FaultCode L1(New)'].unique()))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(metrics.accuracy_score(y_test,y_pred))

#############################################################################################
