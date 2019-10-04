# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:20:39 2019

@author: DIVNA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
filtered_data_2019['Text']= filtered_data_2019['Case Description Eng'].map(str) +' ' + filtered_data_2019['Investigation Eng'].map(str) + ' ' + filtered_data_2019['Corrective Action Eng'].map(str)
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
dataframe_2017['Text'] = filtered_data_2017['INESTIGATION (ENG)'].map(str) + ' ' + filtered_data_2017['CORRECTION(ENG)'].map(str)


# Combine 2 excel sheets
frame = [dataframe_2017, dataframe_2019]
dataframe = pd.concat(frame)
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
y = dataframe['category_id']
from sklearn.model_selection import train_test_split
SEED = 2000
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.10, random_state=SEED)

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

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(xtrain_tfidf, y_train)

#from sklearn.decomposition import PCA
## if we want 0.9 as variance, what is the n_components that should be in PCA
#pca = PCA(0.99)
#pca.fit(X_train_res)
#PCA(copy=True, iterated_power='auto', n_components=0.99, random_state=None,  svd_solver='auto', tol=0.0, whiten=False)
#svd_components = pca.n_components_
#Out[70]: 1673 for 90% variance
# 4777 for 99% variance

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 200)
svd.fit(X_train_res)
svd_X_train_res = svd.transform(X_train_res)
svd_xtest_tfidf = svd.transform(xtest_tfidf)

######################################################################
X_train = np.reshape(svd_X_train_res, (svd_X_train_res.shape[0], svd_X_train_res.shape[1], 1))
X_test = np.reshape(svd_xtest_tfidf, (svd_xtest_tfidf.shape[0], svd_xtest_tfidf.shape[1], 1))

#X_train = np.reshape(X_train_res, (X_train_res.shape[0], X_train_res.shape[1], 1))
#X_test = np.reshape(xtest_tfidf, (xtest_tfidf.shape[0], xtest_tfidf.shape[1], 1))

from keras.utils import to_categorical
y_binary = to_categorical(y_train_res)
y_binary_test = to_categorical(y_test)
#Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping

##################################################################################
# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 512, return_sequences = True, input_shape = (X_train.shape[1], 1)))

# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 512, activation='relu',dropout=0.2, return_sequences = True))

#model.add(LSTM(units = 512, activation='relu',dropout=0.2, return_sequences = True))
## Adding a third LSTM layer and some Dropout regularisation
#moeld.add(LSTM(units = 50, return_sequences = True))
#moeld.add(Dropout(0.2))
#
## Adding a fourth LSTM layer and some Dropout regularisation
#moeld.add(LSTM(units = 50))
#moeld.add(Dropout(0.2))
#model.add(LSTM(units = 512))
model.add(Flatten())
# Adding the output layer
model.add(Dense(units = 5, activation='softmax'))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics=['accuracy'])


# Fitting the RNN to the Training set
model.fit(X_train, y_binary, epochs = 10, batch_size = 32)

y_pred = model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis = 1)

score = model.evaluate(X_test, y_binary_test,batch_size=32)

print('Test accuracy:', score[1])

from sklearn import metrics
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred_argmax)
print(metrics.accuracy_score(y_test, y_pred_argmax))
import pickle

filename='NLP-LSTM1.pkl'
pickle.dump(model,open(filename,'wb'))

########################################################################################
model1 = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model1.add(LSTM(units = 512, return_sequences = True, input_shape = (X_train.shape[1], 1)))

# Adding a second LSTM layer and some Dropout regularisation
model1.add(LSTM(units = 512, activation='relu',dropout=0.2, return_sequences = True))

model1.add(LSTM(units = 512, activation='relu',dropout=0.2, return_sequences = True))
## Adding a third LSTM layer and some Dropout regularisation
#moeld.add(LSTM(units = 50, return_sequences = True))
#moeld.add(Dropout(0.2))
#
## Adding a fourth LSTM layer and some Dropout regularisation
#moeld.add(LSTM(units = 50))
#moeld.add(Dropout(0.2))
#model.add(LSTM(units = 512))
model1.add(Flatten())
# Adding the output layer
model1.add(Dense(units = 5, activation='softmax'))

# Compiling the RNN
model1.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics=['accuracy'])


# Fitting the RNN to the Training set
model1.fit(X_train, y_binary, epochs = 4, batch_size = 32)

y_pred = model1.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis = 1)

score = model1.evaluate(X_test, y_binary_test,batch_size=32)

print('Test accuracy:', score[1])

from sklearn import metrics
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred_argmax)
print(metrics.accuracy_score(y_test, y_pred_argmax))
import pickle

filename='NLP-LSTM1.pkl'
pickle.dump(model,open(filename,'wb'))
