#!/usr/bin/env python
# coding: utf-8

# Importing all the Libraries

# In[41]:


import numpy as np
import pandas as pd
import re
import string
from textblob import TextBlob
from textblob import Word
from nltk.stem.wordnet import WordNetLemmatizer


# In[42]:


df= pd.read_csv("data-2_train.csv", skiprows =1, names = ['ID', 'Text', 'Aspect_Term', 'Term_Location', 'Class'], index_col = 'ID')
text = df.Text
aspect = df.Aspect_Term
term_Loc = df.Term_Location


# In[43]:


df1= pd.read_csv("Data-2_test.csv", skiprows =1, names = ['ID', 'Text', 'Aspect_Term', 'Term_Location'], index_col = 'ID')
text1 = df1.Text
aspect1 = df1.Aspect_Term
term_Loc1 = df1.Term_Location


# In[44]:


butt = {'or','nor','so', 'yet','because', 'but', 'that','except', 'while','after','although','plus'}
from textblob import Word
def get_lexicon_value(sentence, termLoc):
    sentence = sentence.replace("[comma]", ",")
    splitLocations = termLoc.split("--")

    beforeAspect, afterAspect = sentence[:int(splitLocations[0])], sentence[int(splitLocations[1]):]
    beforeWords =beforeAspect.replace(",", " XX ")
    afterWords = afterAspect.replace(",", " XX ")
    beforeWords = beforeWords.split()
    afterWords = afterWords.split()

    start = 0
    for i, words in enumerate(beforeWords):
        if words in butt:
            start = max(start, i)
    if start == 0:
        final_sentence = ' '.join(beforeWords[start:]) + ' '
    else:
        final_sentence = ' '.join(beforeWords[start+1:]) + ' '


    pos = len(afterWords)
    i = 0
    for i, words in enumerate(afterWords):
        if words in butt:
            pos = i
            break
    final_sentence += ' '.join(afterWords[:pos])
    return re.sub('\W+', ' ', final_sentence)


# In[45]:


dummy = []
for i in range(len(text)):
    dummy.append(get_lexicon_value(text[i],term_Loc[i]))


# In[46]:


sub = []
pol = []
for i in dummy:
    val = TextBlob(i).sentiment
    pol.append(val[0])
    sub.append(val[1])


# In[47]:


df['Pol'] = np.array(pol).reshape(-1,1)
df['Sub'] = np.array(sub).reshape(-1,1)
Y = np.array(df['Class']).reshape(-1,1)
# X = df['Pol'].reshape(-1,1)
X = df[['Pol', 'Sub']]
X_train = X[:2000]
y_train = Y[:2000]
X_test = X[2000:]
y_test = Y[2000:]


# In[48]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=600,
    learning_rate=1)


bdt_discrete = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=3),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")


# Testing using Trained classifier

# In[49]:


df= pd.read_csv("Data-2_test.csv", skiprows =1, names = ['ID', 'Text', 'Aspect_Term', 'Term_Location', 'Class'])


# In[50]:


df.head()


# In[51]:


text = df.Text
aspect = df.Aspect_Term
term_Loc = df.Term_Location
sub = []
pol = []

dummy = []
for i in range(len(text)):
    dummy.append(get_lexicon_value(text[i],term_Loc[i]))
    


# In[52]:



print(dummy)


# In[53]:


for i in dummy:
    val = TextBlob(i).sentiment
    pol.append(val[0])
    sub.append(val[1])


# In[54]:


print("pol",sub)


# In[55]:


sub1 = []
pol1 = []

dummy1 = []
for i in range(len(text)):
    dummy1.append(get_lexicon_value(text[i], term_Loc[i]))

for i in dummy1:
    val = TextBlob(i).sentiment
    pol1.append(val[0])
    sub1.append(val[1])

df['Pol'] = np.array(pol1).reshape(-1,1)
df['Sub'] = np.array(sub1).reshape(-1,1)
# X = df['Pol'].reshape(-1,1)
X1 = df[['Pol', 'Sub']]



Xnew = df[['Pol', 'Sub']]

bdt_discrete.fit(X,Y)
result = bdt_discrete.predict(Xnew)

print(result)


# In[56]:


#outputting the results to the file
#change the file name.

f = open('output.txt','a')
id = df.ID

for i in range(len(df['ID'])):
    f.write(str(id[i])+";;"+str(result[i])+'\n')
f.close()


# Additional Methods Tried

# In[57]:


from sklearn import model_selection
seed = 10
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[58]:


from sklearn import svm
from sklearn.model_selection import GridSearchCV
svc = svm.SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
clf = GridSearchCV(svc, parameters)
clf.fit(X,Y)


# In[59]:


scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']


# In[60]:


print(scores)


# In[61]:


print(scores_std)


# In[ ]:


clf = svm.SVC()
# if tune_hyper_parameters:
parameters = {
'C': np.arange(1, 5, 1).tolist(),
'kernel': ['rbf', 'poly'],  # precomputed,'poly', 'sigmoid'
'degree': np.arange(0, 3, 1).tolist(),
'gamma': np.arange(0.0, 1.0, 0.1).tolist(),
'coef0': np.arange(0.0, 1.0, 0.1).tolist(),
'shrinking': [True],
'probability': [False],
'tol': np.arange(0.001, 0.01, 0.001).tolist(),
'cache_size': [2000],
'class_weight': [None],
'verbose': [False],
'max_iter': [-1],
'random_state': [None],
}
gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
gs_clf.fit(X_train, y_train)
print(gs_clf.best_score_)


# In[ ]:


for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


# In[ ]:


lala =[]
for i in range(len(dummy)):
    if df['Class'][i] == 0:
        lala.append(TextBlob(dummy[i]).sentiment.polarity)


# In[ ]:


print(max(lala))
print(sum(lala)/len(lala))

