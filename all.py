
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


short_pos = open("pos.txt","r").read()
short_neg = open("neg.txt","r").read()


# In[3]:


documents = []
documents1 = []
documents2 = []

for r in short_pos.split('\n'):
    documents1.append( r )
    documents.append( r )

for r in short_neg.split('\n'):
    documents2.append( r )
    documents.append( r )


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer(stop_words='english')
vector.fit(documents)


# In[22]:


counts1 = vector.transform(documents1)
array1 = counts1.toarray()
DataFrame1 = np.array(array1)
myDataFrame1 = pd.DataFrame(DataFrame1)
myDataFrame1['class'] = 'pos'


# In[6]:


counts2 = vector.transform(documents2)
array2 = counts2.toarray()
DataFrame2 = np.array(array2)
myDataFrame2 = pd.DataFrame(DataFrame2)
myDataFrame2['class'] = 'neg'
counts2.toarray()


# In[7]:


frames = [myDataFrame1, myDataFrame2]
df = pd.concat(frames)
df = df.sample(frac=1).reset_index(drop=True)
df


# In[8]:


#split data into 2 matrix - X and y
X = df.iloc[:, :18072]
y = df.iloc[:,-1]


# In[9]:


X = df.iloc[:, :18072].values
y = df.iloc[:,-1].values
X


# In[10]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
X = tfidf.fit_transform(X,y)
print(X.shape)
print(X.toarray())


# In[11]:


#splitting of data into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[12]:


X_train


# In[13]:


# classifierLogisticRegression
from sklearn.linear_model import LogisticRegression
classifierLogisticRegression = LogisticRegression()
classifierLogisticRegression.fit(X_train, y_train)

y_predictorLogisticRegression = classifierLogisticRegression.predict(X_test)
y_predictorLogisticRegression


# In[14]:


from sklearn.svm import SVC
classifierSVC = SVC(kernel='linear')
classifierSVC.fit(X_train, y_train.ravel())

y_predictorSVC = classifierSVC.predict(X_test)
y_predictorSVC


# In[15]:


from sklearn.naive_bayes import MultinomialNB
classifierMultinomialNB = MultinomialNB(alpha=1)
classifierMultinomialNB.fit(X_train, y_train)

y_predictorMultinomialNB = classifierMultinomialNB.predict(X_test)
y_predictorMultinomialNB


# In[16]:


#Move to KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
classifierKNeighborsClassifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
classifierKNeighborsClassifier.fit(X_train, y_train)

y_predictorKNeighborsClassifier = classifierKNeighborsClassifier.predict(X_test)
y_predictorKNeighborsClassifier


# In[17]:


#fit your training data to NB
from sklearn.naive_bayes import GaussianNB
classifierGaussianNB = GaussianNB()
classifierGaussianNB.fit(X_train.toarray(), y_train)
y_predictorGaussianNB = classifierGaussianNB.predict(X_test.toarray())
y_predictorGaussianNB


# In[18]:


# from sklearn.metrics import confusion_matrix
# falseCaci = confusion_matrix(y_test, y_predictorSVC)
# falseCaci


# In[19]:


from sklearn.metrics import accuracy_score
print("LogisticRegression : ",accuracy_score(y_test, y_predictorLogisticRegression, normalize=True, sample_weight=None))   
print("SVM(SVC) : ",accuracy_score(y_test, y_predictorSVC, normalize=True, sample_weight=None)) 
print("MultinomialNB : ",accuracy_score(y_test, y_predictorMultinomialNB, normalize=True, sample_weight=None))  
print("KNeighbors : ",accuracy_score(y_test, y_predictorKNeighborsClassifier, normalize=True, sample_weight=None))   
print("GaussianNB: ",accuracy_score(y_test, y_predictorGaussianNB, normalize=True, sample_weight=None))     

