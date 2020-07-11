#!/usr/bin/env python
# coding: utf-8

# ## Spam Text Message Detection Using NLP

# In[1]:


#pip install nltk


# In[47]:


#Natural language tool kit
import nltk 
nltk.download('stopwords')


# In[48]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[49]:


#First we will import our data
data = pd.read_csv("SPAM text message 20170820 - Data.csv")


# In[50]:


#Natural language tool kit
import nltk
nltk.download('punkt')


# In[51]:


#Natural language tool kit
import nltk
nltk.download('wordnet')


# In[52]:


#Reading the first few rows of the dataset to see what data is present in the dataset
data.head()
#It showed that dataset has two variables category and message


# In[53]:


#Identifying null values
print('No. of Samples: {}'.format(data.index.max()))
print('No. of nulls:\n{}'.format(data.isnull().sum()))


# In[54]:


#Making copy of the dataset
msgd=data


# In[55]:


#looking at the dataset
data.head()


# In[56]:


#calculating no. of ham and spam in the msg
msgd['Category'].value_counts()


# In[57]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


#Plotting the frequency of ham and spam in the message
data["Category"].value_counts().plot(kind = 'pie', explode = [0, 0.1], figsize = (6, 6), autopct = '%1.1f%%', shadow = True)
plt.ylabel("Spam vs Ham")
plt.legend(["Ham", "Spam"])
plt.show()


# In[59]:


import seaborn as sns


# In[60]:


#calculating the length of each message
msgd['Msg_Length']=data['Message'].apply(lambda X:len(X))


# In[61]:


# relation between spam messages and length
plt.rcParams['figure.figsize'] = (10, 7)
sns.boxenplot(x = msgd['Category'], y = msgd['Msg_Length'])
plt.title('Relation between Messages and Length', fontsize = 20)
plt.show()


# In[62]:


#looking at the data after length calculation
data.head()


# In[63]:


#looking the copy dataset
msgd.head()


# In[64]:


#extracting all the spam and ham word in the dataset
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import string
spam_messages = data[data["Category"] == "spam"]["Message"]
ham_messages = data[data["Category"] == "ham"]["Message"]

spam_words = []
ham_words = []

# Since this is just classifying the message as spam or ham, we can use isalpha(). 
# This will also remove the not word in something like can't etc. 
# In a sentiment analysis setting, its better to use 
# sentence.translate(string.maketrans("", "", ), chars_to_remove)

def extractSpamWords(spamMessages):
    global spam_words
    words = [word.lower() for word in word_tokenize(spamMessages) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]
    spam_words = spam_words + words
    
def extractHamWords(hamMessages):
    global ham_words
    words = [word.lower() for word in word_tokenize(hamMessages) if word.lower() not in stopwords.words("english") and word.lower().isalpha()]
    ham_words = ham_words + words

spam_messages.apply(extractSpamWords)
ham_messages.apply(extractHamWords)


# In[65]:


#pip install wordcloud


# In[65]:


#library for plotting word cloud
from wordcloud import WordCloud


# In[22]:


#Spam Word cloud

spam_wordcloud = WordCloud(width=600, height=400).generate(" ".join(spam_words))
plt.figure( figsize=(10,8))
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[23]:


#Ham word cloud

ham_wordcloud = WordCloud(background_color = 'black',width=600, height=400).generate(" ".join(ham_words))
plt.figure( figsize=(10,8))
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[24]:


# Top 10 spam words

spam_words = np.array(spam_words)
print("Top 10 Spam words are :\n")
pd.Series(spam_words).value_counts().head(n = 10)


# In[25]:


# Top 10 Ham words

ham_words = np.array(ham_words)
print("Top 10 Ham words are :\n")
pd.Series(ham_words).value_counts().head(n = 10)


# In[82]:


#plotting frequency distribution for spam and ham message by their length
f, ax = plt.subplots(1, 2, figsize = (20, 8))

sns.distplot(msgd[msgd["Category"] ==1]["Msg_Length"], bins = 20, ax = ax[0])
ax[0].set_xlabel("Spam Message Word Length")

sns.distplot(msgd[msgd["Category"] == 0]["Msg_Length"], bins = 20, ax = ax[1])
ax[0].set_xlabel("Ham Message Word Length")

plt.show()


# In[66]:


#Converting Category from categorical to binary numeric format
#Assigning ham=0 and spam=1
data["Category"] = [1 if each == "spam" else 0 for each in data["Category"]]


# In[67]:


#Checking if changes are made to category variable
msgd.head()


# In[28]:


#Histogram for length of spam and ham category messages
sns.set_context(context='notebook',font_scale=2)
msgd.hist(column='Msg_Length',by='Category',bins=100,figsize=(16,6),color='brown')


# In[68]:


#calculating avg. length of ham and spam messages
print('Average length of spam messages: ',data[data['Category']==1]['Msg_Length'].mean(),'characters')
print('Average length of ham messages: ',data[data['Category']==0]['Msg_Length'].mean(),'characters')
#this shows spam msgs are longer in length


# In[69]:


import nltk as nlp
import re
description_list = []
for description in data["Message"]:
    #Replacing all the puctuation with space
    description = re.sub("[^a-zA-Z]"," ",description)
    #Converting all the message data into lower case
    description = description.lower()   # buyuk harftan kucuk harfe cevirme
    #Separting each word in a row seperately (Hence doing this process for the entire column message)
    description = nlp.word_tokenize(description)
    #description = [ word for word in description if not word in set(stopwords.words("english"))]
    #finding word root for each word
    lemma = nlp.WordNetLemmatizer()
    description = [ lemma.lemmatize(word) for word in description]
    #appending the root word together for the entire one message
    description = " ".join(description)
    #appending the root word for each message in one list
    description_list.append(description) #we hide all word one section


# In[70]:


#We make bag of word it is including number of all word's info
#Finding the most common words used from the description_list
from sklearn.feature_extraction.text import CountVectorizer 
max_features = 3000 #We use the most common word
count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
print("the most using {} words: {}".format(max_features,count_vectorizer.get_feature_names()))


# In[71]:


#We separate our data is train and test
y = data.iloc[:,0].values
x = sparce_matrix
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)


# In[72]:


#Model Creation
#NB Model
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("the accuracy of our model: {}".format(nb.score(x_test,y_test)))


# In[73]:


#Logistic Regression Model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs',max_iter = 200)
lr.fit(x_train,y_train)
print("our accuracy is: {}".format(lr.score(x_test,y_test)))


# In[74]:


#knn model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test))


# In[75]:


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


# In[76]:


#implement grid search cv
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    },
    'naive_bayes_gaussian': {
        'model': GaussianNB(),
        'params': {}
    },
#     'naive_bayes_multinomial': {
#         'model': MultinomialNB(),
#         'params': {}
#     },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini','entropy'],
            
        }
    }     
}


# In[77]:


#prdicting accuracy scores for test data using grid search cv
from sklearn.model_selection import GridSearchCV
import pandas as pd
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x_train,y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })


# In[79]:


#accuracy scores for all the classifiers
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df


# In[83]:


#bar plot showing the accuracy scores
df.plot(kind='bar', ylim=(0.85,1.0), figsize=(11,6), align='center',color='grey')
plt.xticks(np.arange(5), df.index)
plt.ylabel('Accuracy Score')
plt.title('Distribution by Classifier')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)


# In[ ]:


#SVM andlogistic regression show highest accuracy with almost 98% 

