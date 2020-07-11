# Spam-Ham-Classification

The project detects and classify whether the message is a spam message or not. The project is carried out using NLP and different classifiers to obtain the best results.
The project includes SVC, Logistic Regression, Decision Tree, Naive Bayes and Random Forest as classifiers. In order to obtain the best results in terms of accuracy GridSearchCV in the project.

Spamming is the use of messaging systems to send an unsolicited message (spam), especially advertising, as well as sending messages repeatedly on the same website. Email Spam, Instant Messaging Spam, Usenet Newsgroup Spam, Web Search Engine Spam, Spam in Blogs, Online Classified ads spam etc. are common type of spams. 

DATA ENGINEERING AND PROCESS 

EDA 
(i)	The Dataset contains two variables category and message. 
(ii)	Removed Null values, Calculated message length for each message. 
(iii)	Plotted chart to see frequency of each category and to depict relationship between message and its length. 
(iv)	Converted Category(spam and ham) into binary numerical values and calculated their frequency.
(v)	  Plotted word cloud for both spam and ham to see more frequent words in each category. 
(vi)	Removed punctuation, converted to lower case, applied tokenization and calculated the root of each word in the message. Created bag of words and created sparce matrix with the help of it. 

Modeling and Performance 
(i) 	Split the data into training and testing sets. 
(ii)  Applied Grid Search CV to classify the message text as ham or spam. Classifiers Used: Decision Tree, Random Forest, Logistic Regression, SVM and Naive Bayes. Logistic Regression and SVM gave the best results having 98 percent accuracy
