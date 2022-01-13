######### PYTHON VERSION 3.9.6 ##########
import pandas as pd
import numpy as np
import nltk    # stemming, lemmatization etc
nltk.download('wordnet')
nltk.download('stopwords')
import re      # for removing urls etc
import urllib
import contractions # won't to will not, don't to do not 
from bs4 import BeautifulSoup # remove html content 
import sklearn
text_df = pd.read_csv('data.tsv', error_bad_lines = False, sep = '\t', warn_bad_lines=False)
import warnings
warnings.filterwarnings("ignore")
# \t because its tsv file 
# it will throw an error if number of rows are not in correct allignment 
# error_bad_lines = False will ignore such lines, when True (default)it will throw an error



# Keep rating and reviews only
text_df = text_df[['review_body','star_rating']] 

# drop na rows from ratings and reviews
text_df.dropna(subset = ["review_body"],inplace=True)
text_df.dropna(subset = ["star_rating"],inplace=True)

# Print 3 reviews
#print("\n ************ 3 Sample Reviews ***********\n")   
#print(text_df.sample(3))

#print("\n ************ Frequency of each rating  ************\n")

# Print frequency of the rating
count = text_df['star_rating'].value_counts() # how many rows for each rating 1,2,3,4,5 
#print(count)

# 1/2 rating negative sentinment 
# if 3 discard because its neutral 
# if its 4/5 positive sentiment



text_df['label'] = np.where(text_df["star_rating"] >= 4, 1, 0)     # create positive and negative sentiment label
text_df['sentiment'] = np.where(text_df["star_rating"] >= 4, "positive", "negative")     # create positive and negative sentiment label
text_df['label'] = np.where(text_df["star_rating"] == 3, -1,text_df['label'])
text_df['sentiment'] = np.where(text_df["star_rating"] == 3, "neutral",text_df['sentiment'])
text_df = text_df[['star_rating','review_body','label','sentiment']]           # copying to a new data frame 


count = text_df['sentiment'].value_counts()                         # counting frequency of each label 
#print("\n************ Frequency of each sentiment ************\n")
print(count['positive'],",",count['negative'],",",count['neutral'])
 
text_df = text_df[text_df.star_rating != 3]                           # delete the rows with neutral rating 3.
count = text_df['sentiment'].value_counts()                         # counting frequency of each label after neutral is dropped
#print("\n************ Frequency of each sentiment after removing neutral reviews ************\n")
#print(count)
text_df = text_df[['review_body','star_rating','label']]

sm0 = text_df.label[text_df.label.eq(0)].sample(100000,random_state=80).index    #randomly select 100000 positive reviews
sm1 = text_df.label[text_df.label.eq(1)].sample(100000, random_state=80).index    #randomly select 100000 negative reviews

text_df = text_df.loc[sm0.union(sm1)]     # combine into one dataset

avg_char_before = text_df['review_body'].apply(lambda a :len(str(a))).mean()  # finding avg no. of characters in a review
#print(" Avg. character length before cleaning :",avg_char)

#print("\n ************ 3 Sample Reviews before cleaning ***********\n")  
#print(text_df['review_body'].sample(3))

text_df["review_body"] = text_df["review_body"].str.lower()

# Removing html tags using beautiful soup like <br> tags
text_df["review_body"] = text_df["review_body"].apply(lambda x: BeautifulSoup(str(x)).get_text()) 

# Removing urls from reviews
text_df["review_body"] = text_df["review_body"].apply(lambda x: re.sub(r'\s*(https?://|www\.)+\S+(\s+|$)', " ", str(x), flags=re.UNICODE))



# Removing Digits from the review_body 
text_df["review_body"] = text_df["review_body"].apply(lambda x: re.sub(r"[^\D']+", " ", str(x), flags=re.UNICODE)) # remove all numbers

# Removing Special Characters
text_df["review_body"] = text_df["review_body"].apply(lambda x: re.sub(r"[^\w']+", " ", str(x), flags=re.UNICODE)) # remove all special characters



#remove more than one spaces
text_df["review_body"] = text_df["review_body"].apply(lambda x: re.sub(r'\s+',' ', str(x), flags = re.UNICODE))  
#print(text_df.sample(10))

#print(text_df.columns)

def contractionfunction(s):                      
    s = s.apply(lambda x: contractions.fix(x))
    return s

text_df_onecol = contractionfunction(text_df["review_body"])
text_df["review_body"] = text_df_onecol
text_df["review_body"] = text_df["review_body"].str.lower()     # convert everything to lower case again because contractions adds I 


#print(text_df.columns)

avg_char_before_processing = text_df['review_body'].apply(lambda a :len(str(a))).mean()  # finding avg no. of characters in a review
#print("Avg. character length before preprocessing", avg_char)

from nltk.corpus import stopwords
# storing all the stop words
stop_words = stopwords.words('english')

# remove stop words from each review  
text_df["review_body"] = text_df["review_body"].apply(lambda x: " ".join([item  for item in str(x).split() if item not in stop_words]))
#print(text_df.sample(10))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

text_df["review_body"] = text_df["review_body"].apply(lambda x: " ".join([lemmatizer.lemmatize(item)  for item in str(x).split()]))

avg_char_after = text_df['review_body'].apply(lambda x : (len(str(x)))).mean()
#print("Avg. character length after data cleaning + preprocessing :", avg_char_after)
print(avg_char_before,",",avg_char_before_processing)
print(avg_char_before_processing,",",avg_char_after)

from sklearn.feature_extraction.text import TfidfVectorizer

# fit and transform on train data and only transform test data respectively.
# converting each review to a vector of max 2000 words 
# min_df specifies min frequency of a word selected as a feature i.e the word has to occur atleast once 
# max_df ensures that a word used in more than 70% of the reviews is not considered as a feature

from sklearn.model_selection import train_test_split 
#Split train and test into 80 20 split
X_train, X_test, y_train, y_test = train_test_split(text_df["review_body"],text_df["label"], test_size=0.2, random_state=90)

tfidfconverter = TfidfVectorizer(max_features=2000, min_df=1, max_df=0.7)

# fit decides the features based on the train dataset whose retrictions were described in the above TfidfVectorizer function 
X_train = tfidfconverter.fit_transform(X_train)
X_test = tfidfconverter.transform(X_test)

from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

# standard scalera is a function used to normalize the review vectors 
sc = StandardScaler(with_mean=False)

# using the normalizing function to create a normalized training dataset
X_train_std = sc.fit_transform(X_train)

# normalize the test data using the same scaler
X_test_std = sc.transform(X_test)

# Create a perceptron object with the parameters: 40 iterations (epochs) over the data, and a learning rate of 0.1
ppn = Perceptron(max_iter=100, eta0=0.1, random_state=0)

# Train the perceptron
ppn.fit(X_train_std, y_train)


#print("\n ************ Evaluation metrics on training data ***********\n") 

y_pred_train = ppn.predict(X_train_std)

#print('Training Accuracy: %.5f' % accuracy_score(y_train, y_pred_train))
#print('Training F1 Score: %.5f' % f1_score(y_train, y_pred_train))
#print('Training Precision Score: %.5f' % precision_score(y_train, y_pred_train))
#print('Training Recall Score: %.5f' % recall_score(y_train, y_pred_train))


y_pred_test = ppn.predict(X_test_std)

#print("\n ************ Evaluation metrics on test data ***********\n") 

#print('Testing Accuracy: %.5f' % accuracy_score(y_test, y_pred_test))
#print('Testing F1 Score: %.5f' % f1_score(y_test, y_pred_test))
#print('Testing Precision Score: %.5f' % precision_score(y_test, y_pred_test))
#print('Testing Recall Score: %.5f' % recall_score(y_test, y_pred_test))
# from sklearn import svm
print('%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f' % (accuracy_score(y_train, y_pred_train), precision_score(y_train, y_pred_train), recall_score(y_train, y_pred_train), f1_score(y_train, y_pred_train),accuracy_score(y_test, y_pred_test),precision_score(y_test, y_pred_test), recall_score(y_test, y_pred_test),f1_score(y_test, y_pred_test)))

from sklearn import svm

#Create a svm Classifier
svm_clf = svm.LinearSVC() # Linear Kernel

#Train the model using the training sets
svm_clf.fit(X_train, y_train)



#print("\n ************ Evaluation metrics on training data ***********\n") 

y_pred_train = svm_clf.predict(X_train)

#print('Training Accuracy: %.5f' % accuracy_score(y_train, y_pred_train))
#print('Training F1 Score: %.5f' % f1_score(y_train, y_pred_train))
#print('Training Precision Score: %.5f' % precision_score(y_train, y_pred_train))
#print('Training Recall Score: %.5f' % recall_score(y_train, y_pred_train))

y_pred_test = svm_clf.predict(X_test)

#print("\n ************ Evaluation metrics on test data ***********\n") 

#print('Testing Accuracy: %.5f' % accuracy_score(y_test, y_pred_test))
#print('Testing F1 Score: %.5f' % f1_score(y_test, y_pred_test))
#print('Testing Precision Score: %.5f' % precision_score(y_test, y_pred_test))
#print('Testing Recall Score: %.5f' % recall_score(y_test, y_pred_test))

print('%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f' % (accuracy_score(y_train, y_pred_train), precision_score(y_train, y_pred_train), recall_score(y_train, y_pred_train), f1_score(y_train, y_pred_train),accuracy_score(y_test, y_pred_test),precision_score(y_test, y_pred_test), recall_score(y_test, y_pred_test),f1_score(y_test, y_pred_test)))

from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)


#print("\n ************ Evaluation metrics on training data ***********\n") 

y_pred_train = logreg.predict(X_train)

#print('Training Accuracy: %.5f' % accuracy_score(y_train, y_pred_train))
#print('Training F1 Score: %.5f' % f1_score(y_train, y_pred_train))
#print('Training Precision Score: %.5f' % precision_score(y_train, y_pred_train))
#print('Training Recall Score: %.5f' % recall_score(y_train, y_pred_train))

y_pred_test = logreg.predict(X_test)

#print("\n ************ Evaluation metrics on test data ***********\n") 

#print('Testing Accuracy: %.5f' % accuracy_score(y_test, y_pred_test))
#print('Testing F1 Score: %.5f' % f1_score(y_test, y_pred_test))
#print('Testing Precision Score: %.5f' % precision_score(y_test, y_pred_test))
#print('Testing Recall Score: %.5f' % recall_score(y_test, y_pred_test))

print('%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f' % (accuracy_score(y_train, y_pred_train), precision_score(y_train, y_pred_train), recall_score(y_train, y_pred_train), f1_score(y_train, y_pred_train),accuracy_score(y_test, y_pred_test),precision_score(y_test, y_pred_test), recall_score(y_test, y_pred_test),f1_score(y_test, y_pred_test)))


from sklearn.naive_bayes import MultinomialNB

#Create a Multinomial Naive Bayes model with default parameters 
model = MultinomialNB()

# Train the model using the training set
model.fit(X_train, y_train)

#Predict Output
y_pred = model.predict(X_test) # 0:Overcast, 2:Mild


#print("\n ************ Evaluation metrics on training data ***********\n") 

y_pred_train = model.predict(X_train)

#print('Training Accuracy: %.5f' % accuracy_score(y_train, y_pred_train))
#print('Training F1 Score: %.5f' % f1_score(y_train, y_pred_train))
#print('Training Precision Score: %.5f' % precision_score(y_train, y_pred_train))
#print('Training Recall Score: %.5f' % recall_score(y_train, y_pred_train))

y_pred_test = model.predict(X_test)

#print("\n ************ Evaluation metrics on test data ***********\n") 

#print('Testing Accuracy: %.5f' % accuracy_score(y_test, y_pred_test))
#print('Testing F1 Score: %.5f' % f1_score(y_test, y_pred_test))
#print('Testing Precision Score: %.5f' % precision_score(y_test, y_pred_test))
#print('Testing Recall Score: %.5f' % recall_score(y_test, y_pred_test))

print('%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f' % (accuracy_score(y_train, y_pred_train), precision_score(y_train, y_pred_train), recall_score(y_train, y_pred_train), f1_score(y_train, y_pred_train),accuracy_score(y_test, y_pred_test),precision_score(y_test, y_pred_test), recall_score(y_test, y_pred_test),f1_score(y_test, y_pred_test)))
