
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMa
trixDisplay
df = pd.read_csv('C:/Users/dell/Desktop/Sentimental analysis ofCOVID-19 Tweets.csv')
df.head(100000)
df.info()
df.isnull().sum()
df.columns
text_df = df.drop([ 'sentiment'], axis = 1)
19
text_df.head()
print(text_df['tweets'].iloc[0],"\n")
print(text_df['tweets'].iloc[1],"\n")
print(text_df['tweets'].iloc[2],"\n")
print(text_df['tweets'].iloc[3],"\n")
print(text_df['tweets'].iloc[4],"\n")
print(text_df['tweets'].iloc[5],"\n")
#data processing
def data_processing(tweets):
  tweets = tweets.lower()
  tweets = re.sub(r'https\s+|www\s+https\s+', '', tweets, flags=re.MULTILINE)
  tweets = re.sub(r'\@W+|\#', '', tweets)
  tweets_tokens = word_tokenize(tweets)
  filtered_tweets = [w for w in tweets_tokens if not w in stop_words]
  return " ".join(filtered_tweets)
text_df['tweets'] = text_df['tweets'].apply(data_processing)
text_df = text_df.drop_duplicates('tweets')
def stemming(tweets):
  tweets_tokens = word_tokenize(tweets)
  stemmed_tweets = [porter.stem(word) for word in tweets_tokens]
  return " ".join(stemmed_tweets)
text_df['tweets'] = text_df['tweets'].apply(lambda x: stemming(x))
text_df.head()
print(text_df['tweets'].iloc[0],"\n")
print(text_df['tweets'].iloc[1],"\n")
print(text_df['tweets'].iloc[2],"\n")
print(text_df['tweets'].iloc[3],"\n")
print(text_df['tweets'].iloc[4],"\n")
print(text_df['tweets'].iloc[5],"\n")
text_df.info()
def polarity(tweets):
 return TextBlob(tweets).sentiment.polarity
text_df['polarity'] = text_df['tweets'].apply(polarity)
text_df.head(10)
def sentiment(label):
  if label <0:
    return "Negative"
  elif label ==0:
    return "Neutral" 
  elif label>0:
    return "positive"
text_df['sentiment'] = text_df['polarity'].apply(sentiment)
text_df.head(100)
fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment',data = text_df)
#Data Visualization
fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment',data = text_df)
fig = plt.figure(figsize=(7,7))
colors = ("yellowgreen","gold","red") 
wp ={'linewidth':2,'edgecolor':"black"} 
tags =text_df['sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie',autopct='%1.1f%%',shadow=True,colors=
colors,startangle =90,wedgeprops=wp,explode = explode,label='' )
plt.title('Distribution Of Sentiments')
pos_tweets = text_df[text_df.sentiment == 'positive']
pos_tweets = pos_tweets.sort_values(['polarity'],ascending=False)
pos_tweets.head()
text = ' '.join([word for word in pos_tweets['tweets']])
plt.figure(figsize=(20,15),facecolor='None') 
WordCloud =WordCloud(max_words=500,width=1600,height=800).generate(text)
plt.imshow(WordCloud,interpolation='bilinear')
plt.axis("off")
plt.title('Most Frequent words in possitive tweets',fontsize=19)
plt.show()
neg_tweets = text_df[text_df.sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(by=['polarity'], ascending=False)
print(neg_tweets.head())
from wordcloud import WordCloud
text = ' '.join([word for word in neg_tweets['tweets']])
plt.figure(figsize=(20,15),facecolor='None') 
wordcloud =WordCloud(max_words=500,width=1600,height=800).generate(text)
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.title('Most Frequent words in negative tweets',fontsize=19)
plt.show()
#data training
vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['tweets'])
feature_names= vect.get_feature_names_out()
print("Number Of Features:{}\n".format(len(feature_names)))
print("First 20 features:\n{}".format(feature_names[:20]))
X = text_df['tweets']
Y = text_df['sentiment']
X = vect.transform(X)
x_train, x_test, y_train, y_test =train_test_split(X,Y,test_size=0.2,random_state=42)
print("size ofx_train:",(x_train.shape))
print("size of y_train:",(y_train.shape))
print("size of x_test:",(y_train.shape))
print("size of y_test:",(y_test.shape))
import warnings
warnings.filterwarnings('ignore')
#logistical Regression
logreg = LogisticRegression()
logreg.fit(x_train,y_train) 
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred,y_test)
print("Test accuracy:{:.2f}%".format(logreg_acc*100))
print(confusion_matrix(y_test,logreg_pred))
print("\n")
print(classification_report(y_test,logreg_pred))
style.use('classic')
cm = confusion_matrix(y_test,logreg_pred,labels=logreg.classes_) 
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=logre.classes_)
disp.plot()
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.001,0.01,0.1,10]}
grid = GridSearchCV(LogisticRegression(),param_grid)
grid.fit(x_train,y_train)
print("Best parameters:",grid.best_params_)
y_pred = grid.predict(x_test)
logreg_acc = accuracy_score(y_pred,y_test)
print("Test accuracy:{:.2f}%".format(logreg_acc*100))
print(confusion_matrix(y_test,logreg_pred)) 
print("\n")
print(classification_report(y_test,logreg_pred))
#svc model
from sklearn.svm import LinearSVC
SVCmodel = LinearSVC()
SVCmodel.fit(x_train,y_train)
svc_pred = SVCmodel.predict(x_test)
svc_acc = accuracy_score(svc_pred,y_test)
print("testaccuracy:{:.2f}%".format(svc_acc*100))
print(confusion_matrix(y_test,svc_pred))
print("\n")
print(classification_report(y_test,svc_pred))
grid = {
'c':[0.01,0.1,1,10],
'kernel':["linear","poly","rbf","sigmoid"],'degree':[1,3,5,7],'gamma ':[0.01,1]
}
grid = GridSearchCV(SVCmodel,param_grid)
grid.fit(x_train,y_train)
print("Best parameter:",grid.best_params_)
y_pred = grid.predict(x_test)
logreg_acc = accuracy_score(y_pred,y_test)
print("Test accuracy:{:.2f}%".format(logreg_acc*100))
print(confusion_matrix(y_test,y_pred))
print("\n")
print(classification_report(y_test,y_pred))
