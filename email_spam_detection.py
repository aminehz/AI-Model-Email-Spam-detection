
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_curve, auc

df = pd.read_csv('emails.csv')
df.head()

df.shape

df.columns

# verify the duplicates and remove them
df.drop_duplicates(inplace=True)
print(df.shape)

#Number of missing data for each column
print(df.isnull().sum())

#download the stopwords package
nltk.download('stopwords')

#Clean the Text
def process(text):
  nopunc= [char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)
  clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
  return clean
#show the tokenization
df['text'].head().apply(process)

#Convert the text into a matrix of token counts
message = CountVectorizer(analyzer=process).fit_transform(df['text'])

xtrain, xtest, ytrain, ytest = train_test_split(message, df['spam'], test_size=0.20)
#see the shape of the data
print(message.shape)

#create and train the Naive Bayes Classifier
classifier = MultinomialNB().fit(xtrain,ytrain)

#See the classifiers prediction and actual values on the dataset
print(classifier.predict(xtrain))
print(ytrain.values)

#Evaluating the Model on the training data set
pred = classifier.predict(xtrain)
print(classification_report(ytrain, pred))
print()
print("Confusion Matrix:\n", confusion_matrix(ytrain, pred))
print("Accuracy: \n", accuracy_score(ytrain, pred))

#print the predictions
print(classifier.predict(xtest))
#print the actual values
print(ytest.values)

#Evaluate the model on the test data set
pred = classifier.predict(xtest)
print(classification_report(ytest, pred))
print()
print("confusion Matrix: \n", confusion_matrix(ytest, pred))
print("Accuracy:\n", accuracy_score(ytest, pred))

# Plot confusion matrix for test data
plt.figure(figsize=(10, 7))
conf_matrix = confusion_matrix(ytest, pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Compute ROC Curve and AUC
y_prob = classifier.predict_proba(xtest)[:, 1]  # Get probability estimates for the positive class
fpr, tpr, _ = roc_curve(ytest, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

