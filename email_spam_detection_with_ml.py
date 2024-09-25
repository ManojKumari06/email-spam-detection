# -*- coding: utf-8 -*-


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
import time

from google.colab import drive
drive.mount('/content/drive')

"""# Data Loading
---
"""

df = pd.read_csv('/content/drive/My Drive/Oasis Infobyte/Data Science - Internship/Email-Spam-Detection/spam.csv', encoding='latin-1')

df

df.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)

df.head()

df.tail()

"""# EDA
---

# 1. Handling Null Values
"""

df.isna().any()

df.isna().sum()

"""# 2. Handling Duplicate Values"""

df['v2'].nunique()

df.shape

df['v2'].drop_duplicates(inplace = True)

df.shape

df

"""# 3. Class Distributions"""

# Create a bar plot of the class distribution
class_counts = df['v1'].value_counts()
class_counts.plot(kind='bar')
plt.title('Class Distribution of Spam/Ham')
plt.xlabel('Spam/Ham')
plt.ylabel('Number of Mails')
plt.show()

"""# Word Count"""

from collections import Counter
import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Concatenate all tweet texts into a single string
all_text = ' '.join(df['v2'].values)
# Remove URLs, mentions, and hashtags from the text
all_text = re.sub(r'http\S+', '', all_text)
all_text = re.sub(r'@\S+', '', all_text)
all_text = re.sub(r'#\S+', '', all_text)

# Split the text into individual words
words = all_text.split()

# Remove stop words
stop_words = set(stopwords.words('english'))
words = [word for word in words if not word in stop_words]

# Count the frequency of each word
word_counts = Counter(words)
top_words = word_counts.most_common(100)
top_words

# Create a bar chart of the most common words
top_words = word_counts.most_common(10) # Change the number to show more/less words
x_values = [word[0] for word in top_words]
y_values = [word[1] for word in top_words]
plt.bar(x_values, y_values)
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Most Commonly Used Words')
plt.show()

"""# Natural Language Processing
---

# 1. Data Cleaning
"""

# Clean the data
def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    # Tokenize the text
    words = nltk.word_tokenize(text)
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    # Stem the words
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    # Join the words back into a string
    text = ' '.join(words)
    return text

import nltk
nltk.download('punkt')

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# tqdm.pandas()
# 
# df['cleaned_text'] = df['v2'].progress_apply(clean_text)

"""# 2. Feature Extraction"""

# Create the Bag of Words model
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(df['cleaned_text']).toarray()
y = df['v1']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# Classification Model
---

# 1. Logistic Regression Model
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# train a Logistic Regression Model
clf = LogisticRegression()

clf.fit(X_train, y_train)

"""# 2. Predictions"""

# evaluate the classifier on the test set
y_pred = clf.predict(X_test)

y_pred

"""# 3. Accuracy"""

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

"""# 4. Confusion Matrix"""

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

import seaborn as sns
sns.heatmap(cm, annot=True)

cm

"""# 5. Classification Report"""

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)