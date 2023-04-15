import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.pipeline import Pipeline
from nltk import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

class LemmaTokenizer:
    def __init__(self):
         self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        from nltk.corpus import stopwords
        stopwords = stopwords.words('english')
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t is not stopwords]

training_dataset=pd.read_csv("../dataset/trainingset.csv")

X = training_dataset['Reviews'].values  #text
y = training_dataset['Rating'].values  #sentiment

train, test = train_test_split(training_dataset, test_size=0.2)

x_train=train['Reviews']
y_train=train['Rating']

x_test=test['Reviews']
y_test=test['Rating']

target_names = ['negative', 'neutral', 'positive']
print("Number of comments:", len(X))

#Pipeline Classifier3
linearsvc_clf = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(), strip_accents='ascii', lowercase=True,
                             token_pattern=r"(?u)\b[^\d\W][^\d\W]+\b", max_features=3000)),
    ('tfidf', TfidfTransformer()),
    ('clf', svm.LinearSVC(C=0.1)),
])

linearsvc_clf.fit(x_train, y_train)
print("ending clf training")


final_dataset=pd.read_csv("../dataset/removed_null_rows.csv")
X_test = final_dataset['Reviews'].values
print("starting sentiment prediction")

final_dataset['sentiment']=linearsvc_clf.predict(X_test)
final_dataset.to_csv(r'../dataset/predicted_all_comments.csv', index=False)
