import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from mlxtend.evaluate import paired_ttest_kfold_cv


class LemmaTokenizer:
    def __init__(self):
         self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        from nltk.corpus import stopwords
        stopwords = stopwords.words('english')
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t is not stopwords]


df = pd.read_csv("../dataset/trainingset.csv")

X = df['Reviews'].values  #text
y = df['Rating'].values  #sentiment

train, test = train_test_split(df, test_size=0.2)

x_train = train['Reviews']
y_train = train['Rating']

x_test = test['Reviews']
y_test = test['Rating']

target_names = ['negative', 'neutral', 'positive']
print("Number of comments:", len(X))


#Pipeline Classifier1
text_clf = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(), strip_accents='ascii', max_features=3000)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(x_train, y_train)

#calculating accuracies in cross-valudation
scores = cross_val_score(text_clf, X, y, cv=5)
print("Accuracy MultinomialNB : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# prediction in cross-validation
predicted = cross_val_predict(text_clf, X, y, cv=5)

print(metrics.classification_report(y, predicted,
                                    target_names=target_names))  # metrics extractions (precision    recall  f1-score   support)
print(metrics.confusion_matrix(y, predicted))

ConfusionMatrixDisplay.from_predictions(y, predicted)
plt.title("Multinominal Naive Bayes Confusion Matrix")
plt.show()


#Pipeline Classifier2
text_clf2 = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(), strip_accents='ascii', max_features=3000)),
    ('tfidf', TfidfTransformer()),
    ('feat_sel', SelectKBest(chi2,k=100)),
    ('clf', tree.DecisionTreeClassifier(max_leaf_nodes=30)),
])

text_clf2.fit(x_train,y_train)

#calculating accuracies in cross-valudation
scores2 = cross_val_score(text_clf2, X, y, cv=5)
print("Accuracy Decision Tree : %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

# prediction in cross-validation
predicted = cross_val_predict(text_clf2, X, y, cv=5)

print(metrics.classification_report(y, predicted,
                                    target_names=target_names))  # metrics extractions (precision    recall  f1-score   support)
print(metrics.confusion_matrix(y, predicted))
ConfusionMatrixDisplay.from_predictions(y,predicted)
plt.title("Decision Tree Confusion Matrix")
plt.show()


#Pipeline Classifier3
text_clf3 = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(), strip_accents='ascii', max_features=3000)),
    ('tfidf', TfidfTransformer()),
    ('clf', svm.LinearSVC(C=0.1)),
])

text_clf3.fit(x_train,y_train)

#calculating accuracies in cross-valudation
scores3 = cross_val_score(text_clf3, X, y, cv=5)
print("Accuracy SVM : %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))

# prediction in cross-validation
predicted = cross_val_predict(text_clf3, X, y, cv=5)

print(metrics.classification_report(y, predicted,
                                    target_names=target_names))  # metrics extractions (precision    recall  f1-score   support)
print(metrics.confusion_matrix(y, predicted))
ConfusionMatrixDisplay.from_predictions(y,predicted)
plt.title("Linear SVC Confusion Matrix")
plt.show()


#Pipeline Classifier4
text_clf4 = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(), strip_accents='ascii', max_features=3000)),
    ('tfidf', TfidfTransformer()),
    ('clf', KNeighborsClassifier(n_neighbors=40, weights='distance',n_jobs=-1)),
])

text_clf4.fit(x_train,y_train)

#calculating accuracies in cross-valudation
scores4 = cross_val_score(text_clf4, X, y, cv=5)
print("Accuracy KNN : %0.2f (+/- %0.2f)" % (scores4.mean(), scores4.std() * 2))

# prediction in cross-validation
predicted = cross_val_predict(text_clf4, X, y, cv=5)

print(metrics.classification_report(y, predicted,target_names=target_names)) # metrics extractions (precision recall f1-score support)
print(metrics.confusion_matrix(y, predicted))

ConfusionMatrixDisplay.from_predictions(y, predicted)
plt.title("KNeighborsClassifier Confusion Matrix")
plt.show()

#Pipeline Classifier5
text_clf5 = Pipeline([
    ('vect', CountVectorizer(tokenizer=LemmaTokenizer(), strip_accents='ascii', max_features=3000)),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier()),
])

text_clf5.fit(x_train,y_train)

#calculating accuracies in cross-valudation
scores5 = cross_val_score(text_clf5, X, y, cv=5)
print("Accuracy RandomForest : %0.2f (+/- %0.2f)" % (scores5.mean(), scores5.std() * 2))

# prediction in cross-validation
predicted = cross_val_predict(text_clf5, X, y, cv=5)

print(metrics.classification_report(y, predicted,target_names=target_names)) # metrics extractions (precision recall f1-score support)
print(metrics.confusion_matrix(y, predicted))

ConfusionMatrixDisplay.from_predictions(y, predicted)
plt.title("Random Forest Confusion Matrix")
plt.show()

# t_test between the two best classifiers
t, p = paired_ttest_kfold_cv(estimator1=text_clf3, estimator2=text_clf5, X=X, y=y, cv=5, random_seed=1)
print('p value: %.3f' % p)
