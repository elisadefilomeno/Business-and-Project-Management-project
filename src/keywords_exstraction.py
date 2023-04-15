import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from os import path
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

dataset=pd.read_csv("../dataset/final_dataset_all_columns.csv")

#dataset
dataset_sam = dataset[dataset["Brand Name"] == 'Apple']

dataset_sam_neg = dataset_sam[dataset_sam["Rating"] == 'negative']

dataset_sam_pos = dataset_sam[dataset_sam["Rating"] == 'positive']


corpus=[]
def clean(x):
    # Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', x)

    # Convert to lowercase
    text = text.lower()

    # remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    # remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    ##Convert to list from string
    text = text.split()

    # Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in
                                                        stop_words]
    text = " ".join(text)
    corpus.append(text)
    return text

def build_graph():
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_words,
        max_words=100,
        max_font_size=50,
        random_state=42
    ).generate(str(corpus))
    print(wordcloud)
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    # fig.savefig("word1.png", dpi=900)

stop_words = set(stopwords.words("english"))

dataset_sam_neg['Reviews']=dataset_sam_neg['Reviews'].apply(clean)

#Identify common words
freq = pd.Series(' '.join(dataset_sam_neg['Reviews']).split()).value_counts()[:20]


#build_graph()

def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1],
                reverse=True)
    return words_freq[:n]


top2_words = get_top_n2_words(corpus, n=20)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)
#Barplot of most freq Bi-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)

plt.title("commenti negativi")
plt.show()

corpus=[]

dataset_sam_pos['Reviews']=dataset_sam_pos['Reviews'].apply(clean)
freq = pd.Series(' '.join(dataset_sam_pos['Reviews']).split()).value_counts()[:20]


top2_words = get_top_n2_words(corpus, n=20)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
print(top2_df)
#Barplot of most freq Bi-grams
import seaborn as sns
sns.set(rc={'figure.figsize':(13,8)})
h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
h.set_xticklabels(h.get_xticklabels(), rotation=45)

plt.title("commenti positivi")
plt.show()
