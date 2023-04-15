from langdetect import detect
import pandas as pd
import numpy as np

df = pd.read_csv("../dataset/removed_emoji.csv")


def detect_language(x):
    try:
        return detect(x)
    except:
        return np.nan


print(df['Reviews'].shape)

df['language']=df.Reviews.apply(detect_language)
df = df[df.language.eq('en')]
df = df.drop(columns=['language'])

print(df['Reviews'].shape)

df.to_csv(r'../dataset/english_comments.csv', index=False)
