import pandas as pd

df = pd.read_csv("../dataset/Amazon_unlocked_Mobile.csv")

list_of_column_names = list(df.columns)

# displaying the list of column names
print('List of column names : ', list_of_column_names)

print(df['Reviews'].shape)

df = df.dropna(how='any')
df = df.drop(columns=['Price'])

df.to_csv(r'../dataset/removed_null_rows.csv', index=False)
list_of_column_names = list(df.columns)
print(df['Reviews'].shape)
