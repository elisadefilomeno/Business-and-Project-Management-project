import pandas as pd
import sklearn

df_input = pd.read_csv("C:/Users/ireca/Desktop/dataset/modified_comments.csv")

# considero solo i commenti con rating 1,3 e 5 perchè presenteranno differenze più nette
df_output=df_input.loc[df_input['Rating'].isin([1,3,5])]

#al rating 1 viene assegnato il sentimento negativo
#al rating 3 viene assegnato il sentimento neutrale
#al rating 5 viene assegnato il sentimento positivo
df_output['Rating']=df_output['Rating'].replace(1,'negative')
df_output['Rating']=df_output['Rating'].replace(3,'neutral')
df_output['Rating']=df_output['Rating'].replace(5,'positive')

#costruisco un dataset bilanciato

#costruisco un dataframe contenete solo commenti negativo e di questi ne seleziono solo 5000
dataset_negative = df_output.loc[df_output['Rating']=='negative']
#riordino casualmente il dataset per non prendere i commenti in sequenza
df_shuffled=sklearn.utils.shuffle(dataset_negative)
dataset_negative_final = df_shuffled[:5000]

#costruisco un dataframe contenete solo commenti positivi e di questi ne seleziono solo 5000
dataset_positive = df_output.loc[df_output['Rating']=='positive']
df_shuffled=sklearn.utils.shuffle(dataset_positive)
dataset_positive_final = df_shuffled[:5000]

#costruisco un dataframe contenete solo commenti neutrali e di questi ne seleziono solo 5000
dataset_neutral = df_output.loc[df_output['Rating']=='neutral']
df_shuffled=sklearn.utils.shuffle(dataset_neutral)
dataset_neutral_final = df_shuffled[:5000]

#unisco i tre dataframe per realizzarne uno unico con tutti i setimenti
df_output=pd.merge(dataset_negative_final, dataset_positive_final, how='outer')
df_output=pd.merge(df_output, dataset_neutral_final, how='outer')

#seleziono solo le colonne di interesse per classificare

df_output.to_csv(r'../dataset/final_dataset_all_columns.csv', index=False)

df_output=df_output[['Reviews', 'Rating']]

#salvo dataset
df_output.to_csv(r'../dataset/trainingset.csv', index=False)