import pandas as pd
import numpy as np
from pymorphy2 import tokenizers
import matplotlib.pyplot as plt

df_train = pd.read_csv('data/news_train.txt',sep='\t', header = None)
df_train.head(2)

df_train['len_text'] = df_train[2].apply(len)
df_train['tokens'] = df_train[2].apply(lambda x: tokenizers.simple_word_tokenize(x))
df_train['num_of_words'] = df_train['tokens'].apply(len)
df_train.head(2)

plt.plot(df_train.len_text, df_train.num_of_words)

