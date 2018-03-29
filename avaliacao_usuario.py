import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

movies = pd.read_csv('avaliacoes_usuario.csv')

Counter(movies['Gostou'])

caracteristicas = movies[movies.columns[1:16]]
gostos = movies[movies.columns[16:]]

treino, teste, treino_marcacoes, teste_marcacoes = train_test_split(caracteristicas, gostos)

treino = np.array(treino).reshape(len(treino), 15)
teste = np.array(teste).reshape(len(teste), 15)

treino_marcacoes = treino_marcacoes.values.ravel()
teste_marcacoes = teste_marcacoes.values.ravel()

modelo = LogisticRegression()
modelo.fit(treino, treino_marcacoes)

previsoes = modelo.predict(teste)

acuracia = accuracy_score(teste_marcacoes, previsoes)

modeloNB = MultinomialNB()
modeloNB.fit(treino, treino_marcacoes)
previsoesNB = modeloNB.predict(teste)
acuraciaNB = accuracy_score(teste_marcacoes, previsoesNB)