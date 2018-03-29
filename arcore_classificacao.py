import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

movies = pd.read_csv('avaliacoes_usuario.csv')
caracteristicas = movies[movies.columns[1:16]]
gostos = movies[movies.columns[16:]]

treino, teste, treino_marcacoes, teste_marcacoes = train_test_split(caracteristicas, gostos)

treino = np.array(treino).reshape(len(treino), 15)
teste = np.array(teste).reshape(len(teste), 15)
treino_marcacoes = np.array(treino_marcacoes).reshape(len(treino_marcacoes), 1)
teste_marcacoes = np.array(teste_marcacoes).reshape(len(teste_marcacoes), 1)

modelo = tree.DecisionTreeClassifier(max_depth = 5)
modelo.fit(treino, treino_marcacoes)

previsoes = modelo.predict(teste)
acuracia = accuracy_score(teste_marcacoes, previsoes)