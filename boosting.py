from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

movies = pd.read_csv('avaliacoes_usuario.csv')
caracteristicas = movies[movies.columns[1:16]]
gostos = movies[movies.columns[16:]]

treino, teste, treino_marcacoes, teste_marcacoes = train_test_split(caracteristicas, gostos)

treino = np.array(treino).reshape(len(treino), 15)
teste = np.array(teste).reshape(len(teste), 15)
treino_marcacoes = np.array(treino_marcacoes).reshape(len(treino_marcacoes), 1)
teste_marcacoes = np.array(teste_marcacoes).reshape(len(teste_marcacoes), 1)

#Pensando em Regressão - Pegando média
modeloAdaRegressor = AdaBoostRegressor()
modeloAdaRegressor.fit(treino, treino_marcacoes.ravel())
modeloAdaRegressor.score(treino, treino_marcacoes)
modeloAdaRegressor.score(teste, teste_marcacoes)

#Pensando em Regressão - Pegando média
modeloGradientRegressor = GradientBoostingRegressor()
modeloGradientAdaRegressor.fit(treino, treino_marcacoes.ravel())
modeloGradientAdaRegressor.score(treino, treino_marcacoes)
modeloGradientAdaRegressor.score(teste, teste_marcacoes)

#Pensando em Classificacao - Pegando maioritario
modeloAdaClassifier = AdaBoostClassifier()
modeloAdaClassifier.fit(treino, treino_marcacoes)
previsoesAda = modeloAdaClassifier.predict(teste)
acuraciaGradiente = accuracy_score(teste_marcacoes, previsoes)

#Pensando em Classificacao - Pegando maioritario
modeloGradienteClassifier = GradientBoostingClassifier()
modeloGradienteClassifier.fit(treino, treino_marcacoes)
previsoesGradiente = modeloGradienteClassifier.predict(teste)
acuraciaGradiente = accuracy_score(teste_marcacoes, previsoes)

#Pensando em Classificacao - Pegando maioritario
modeloLogisticRegression = LogisticRegression()
modeloLogisticRegression.fit(treino, treino_marcacoes)
previsoesLogisticRegression = modeloLogisticRegression.predict(teste)
acuraciaLogisticRegression = accuracy_score(teste_marcacoes, previsoes)

#Pensando em Classificacao - Pegando maioritario
modeloRandomForest = RandomForestClassifier()
modeloRandomForest.fit(treino, treino_marcacoes)
previsoesRandomForest = modeloRandomForest.predict(teste)
acuraciaRandomForest = accuracy_score(teste_marcacoes, previsoes)