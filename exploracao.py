import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression

movies = pd.read_csv("filmes.csv")

x = movies['Investimento (em milhoes)']
y = movies['Bilheteria (pessoas)']


#Faz o grafico com os pontos
plt.scatter(x, y)
#Mostra o grafico
plt.show()

sample = movies.sample(n=200)
x = sample['Investimento (em milhoes)']
y = sample['Bilheteria (pessoas)']

plt.scatter(x, y)
plt.show()	 

filmes_investimento = movies['Investimento (em milhoes)']
filmes_bilheteria = movies['Bilheteria (pessoas)']

treino_investimento, teste_investimento, treino_bilheteria, teste_bilheteria = train_test_split(filmes_investimento, filmes_bilheteria)

treino_investimento = np.array(treino_investimento).reshape(len(treino_investimento), 1)
teste_investimento = np.array(teste_investimento).reshape(len(teste_investimento), 1)
treino_bilheteria = np.array(treino_bilheteria).reshape(len(treino_bilheteria), 1)
teste_bilheteria = np.array(teste_bilheteria).reshape(len(teste_bilheteria), 1)

modelo = LinearRegression()
modelo.fit(treino_investimento, treino_bilheteria)

#modelo.intercept_ devolve o valor da constante independente da funcao linear
print("Constante independente: " +str(modelo.intercept_))
#modelo.coef_ retorna o valor da constante que multiplica x na funcao linear
print("Constante que multiplica x: " +str(modelo.coef_))

print("Predict Bilheteria Zootopia: " +str(modelo.predict(27.74456356)))

print(modelo.score(treino_investimento, treino_bilheteria))
print(modelo.score(teste_investimento, teste_bilheteria))