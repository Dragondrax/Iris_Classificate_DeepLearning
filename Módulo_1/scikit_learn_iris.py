from  sklearn import datasets
from sklearn.neural_network import MLPClassifier

iris = datasets.load_iris()
entradas = iris.data
saidas = iris.target

redeNeural = MLPClassifier(verbose=True,
                           max_iter=10000,
                           tol=0.00100,
                           #activation='logistic',
                           learning_rate_init=0.001)

redeNeural.fit(entradas, saidas)
redeNeural.predict([[5, 7.2, 5.1, 2.2]])