import pandas as pd

previsoes = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

from sklearn.model_selection import train_test_split
previsoes_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsoes, 
                                                                                             classe, 
                                                                                             test_size = 0.25
                                                                                             #25% de todos os registros para fazer testes
                                                                                             )

import keras
from keras.models import Sequential
from keras.layers import Dense

#Criando a rede neural
classificador = Sequential()

#Adicionando a primeira camada oculta
classificador.add(Dense(units = 16, #Quantos neuronios, Dense = Full conection ou seja todos neuronios vao ser conectados entre todos eles
                        activation='relu', #funcao de ativacao
                        kernel_initializer='random_uniform', #Inicializacao dos pesos
                        input_dim=30, #Quantos elementos tem na camada de entrada, somente cadastrar para a primeira camada oculta
                        use_bias=True)) #Um neuronio a mais para a camada oculta chamada de 'bias'

#Adicionando mais uma camada oculta
classificador.add(Dense(units = 16,
                        activation='relu',
                        kernel_initializer='random_uniform'))

#Adicionando Camada de Saida
classificador.add(Dense(units= 1,
                        activation = 'sigmoid'
                        ))

#Calculando o erro

otimizador = keras.optimizers.Adam(lr = 0.0001,#Taxa de aprendizagem - Quanto menor valor mais chance de atingir o máximo global
                                   decay = 0.00006, #Valor que vai decair o learning rate de acordo com a epoca
                                   clipvalue = 0.5 #Evitar um "ping pong no gradiente, limita um lugar máximo que vai chegar a rede neural"
                                   )

classificador.compile(optimizer=  otimizador,#'adam', #Funcao de Erro
                      loss = 'binary_crossentropy', #Funcao de perda binary por que o resultado só retorna 0 e 1
                      metrics = ['binary_accuracy']) #Metricas para classificacao)
 

                     
##treinamento da rede neural
classificador.fit(previsoes_treinamento,
                  classe_treinamento, 
                  batch_size = 10,
                  epochs = 1000) #Calcula o erro para 10 registros e depois faz ajuste dos pesos) 
 
#Pegando os pesos de cada camada oculta
peso0 = classificador.layers[0].get_weights() #primeira camada 
pesos1 = classificador.layers[1].get_weights() #primeira camada oculta
pesos3 = classificador.layers[2].get_weights() #segunda camada oculta

##Retornando uma classificação real
previsoes = classificador.predict(previsores_teste)
previsoes = ( previsoes > 0.5)

#analisando a previsao da classificacao real
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)