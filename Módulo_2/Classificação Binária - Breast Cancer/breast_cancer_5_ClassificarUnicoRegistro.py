import pandas as pd 
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

#Criando a rede neural
classificador = Sequential()

#Adicionando a primeira camada oculta
classificador.add(Dense(units = 8, #Quantos neuronios, Dense = Full conection ou seja todos neuronios vao ser conectados entre todos eles
                        activation = 'relu', #funcao de ativacao
                        kernel_initializer = 'normal', #Inicializacao dos pesos
                        input_dim=30, #Quantos elementos tem na camada de entrada, somente cadastrar para a primeira camada oculta
                        use_bias=True)) #Um neuronio a mais para a camada oculta chamada de 'bias'
   
#Adicionando dropout para evitar Overfiting -- Valores recomendados de 20 a 30 %
classificador.add(Dropout(0.25)) #20%

#Adicionando mais uma camada oculta
classificador.add(Dense(units = 8,
                        activation = 'relu',
                        kernel_initializer = 'normal'))
classificador.add(Dropout(0.2)) #20%

#Adicionando Camada de Saida
classificador.add(Dense(units= 1,
                        activation = 'sigmoid'
                        ))

classificador.compile(optimizer = 'adam', #'adam', #Funcao de Erro
                      loss = 'binary_crossentropy', #Funcao de perda binary por que o resultado só retorna 0 e 1
                      metrics = ['binary_accuracy']) #Metricas para classificacao)

classificador.fit(previsores, classe, batch_size = 30, epochs = 100)

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 0.05, 
                  1098, 0.97, 4500, 145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007, 23.15,
                  16.64, 178.5, 2018, 0.14, 0.185, 0.84, 158, 0.363]]) #valores inseridos pelo usuário 

previsao = classificador.predict(novo) #previsao final para o usuário
previsao = (previsao > 0.5)