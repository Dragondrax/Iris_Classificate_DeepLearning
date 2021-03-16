import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import  KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRedeNeural(optimizer, loos, kernel_initializer, activation, neurons, use_bias):
    #Criando a rede neural
    classificador = Sequential()
    
    #Adicionando a primeira camada oculta
    classificador.add(Dense(units = neurons, #Quantos neuronios, Dense = Full conection ou seja todos neuronios vao ser conectados entre todos eles
                            activation = activation, #funcao de ativacao
                            kernel_initializer = kernel_initializer, #Inicializacao dos pesos
                            input_dim=30, #Quantos elementos tem na camada de entrada, somente cadastrar para a primeira camada oculta
                            use_bias=use_bias)) #Um neuronio a mais para a camada oculta chamada de 'bias'
   
    #Adicionando dropout para evitar Overfiting -- Valores recomendados de 20 a 30 %
    classificador.add(Dropout(0.25)) #20%
    
    #Adicionando mais uma camada oculta
    classificador.add(Dense(units = neurons,
                            activation = activation,
                            kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2)) #20%
    
    #Adicionando Camada de Saida
    classificador.add(Dense(units= 1,
                            activation = 'sigmoid'
                            ))
    
    classificador.compile(optimizer=  optimizer, #'adam', #Funcao de Erro
                          loss = loos, #Funcao de perda binary por que o resultado só retorna 0 e 1
                          metrics = ['binary_accuracy']) #Metricas para classificacao)
     
    return classificador


classificador = KerasClassifier(build_fn = criarRedeNeural)

#testando diversas parametros para a rede neural tiver o melhor desempenho
parametros = {'batch_size': [10, 30], 
              'epochs': [100,500,1000],
              'optimizer': ['adam', 'sgd'],
              'loos': ['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 24, 32],
              'use_bias': ['True', 'False']}

#Fazer a busca efetivamente do melhor parametro

grid_search = GridSearchCV(estimator = classificador,
                           param_grid=parametros,
                           scoring= 'accuracy', 
                           cv = 10) #validacao cruzada

grid_search = grid_search.fit(previsores, classe)


melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

#Executar uma vez só para pegar os melhores valores e montar a rede neural final com os valores encontrados