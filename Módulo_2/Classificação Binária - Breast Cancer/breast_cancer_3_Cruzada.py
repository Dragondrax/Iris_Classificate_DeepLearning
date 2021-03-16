import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import  KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRedeNeural():
    #Criando a rede neural
    classificador = Sequential()
    
    #Adicionando a primeira camada oculta
    classificador.add(Dense(units = 16, #Quantos neuronios, Dense = Full conection ou seja todos neuronios vao ser conectados entre todos eles
                            activation='relu', #funcao de ativacao
                            kernel_initializer='random_uniform', #Inicializacao dos pesos
                            input_dim=30, #Quantos elementos tem na camada de entrada, somente cadastrar para a primeira camada oculta
                            use_bias=True)) #Um neuronio a mais para a camada oculta chamada de 'bias'
   
    #Adicionando dropout para evitar Overfiting -- Valores recomendados de 20 a 30 %
    classificador.add(Dropout(0.2)) #20%
    
    #Adicionando mais uma camada oculta
    classificador.add(Dense(units = 16,
                            activation='relu',
                            kernel_initializer='random_uniform'))
    classificador.add(Dropout(0.2)) #20%
    
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
     
    return classificador

classificador = KerasClassifier(build_fn = criarRedeNeural, #criar rede neural 
                                epochs = 500, #epocas
                                batch_size = 10) 

#Fazendo validacao cruzada
resultados = cross_val_score(estimator = classificador,
                             X = previsores, #Atributos Previsores
                             y = classe,
                             cv = 10, #Quantas vezes realizar o teste
                             scoring = 'accuracy' #retorno do resultado
                             )

#Calculando media de porcentagem de acerto
media = resultados.mean()

#Desvio padrao (Variacao de cada valor perante a media)
desvio = resultados.std() #Quanto menor melhor

