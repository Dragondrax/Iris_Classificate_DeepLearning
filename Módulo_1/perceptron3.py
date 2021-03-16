# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 23:16:10 2021

@author: IgorS
"""

import numpy as np

#and
#entradas = np.array([ [0,0], [0,1], [1,0], [1,1] ])
#saidas = np.array([0, 0, 0, 1])

#or
entradas = np.array([ [0,0], [0,1], [1,0], [1,1] ])
saidas = np.array([0, 1, 1, 1])

#XOR
#entradas = np.array([ [0,0], [0,1], [1,0], [1,1] ])
#saidas = np.array([0, 1, 1, 0])


pesos = np.array([0.0, 0.0])
taxaAprendizagem = 0.1

def stepFunction(soma):
    if(soma >= 1):
        return 1
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepFunction(s)

def treinar():
    erroTotal = 1
    while(erroTotal != 0):
        erroTotal =  0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)#abs = o código não fica negativo
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print('peso atualizado: ' + str(pesos[j]))
        print('total de erros: '+str(erroTotal))
        
treinar()

print('rede neural treinada')

