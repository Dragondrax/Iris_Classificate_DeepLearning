# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

entradas = [7, 7, -2]
pesos = [0.5, 0.1, 2]
 
def soma(e, p):
    s = 0 
    for i in range(3):
        #print(entradas[i])
        #print(pesos[i])
        s += e[i] * p[i]
    return s

s = soma(entradas, pesos)
         
def stepFunction (soma):
    if(soma >= 1):
        return 1
    return 0

r = stepFunction(s)

