# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

entradas = np.array([7, 7, -2])
pesos = np.array([0.5, 0.1, 2])
 
def soma(e, p):
    return e.dot(p)
#dot product / produto escalar


s = soma(entradas, pesos)
         
def stepFunction (soma):
    if(soma >= 1):
        return 1
    return 0

r = stepFunction(s) 