import numpy as np
from sklearn import datasets


def sigmoid(soma):
    return 1 / (1.0 + np.exp(-soma))

def sigmoidDerivadaParcial(sig):
    return sig * (1 - sig)

base = datasets.load_breast_cancer()

entradas = base.data
valoresSaidas = base.target
saidas = np.empty([569, 1], dtype=int)
for i in range(569):
    saidas[i] = valoresSaidas[i]

pesosCamadasOcultas = 2*np.random.random((30,569)) -1
pesosCamadaSaida = 2*np.random.random((569,1)) -1

taxaAprendizagem = 0.3
momento = 1

epocas = 10000

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesosCamadasOcultas)
    camadaOculta =sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesosCamadaSaida)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = (saidas - camadaSaida)
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro: " + str(mediaAbsoluta))
    
    derivadaParcialSaida = sigmoidDerivadaParcial(camadaSaida)
    DeltaCamadaSaida = erroCamadaSaida * derivadaParcialSaida
    
    pesosSaidaTransposta = pesosCamadaSaida.T
    deltaSaidaXPeso = DeltaCamadaSaida.dot(pesosSaidaTransposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivadaParcial(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T #Transposta = transforma linha em coluna e coluna em linha
    pesosNovos1 = camadaOcultaTransposta.dot(DeltaCamadaSaida)
    pesosCamadaSaida = (pesosCamadaSaida * momento) + (pesosNovos1 * taxaAprendizagem)

    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesosCamadasOcultas = (pesosCamadasOcultas * momento) + (pesosNovo0 * taxaAprendizagem)
    
    