import numpy as np

def sigmoid(soma):
    return 1 / (1.0 + np.exp(-soma))

def sigmoidDerivadaParcial(sig):
    return sig * (1 - sig)

entradas = np.array([ [0,0], [0,1], [1,0], [1,1] ])
saidas = np.array([ [0], [1], [1], [0] ])

#pesosCamadasOcultas = np.array([ [-0.424, -0.740, -0.961], [0.358, -0.577, -0.469] ])

#pesosCamadaSaida = np.array([ [-0.017], [-0.893], [0.148] ])

pesosCamadasOcultas = 2*np.random.random((2,3)) -1
pesosCamadaSaida = 2*np.random.random((3,1)) -1

taxaAprendizagem = 0.1
momento = 1

epocas = 100

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
    
    