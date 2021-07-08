from scipy import signal
import numpy as np
import pandas as pd
import CalculTrasaturiSemnale
from ExtragereDate import CANALE_SELECTATE, selecteazaSubtrialDe6sec
subtrial = 768
valoriPSD=[]
valoriRMS=[]
valoriSampleEntropy=[]
valoriStdDev=[]
def aplicaChebyshevInvers(frecventa_low, frecventa_high, fs, order=5):
    frecventa_nyq = 0.5*fs
    low = frecventa_low / frecventa_nyq
    high = frecventa_high / frecventa_nyq
    b, a = signal.cheby2(order, 3, [low, high], btype='bandpass')
    return b, a
def aplicaFiltruChebyshevInvers(data, frecventa_low, frecventa_high, fs, order=5):
    semnal_eeg = np.array(data)
    b, a = aplicaChebyshevInvers(frecventa_low, frecventa_high, fs, order=order)
    semnal_filtrat = signal.lfilter(b, a, semnal_eeg)
    return semnal_filtrat
def verificaTipBanda(tipBanda):
    fs = 128
    if tipBanda == "Alpha":
        frecventa_low = 7.5
        frecventa_high = 12.5
    elif tipBanda == "Beta":
        frecventa_low = 13.0
        frecventa_high = 30.0
    else:
        frecventa_low = 30.0
        frecventa_high = 63.5
    return frecventa_low, frecventa_high, fs
def impartireBenziFrecventa(subiect, experiment):
    print("Impartire pe benzi de frecventa si calcul de trasaturi pentru subiectul  "+str(subiect))
    valoriTrasaturi = []
    fisierTrasaturiCSV = open('CalculTrasaturi/Trasaturi_semnale_EEG.csv', 'a')
    fisierTrasaturiCSV.write("\n")
    for tipBanda in ['Alpha', 'Beta', 'Gamma']:
        print(tipBanda)
        # Se verifica tipul benzii si se returneaza valorile low, high si fs ale frecventei
        frecventa_low, frecventa_high, fs = verificaTipBanda(tipBanda)
        #Se scrie in fisierele corespunzatoare benzilor
        fisierValoriBenziPerUser = open('ImpartirePeBenziDeFrecventa/ValoriBanda'+tipBanda+'.dat', 'a')
        fisierValoriBenziPerUser.write("\n Subiect " + subiect.__str__()+"\n")
        # Se selecteaza un subtrial de 6 secunde
        subtrial = selecteazaSubtrialDe6sec(subiect=subiect, experiment=experiment, splits=10)
        for canal in CANALE_SELECTATE:
            valoriBanda = []
            column = subtrial[canal]
            dateFiltrate = aplicaFiltruChebyshevInvers(column, frecventa_low, frecventa_high, fs)
            for val in dateFiltrate:
                valoriBanda.append(round(val, 4))
            # Se scrie in fisierele corespunzatoare benzilor
            fisierValoriBenziPerUser.write("\nCanal "+canal+"\n")
            fisierValoriBenziPerUser.write(valoriBanda.__str__() + "\n")
            # Se calculeaza cele 4 trasaturi
            trasaturi = CalculTrasaturiSemnale.calculeazaTrasaturiBanda(valoare=valoriBanda, low=frecventa_low, high=frecventa_high)
            #psd
            valoriTrasaturi.append(trasaturi[0])
            #entropie
            valoriTrasaturi.append(trasaturi[1])
            #rms
            valoriTrasaturi.append(trasaturi[2])
            #deviatie
            valoriTrasaturi.append(trasaturi[3])
            #Se scriu datele in fisier .csv
            fisierTrasaturiCSV.write(str(subiect)+"  "+str(experiment)+"  "+tipBanda+"  "+canal + " " + str(trasaturi[0]) + " " + str(trasaturi[1]) + " " + str(trasaturi[2]) + " " + str(trasaturi[3])+"\n")
        fisierValoriBenziPerUser.close()
    return valoriTrasaturi

def impartireBenzi(subiect, experiment, banda, canal):
    print("Impartire pe benzi de frecventa si calcul de trasaturi pentru subiectul  "+str(subiect))
    valoriTrasaturi = []
    # for tipBanda in ['Alpha', 'Beta', 'Gamma']:
    print(banda)
    # Se verifica tipul benzii si se returneaza valorile low, high si fs ale frecventei
    frecventa_low, frecventa_high, fs = verificaTipBanda(banda)
    # Se selecteaza un subtrial de 6 secunde
    subtrial = selecteazaSubtrialDe6sec(subiect=subiect, experiment=experiment, splits=10)
    valoriBanda = []
    column = subtrial[canal]
    dateFiltrate = aplicaFiltruChebyshevInvers(column, frecventa_low, frecventa_high, fs)
    for val in dateFiltrate:
        valoriBanda.append(round(val, 4))

    return valoriBanda
if __name__ == "__main__":
    trasaturi = impartireBenziFrecventa(subiect=3, experiment=2)
    print("S-au calculat "+len(trasaturi).__str__()+" trasaturi")
    print(trasaturi)







