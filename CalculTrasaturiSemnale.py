from scipy import signal
import numpy as np
#Metoda care calculeaza deviatia standard
def calculeazaDeviatiaStandard(date):
    semnal = np.asarray(date)
    max = np.max(np.std(semnal))
    return max
#Metoda care calculeaza radacina medie patratica
def calculeazaRMS(date):
     sampling_rate = 128
     frecventa, psd = signal.welch(date, sampling_rate, scaling='spectrum')
     amplitudineRMS = np.sqrt(psd.max())
     return amplitudineRMS
#Metoda care calculeaza densitatea puterii spectrale
def calculeazaDensitateaPuteriiSpectrale(banda, low, high):
    low = 4*low
    high = 4*high
    frecventa, psd = signal.welch(banda,  nperseg=len(banda), scaling='spectrum')
    #Se preiau datele din domeniul de frecventa pentru fiecare banda
    psd = psd[int(low):int(high)]
    return np.max(psd)
#Metodă pentru calcularea entropiei sablon
def calculeazaEntropieSablon(date, m, r = None):
    def determinaDistantaMaxima(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    def determinaPhi(m):
        x = [[date[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for j in range(len(x)) if i != j and determinaDistantaMaxima(x[i], x[j]) <= r])for i in range(len(x))]
        return sum(C)
    N = len(date)
    return -np.log(determinaPhi(m+1) / determinaPhi(m))
#Se calculeaza PSD, RMS, entropia sablon si deviatia standard pentru fiecare banda (ALPHA, BETA si GAMMA)
def calculeazaTrasaturiBanda(valoare, low, high):
    psd = calculeazaDensitateaPuteriiSpectrale(valoare, low, high)
    r = 0.2 * np.std(valoare)
    entropieSablon = calculeazaEntropieSablon(valoare, 2, r)
    rms = calculeazaRMS(valoare)
    deviatieStandard = calculeazaDeviatiaStandard(valoare)
    return [round(psd, 4), round(entropieSablon, 4), round(rms, 4), round(deviatieStandard, 4)]
