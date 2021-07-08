import pickle
import pandas as pd
#dictionar ce contine denumirea canalelor si indexul in data_preprocessed
dictPozitiiCanale = { 'Cz': 32, 'Fz': 31, 'FC1': 5, 'FC2': 26 , 'Fp1': 1, 'F3': 4,  'Fp2': 30, 'F4': 27,   'O2': 17,'O1': 15, 'P8': 20, 'P7': 11,  'T7': 7 , 'T8': 24 }
#Se selecteaza 16 utilizatori/subiecti
SUBIECTI = [1, 5, 6, 7, 10, 11, 12, 13, 14, 15, 22, 24, 27, 28, 29, 30]
#Canale EEG
CANALE_EEG = ['P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2',
'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1']
#Se selecteaza 14 de canale EEG cele mai relevante pentru analiza starilor emotionale (din cele 32)
CANALE_SELECTATE=['Cz', 'Fz', 'FC1', 'FC2', 'Fp1', 'Fp2', 'F3', 'F4', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8']
numarEtichete, numarExperimente, numarUtilizatori, numarCanale, momenteDeTimp = 4, 40, 32, 40, 8064

#Metoda care extrage valorle corespunzatoare etichetelor din data_preprocessed  pentru un subiect si creaza dataframe-uri folosind biblioteca pandas
def extrageDateEtichete(subiect):
    fisierDataframeEtichete = open("ProcesareDate/DataframeEtichete.dat", 'w')
    if subiect % 1 == 0:
        if subiect < 10:
            numeSubiect = '%0*d' % (2, subiect + 1)
        else:
            numeSubiect = subiect + 1
    numeFisier = "data_preprocessed/s" + str(numeSubiect) + ".dat"
    fisierDataframeEtichete.write("\nSubiect " + subiect.__str__() + "\n")
    dateCuPickle = pickle.load(open(numeFisier, 'rb'), encoding='latin1')
    etichete = dateCuPickle['labels']
    # dictionar de etichete
    dictionarDeEtichete = {'Valenta': list(etichete[:, 0]),
                           'Excitare': list(etichete[:, 1]),
                           'Dominanta': list(etichete[:, 2]),
                           'Placere': list(etichete[:, 3])}
    # Se creaza un dataframe pentru etichete
    dataframeEtichete = pd.DataFrame(dictionarDeEtichete)
    fisierDataframeEtichete.write(dataframeEtichete.__str__())
    return dataframeEtichete

#se vor alege doar valorile mari pentru valenta si arousal (>=5.5), respectiv valorile mici (<=3.5)
#Se vor returna: vectorul cu valori pentru valenta, excitare si cela mai relevante experimente
def filtrareDupaEtichete(subiect):
    fisierFiltrareEtichete = open('ProcesareDate/FiltrareEtichete.dat', 'a')
    valori_valenta = []
    valori_excitare = []
    experimente = []
    dfEtichete = extrageDateEtichete(subiect)
    for experiment in range(numarExperimente):
        if (dfEtichete['Valenta'][experiment] <= 3.5 and dfEtichete['Excitare'][experiment] <= 3.5) or (
                dfEtichete['Valenta'][experiment] >= 5.5 and dfEtichete['Excitare'][experiment] >= 5.5):
            # valorile etichetelor ce se incadreaza in cele 4 intervale se vor stoca in vectorii corespunzatori, de asemenenea si indecsii experimentelor relevante
            valori_valenta.append(dfEtichete['Valenta'][experiment])
            valori_excitare.append(dfEtichete['Excitare'][experiment])
            experimente.append(experiment)
        else:
            dfEtichete = dfEtichete.drop(labels=experiment, axis=0)
    # valorile filtrate vor fi scrise in fisier
    fisierFiltrareEtichete.write("\nSubiect " + subiect.__str__() + "\n")
    fisierFiltrareEtichete.write(dfEtichete.__str__())
    return valori_valenta, valori_excitare, experimente
# Se va realiza o binarizare a datelor pentru etichete in scopul de a putea fi utilizate ca iesiri pentru algoritmul de clasificare
def binarizeazaDate (subiect, margineInf, margineSup, experiment):
    # Se extrag mai intai valorile etichetelor
    dfEtichete = extrageDateEtichete(subiect)
    # Daca valorile pentru valenta si excitare sunt mai mici sau egale cu marginea inferioara (3.5), vor deveni 0
    if dfEtichete['Valenta'][experiment] <= margineInf and dfEtichete['Excitare'][experiment] <= margineInf:
       return 0
    # Daca valorile pentru valenta si excitare sunt mai mari sau egale cu marginea superioara (5.5), vor deveni 1
    elif dfEtichete['Valenta'][experiment] >= margineSup and dfEtichete['Excitare'][experiment] >= margineSup:
        return 1

#Metoda care extrage datele din data_prepreocessed pentru un subiect si returneaza dataframe-ul corespunzator canalelor
def extrageDateCanale(subiect, experiment):
    fisierDataframeCanale = open("ProcesareDate/DataframeCanale.dat", 'a')
    if subiect % 1 == 0:
        if subiect < 10:
            numeSubiect = '%0*d' % (2, subiect + 1)
        else:
            numeSubiect = subiect + 1

    fisierDataframeCanale.write("\nSubiect " + subiect.__str__() + "\n")
    numeFisier = "data_preprocessed/s" + str(numeSubiect) + ".dat"
    # dictionar pentru canale selectate
    dictionarCanale = {'Cz': None,'Fp1': None,'F3': None,'Fp2': None,
                       'F4': None, 'O2': None,'O1': None,'P8': None,
                       'P7': None,'T7': None,'T8': None,'FC1':None,
                       'FC2':None,'Fz':None}
    # Se citesc datele din fisierele .dat folosind metoda load din biblioteca pickle
    dateCuPickle = pickle.load(open(numeFisier, 'rb'), encoding='latin1')
    dateCanale = dateCuPickle['data']
    fisierDataframeCanale.write("EXperiment "+str(experiment)+"\n")
    for canal in CANALE_SELECTATE:
            canalDict = {canal: dateCanale[experiment][dictPozitiiCanale[canal]]}
            dictionarCanale[canal] = canalDict[canal]
        # Se creaza un dataframe pentru canalele selectate
    dataframeCanale = pd.DataFrame(dictionarCanale,
        columns=['Cz', 'Fz', 'FC1', 'FC2', 'Fp1', 'Fp2', 'F3', 'F4', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8'])
    fisierDataframeCanale.write(dataframeCanale.__str__())
    fisierDataframeCanale.close()
    return dataframeCanale

#Metodă pentru selectarea unui subtrial de 6s dintr-un semnal de 60s
def selecteazaSubtrialDe6sec(subiect, experiment, splits=10):
    dictionarSubtrial={ 'Cz': None,  'Fp1': None, 'F3': None,  'Fp2': None,
                       'F4': None,  'O2': None, 'O1': None,   'P8': None,
                       'P7': None,  'T7': None, 'T8': None, 'FC1':None, 'FC2':None, 'Fz':None}
    # Extragem valorile pe canale pentru subiectul si experimentul trimise ca parametri
    dataFrameCanale = extrageDateCanale(subiect, experiment=experiment)
    fisierSubtrial=open("ProcesareDate/FisierSubtrial.dat", 'a')
    fisierSubtrial.write('\n\nSubiect '+subiect.__str__())
    for channel in CANALE_SELECTATE:
        #Se imparte semnalul in 10 intervale  a cate 6 s
        for split in range(0, splits):
            #Se ia un sub-interval de la jumatatea semnalului
            if split == 5:
                    canalDict={channel: dataFrameCanale[channel][split * 768:(split + 1) * 768]}
                    dictionarSubtrial[channel]=canalDict[channel]
    #Se va crea un dataframe care sa contina acest subtrial
    df = pd.DataFrame(dictionarSubtrial, columns=['Cz', 'Fz', 'FC1', 'FC2', 'Fp1', 'Fp2', 'F3', 'F4', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8'])
    fisierSubtrial.write(df.__str__())
    fisierSubtrial.close()
    return df
if __name__ == "__main__":
    df=selecteazaSubtrialDe6sec(subiect=1, experiment=1, splits=10)
    #print(df)
    #extrageDateCanale(subiect =0, experiment=20)
    # extrageDateEtichete(subiect=0)

