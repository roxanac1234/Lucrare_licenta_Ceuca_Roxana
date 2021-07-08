import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from CalculTrasaturiSemnalePeBenzi import impartireBenziFrecventa
from  ExtragereDate import filtrareDupaEtichete, binarizeazaDate, extrageDateEtichete
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve

def clasificare_semnale_EEG():
    trasaturi = []
    dateBinarizate = []
    fisierFericireEtichete = open('RezultateClasificare/Fericire/Etichete.csv', 'a')
    fisierFericireEtichete.write("\n")
    fisierTristeteEtichete = open('RezultateClasificare/Tristete/Etichete.csv', 'a')
    fisierTristeteEtichete.write("\n")
    for subiect in range(1, 32):
        # Se realizeaza o filtrare a etichetelor functie de marginea inferioara (3.5) si cea superioara (5.5)
        # Se determina experimentele relevante
        valenta, excitare, experimente =filtrareDupaEtichete(subiect)
        valoriEtichete = extrageDateEtichete(subiect)
        #Se calculeaza trasaturile
        for exp in experimente:
            print("Experiment "+ exp.__str__())
            tr = impartireBenziFrecventa(subiect=subiect,experiment=exp)
            trasaturi.append(tr)
            # Se binarizeaza datele pentru cele 2 etichete: valenta si excitare pentru experimentul exp
            valoarebinara = binarizeazaDate(subiect, 3.5, 5.5, exp)
            dateBinarizate.append(valoarebinara)

            #Se scriu etichetele in folderele emotiilor
            if (valoriEtichete['Valenta'][exp] <= 3.5 and valoriEtichete['Excitare'][exp] <= 3.5) :
                fisierTristeteEtichete.write(str(subiect)+" "+str(exp)+" "+str(valoriEtichete['Valenta'][exp])+" "+str(valoriEtichete['Valenta'][exp])+"\n")
            elif (valoriEtichete['Valenta'][exp] >=5.5 and valoriEtichete['Excitare'][exp] >= 5.5):
                fisierFericireEtichete.write(str(subiect)+" "+str(exp)+" "+str(valoriEtichete['Valenta'][exp])+" "+str(valoriEtichete['Valenta'][exp])+"\n")
    # Lungimea datelor binarizate este egala cu suma numarului de experimente relevante pentru utilizatori
    lungime  = len(dateBinarizate)
    # Pentru fiecare experiment la care este supus utilizatorul se calculeaza 168 de trasaturi (4 trasaturi X 3 benzi X 14 canale selectate)
    trasaturi = np.array(trasaturi).reshape(lungime, 168)
    print(trasaturi)
    print(dateBinarizate)
    # Iesirile vor fi reprezentate de doua clase:
    #0-indica o stare negativa
    #1-indica o stare pozitiva
    dateBinarizate=np.array(dateBinarizate)
    # Se impart datele: 80% pentru antrenare si 20% pentru testare
    # intrarile sunt reprezentate de trasaturile calculate, iar iesirile de valorile binarizate (clase)
    x_antrenare, x_testare, y_antrenare, y_testare = train_test_split(trasaturi, dateBinarizate, test_size=0.20,
                                                                      random_state=0)
    # # Se aplica Support Vector Machine (SVM) cu o functie obiectiv liniara
    # clasificator_SVM = SVC(kernel='linear')
    # # Se incearca potrivirea datelor de antrenare
    # clasificator_SVM.fit(x_antrenare, y_antrenare)
    # # Se face predictia pe datele de testare
    # y_predictie = clasificator_SVM.predict(x_testare)
    # # Se afiseaza matricea de confuzie
    # print(confusion_matrix(y_testare, y_predictie))

    # Se aplica Random Forest
    clasificatorRandomForest = RandomForestClassifier(n_estimators=8, random_state=0,
                                                      max_features='sqrt', criterion='entropy')
    # Antrenarea datelor
    clasificatorRandomForest.fit(x_antrenare, y_antrenare)
    # Predicțiile rezultate
    y_predictii = clasificatorRandomForest.predict(x_testare)
    # Se afiseaza metricile
    print("Folosind Random Forest, s-a obtinut o  acuratete  de ", accuracy_score(y_testare, y_predictii), ".")
    print("F1 score: ", f1_score(y_testare, y_predictii))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_testare, y_predictii)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("Area under curve: ", roc_auc)


if __name__ == '__main__':
    clasificare_semnale_EEG()

